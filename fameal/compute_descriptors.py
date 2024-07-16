import ast
import numpy as np
import pandas as pd
import sys

import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import CDPL.Chem as Chem
import CDPL.MolProp as MolProp
import CDPL.ForceField as ForceField


# calculate cpdkit descriptor.
# input using .sdf or text file with smiles
# when using .sdf, the columns should have mol_id, site of metabolism labeled as a list of atom index in "soms"
# when using smiles, the columns should be mol_id, preprocessed_smi

sybyl_atom_type_idx_cpdkit = [
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    38,
    47,
    48,
    49,
    54,
]


class FAMEDescriptors:
    def __init__(self, radius):
        self.radius = radius

    def _perceive_mol(self, mol: Chem.Molecule) -> None:
        Chem.calcImplicitHydrogenCounts(
            mol, False
        )  # calculate implicit hydrogen counts and set corresponding property for all atoms
        Chem.perceiveHybridizationStates(
            mol, False
        )  # perceive atom hybridization states and set corresponding property for all atoms
        Chem.perceiveSSSR(
            mol, False
        )  # perceive smallest set of smallest rings and store as Chem.MolecularGraph property
        Chem.setRingFlags(
            mol, False
        )  # perceive cycles and set corresponding atom and bond properties
        Chem.setAromaticityFlags(
            mol, False
        )  # perceive aromaticity and set corresponding atom and bond properties
        Chem.perceiveSybylAtomTypes(
            mol, False
        )  # perceive Sybyl atom types and set corresponding property for all atoms
        Chem.calcTopologicalDistanceMatrix(
            mol, False
        )  # calculate topological distance matrix and store as Chem.MolecularGraph property
        # (required for effective polarizability calculations)
        Chem.perceivePiElectronSystems(
            mol, False
        )  # perceive pi electron systems and store info as Chem.MolecularGraph property
        # (required for MHMO calculations)

        MolProp.calcPEOEProperties(mol, False)
        # calculate sigma charges and electronegativities
        # using the PEOE method and store values as atom properties
        # (prerequisite for MHMO calculations)

        MolProp.calcMHMOProperties(mol, False)
        # calculate pi charges, electronegativities and other properties
        # by a modified Hueckel MO method and store values as properties

        ForceField.perceiveMMFF94AromaticRings(
            mol, False
        )  # perceive aromatic rings according to the MMFF94 aroamticity model and store data as Chem.MolecularGraph property
        ForceField.assignMMFF94AtomTypes(
            mol, False, False
        )  # perceive MMFF94 atom types (tolerant mode) set corresponding property for all atoms
        ForceField.assignMMFF94BondTypeIndices(
            mol, False, False
        )  # perceive MMFF94 bond types (tolerant mode) set corresponding property for all bonds
        ForceField.calcMMFF94AtomCharges(
            mol, False, False
        )  # calculate MMFF94 atom charges (tolerant mode) set corresponding property for all atoms

    def _process_molecule(self, mol: Chem.Molecule, has_soms: bool) -> None:
        if not Chem.hasStructureData(mol):
            print(
                "Error: structure data not available for molecule '%s'!"
                % Chem.getName(mol)
            )
            return

        struct_data = Chem.getStructureData(mol)

        for entry in struct_data:
            if "mol_id" in entry.header:
                mol_id = entry.data
            if "soms" in entry.header:
                soms = ast.literal_eval(entry.data)

        self._perceive_mol(mol)

        property_dict = {}
        for atom in mol.atoms:
            if Chem.getSybylType(atom) == 24:
                continue  # remove hydrogen this way to not mess up the atom index with SoMs

            atom_id = mol.getAtomIndex(atom)
            descriptor_names, descriptors = self._generate_FAME_descriptors(
                atom, mol, self.radius
            )

            if has_soms:
                if atom_id in soms:
                    som_label = 1
                else:
                    som_label = 0
            else:
                som_label = None

            property_dict[(mol_id, atom_id)] = (som_label, descriptors)

        return descriptor_names, property_dict

    def _generate_FAME_descriptors(
        self, ctr_atom: Chem.Atom, molgraph: Chem.MolecularGraph, radius: int
    ) -> tuple:
        properties_dict = {
            Chem.getSybylType: "AtomType",
            MolProp.getHeavyAtomCount: "AtomDegree",
            MolProp.getHybridPolarizability: "HybridPolarizability",
            MolProp.getVSEPRCoordinationGeometry: "VSEPRgeometry",
            MolProp.calcExplicitValence: "AtomValence",
            MolProp.calcEffectivePolarizability: "EffectivePolarizability",
            MolProp.getPEOESigmaCharge: "SigmaCharge",
            ForceField.getMMFF94Charge: "MMFF94Charge",
            MolProp.calcPiElectronegativity: "PiElectronegativity",
            MolProp.getPEOESigmaElectronegativity: "SigmaElectronegativity",
            MolProp.calcInductiveEffect: "InductiveEffect",
        }

        # Sybyl atom types to keep
        sybyl_atom_type_idx_cpdkit = [
            1,
            2,
            3,
            4,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            18,
            19,
            20,
            21,
            22,
            23,
            38,
            47,
            48,
            49,
            54,
        ]

        (fs, names) = zip(*properties_dict.items())
        descriptor_names = [
            prefix + "_" + Chem.getSybylAtomTypeString(i) + "_" + str(j)
            for prefix in names
            for j in range(radius + 1)
            for i in sybyl_atom_type_idx_cpdkit
        ]
        descr = np.zeros(
            (len(properties_dict), len(sybyl_atom_type_idx_cpdkit) * (radius + 1)),
            dtype=float,
        )

        env = Chem.Fragment()  # for storing of extracted environment atoms

        Chem.getEnvironment(
            ctr_atom, molgraph, radius, env
        )  # extract environment of center atom reaching out up to 'radius' bonds
        for atom in env.atoms:  # iterate over extracted environment atoms
            sybyl_type = Chem.getSybylType(
                atom
            )  # retrieve Sybyl type of environment atom
            if sybyl_type not in sybyl_atom_type_idx_cpdkit:
                continue

            for c, t in zip(
                range(len(sybyl_atom_type_idx_cpdkit)), sybyl_atom_type_idx_cpdkit
            ):
                if sybyl_type == t:
                    position = c

            top_dist = Chem.getTopologicalDistance(
                ctr_atom, atom, molgraph
            )  # get top. distance between center atom and environment atom
            descr[
                0, (top_dist * len(sybyl_atom_type_idx_cpdkit) + position)
            ] += 1  # instead of 1 (= Sybyl type presence) also any other numeric atom

            # for properties
            for i, f_, name in zip(range(len(fs)), fs, names):
                if name == "AtomType":
                    continue
                if name in ["SigmaCharge", "MMFF94Charge", "SigmaElectronegativity"]:
                    prop = f_(atom)
                else:
                    prop = f_(atom, molgraph)
                descr[
                    (i, (top_dist * len(sybyl_atom_type_idx_cpdkit) + position))
                ] += prop  # sum up property

        for i in range(len(fs) - 1):
            descr[i + 1, :] = np.divide(
                descr[i + 1, :],
                descr[0, :],
                out=np.zeros_like(descr[i + 1, :]),
                where=descr[0, :] != 0,
            )  # averaging property and when divide by 0 give 0

        # calculate max_top_dist, the longest distance in a molecules, independent from atoms
        max_top_dist = 0
        for atom1 in molgraph.atoms:
            for atom2 in molgraph.atoms:
                distance = Chem.getTopologicalDistance(atom1, atom2, molgraph)
                if distance > max_top_dist:
                    max_top_dist = distance

        # calculate max_distance_center, the longest distance between this center atom and other
        max_distance_center = 0
        for atom in molgraph.atoms:
            distance = Chem.getTopologicalDistance(ctr_atom, atom, molgraph)
            if distance > max_distance_center:
                max_distance_center = distance

        # add the 4 descriptors related to topological distance
        descriptor_names = np.append(
            descriptor_names,
            [
                "longestMaxTopDistinMolecule",
                "highestMaxTopDistinMatrixRow",
                "diffSPAN",
                "refSPAN",
            ],
        )
        descr = descr.flatten()
        if max_top_dist == 0:  # had compound with one heavy atom, lazy solution now
            descr = np.append(
                descr,
                [
                    max_top_dist,
                    max_distance_center,
                    max_top_dist - max_distance_center,
                    0 / 1,
                ],
            )
        else:
            descr = np.append(
                descr,
                [
                    max_top_dist,
                    max_distance_center,
                    max_top_dist - max_distance_center,
                    max_distance_center / max_top_dist,
                ],
            )

        descr = descr.round(4)
        return descriptor_names, descr

    def compute_FAME_descriptors(self, in_file: str, has_soms: bool) -> tuple:
        reader = Chem.FileSDFMoleculeReader(in_file)
        mol = Chem.BasicMolecule()

        mol_ids = []
        atom_ids = []
        som_labels = []
        descriptors_lst = []

        # read and process molecules one after the other until the end of input has been reached
        try:
            while reader.read(mol):
                try:
                    self._perceive_mol(mol)
                    uncommon_element = 0
                    for atom in mol.atoms:
                        atom_type = Chem.getSybylType(atom)
                        if atom_type not in sybyl_atom_type_idx_cpdkit:
                            uncommon_element = True

                    if uncommon_element:
                        continue

                    else:
                        descriptor_names, property_dict = self._process_molecule(
                            mol, has_soms
                        )

                        for key, value in property_dict.items():
                            som_label, descriptors = value
                            mol_id, atom_id = key

                            mol_ids.append(mol_id)
                            atom_ids.append(atom_id)
                            som_labels.append(som_label)
                            descriptors_lst.append(list(descriptors))

                except Exception as e:
                    sys.exit("Error: processing of molecule failed:\n" + str(e))

        except Exception as e:
            sys.exit("Error: reading molecule failed:\n" + str(e))

        if has_soms:

            return (
                np.array(mol_ids, dtype=int),
                np.array(atom_ids, dtype=int),
                np.array(som_labels, dtype=int),
                np.array(descriptors_lst, dtype=float),
            )

        else: 
            return (
                np.array(mol_ids, dtype=int),
                np.array(atom_ids, dtype=int),
                None,
                np.array(descriptors_lst, dtype=float),
            )