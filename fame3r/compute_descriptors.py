# pylint: disable=C0114,E1101,R0903,R0914

import ast
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from CDPL import Chem, ForceField, MolProp

# Suppress pandas performance warning
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

SYBYL_ATOM_TYPE_IDX_CDPKIT = [
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


class MoleculeProcessor:
    """Class for processing molecules and extracting structure data."""

    @staticmethod
    def perceive_mol(mol: Chem.Molecule) -> None:
        """Perceive various molecular properties and set corresponding attributes."""
        Chem.calcImplicitHydrogenCounts(mol, False)
        Chem.perceiveHybridizationStates(mol, False)
        Chem.perceiveSSSR(mol, False)
        Chem.setRingFlags(mol, False)
        Chem.setAromaticityFlags(mol, False)
        Chem.perceiveSybylAtomTypes(mol, False)
        Chem.calcTopologicalDistanceMatrix(mol, False)
        Chem.perceivePiElectronSystems(mol, False)

        MolProp.calcPEOEProperties(mol, False)
        MolProp.calcMHMOProperties(mol, False)

        ForceField.perceiveMMFF94AromaticRings(mol, False)
        ForceField.assignMMFF94AtomTypes(mol, False, False)
        ForceField.assignMMFF94BondTypeIndices(mol, False, False)
        ForceField.calcMMFF94AtomCharges(mol, False, False)

    @staticmethod
    def extract_structure_data(mol: Chem.Molecule) -> dict:
        """Extract structure data from a molecule."""
        struct_data = Chem.getStructureData(mol)
        data_dict = {}

        for entry in struct_data:
            if "mol_id" in entry.header:
                data_dict["mol_id"] = entry.data
            if "soms" in entry.header:
                data_dict["soms"] = ast.literal_eval(entry.data)

        return data_dict


class DescriptorGenerator:
    """Class for generating descriptors for a given atom in a molecule."""

    def __init__(self, radius):
        self.radius = radius

    def generate_descriptors(
        self, ctr_atom: Chem.Atom, molgraph: Chem.MolecularGraph
    ) -> tuple:
        """Generate descriptors for a given atom in a molecule.

        Args:
            ctr_atom (Chem.Atom):
            molgraph (Chem.MolecularGraph):

        Returns:
            tuple: (descriptor_names, descriptors)
        """
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

        fs, names = zip(*properties_dict.items())
        descriptor_names = [
            f"{prefix}_{Chem.getSybylAtomTypeString(i)}_{j}"
            for prefix in names
            for j in range(self.radius + 1)
            for i in SYBYL_ATOM_TYPE_IDX_CDPKIT
        ]
        descr = np.zeros(
            (len(properties_dict), len(SYBYL_ATOM_TYPE_IDX_CDPKIT) * (self.radius + 1)),
            dtype=float,
        )

        env = Chem.Fragment()
        Chem.getEnvironment(ctr_atom, molgraph, self.radius, env)

        for atom in env.atoms:
            sybyl_type = Chem.getSybylType(atom)
            if sybyl_type not in SYBYL_ATOM_TYPE_IDX_CDPKIT:
                continue

            position = SYBYL_ATOM_TYPE_IDX_CDPKIT.index(sybyl_type)
            top_dist = Chem.getTopologicalDistance(ctr_atom, atom, molgraph)
            descr[0, (top_dist * len(SYBYL_ATOM_TYPE_IDX_CDPKIT) + position)] += 1

            for i, f_, name in zip(range(len(fs)), fs, names):
                if name == "AtomType":
                    continue
                prop = (
                    f_(atom)
                    if name in ["SigmaCharge", "MMFF94Charge", "SigmaElectronegativity"]
                    else f_(atom, molgraph)
                )
                descr[
                    i, (top_dist * len(SYBYL_ATOM_TYPE_IDX_CDPKIT) + position)
                ] += prop

        for i in range(1, len(fs)):
            # averaging property and when divide by 0 give 0
            descr[i, :] = np.divide(
                descr[i, :],
                descr[0, :],
                out=np.zeros_like(descr[i, :]),
                where=descr[0, :] != 0,
            )

        max_top_dist, max_distance_center = self._calculate_distances(
            molgraph, ctr_atom
        )
        descriptor_names += [
            "longestMaxTopDistinMolecule",
            "highestMaxTopDistinMatrixRow",
            "diffSPAN",
            "refSPAN",
        ]
        descr = descr.flatten()

        if max_top_dist == 0:
            descr = np.append(
                descr,
                [
                    max_top_dist,
                    max_distance_center,
                    max_top_dist - max_distance_center,
                    0,
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

    def _calculate_distances(
        self, molgraph: Chem.MolecularGraph, ctr_atom: Chem.Atom
    ) -> tuple:
        """Calculate the maximum topological distance across \
            the whole molecule and the maximum distance from the center atom.

        Args:
            molgraph (Chem.MolecularGraph):
            ctr_atom (Chem.Atom):

        Returns:
            tuple: (max_top_dist, max_distance_center)
        """
        max_top_dist = max( 
            Chem.getTopologicalDistance(atom1, atom2, molgraph)
            for atom1 in molgraph.atoms
            for atom2 in molgraph.atoms
        )
        max_distance_center = max(
            Chem.getTopologicalDistance(ctr_atom, atom, molgraph)
            for atom in molgraph.atoms
        )
        return max_top_dist, max_distance_center


class FAMEDescriptors:
    """Class for computing FAME descriptors for a given molecule."""

    def __init__(self, radius):
        self.radius = radius
        self.descriptor_generator = DescriptorGenerator(radius)

    def _process_molecule(self, mol: Chem.Molecule, has_soms: bool) -> tuple:
        """Process a molecule and generate descriptors for each atom.

        Args:
            mol (Chem.Molecule):
            has_soms (bool): Whether the input file contains site of metabolism labels.

        Returns:
            tuple: (descriptor_names, property_dict)
        """
        data_dict = MoleculeProcessor.extract_structure_data(mol)
        mol_id = data_dict.get("mol_id")
        soms = data_dict.get("soms", [])

        MoleculeProcessor.perceive_mol(mol)
        property_dict = {}

        for atom in mol.atoms:
            if Chem.getSybylType(atom) == 24:
                continue  # Skip hydrogen atoms

            atom_id = mol.getAtomIndex(atom)
            (
                descriptor_names,
                descriptors,
            ) = self.descriptor_generator.generate_descriptors(atom, mol)
            som_label = 1 if has_soms and atom_id in soms else 0 if has_soms else None
            property_dict[(mol_id, atom_id)] = (som_label, descriptors)

        return descriptor_names, property_dict

    def compute_fame_descriptors(
        self, in_file: str, out_folder: str, has_soms: bool
    ) -> tuple:
        """Compute FAME descriptors for a given molecule.

        Args:
            in_file (str): Input file path.
            out_folder (str): Output folder path.
            has_soms (bool): Whether the input file contains site of metabolism labels.

        Returns:
            tuple: (mol_ids, atom_ids, som_labels, descriptors_lst)
        """
        in_file_path = Path(in_file)
        out_folder_path = Path(out_folder)

        reader = Chem.FileSDFMoleculeReader(in_file)
        mol = Chem.BasicMolecule()

        mol_ids, atom_ids, som_labels, descriptors_lst = [], [], [], []

        base_filename = in_file_path.stem  # This gets the filename without extension

        out_not_calculated_cpds = (
            out_folder_path / f"{base_filename}_{self.radius}_not_calculated_cpds.csv"
        )
        out_descriptors = (
            out_folder_path / f"{base_filename}_{self.radius}_descriptors.csv"
        )

        if not os.path.exists(out_not_calculated_cpds) and not os.path.exists(
            out_descriptors
        ):
            with (
                open(out_not_calculated_cpds, "w", encoding="UTF-8") as f_not_calc,
                open(out_descriptors, "w", encoding="UTF-8") as f_desc,
            ):
                f_not_calc.write("sybyl_atom_type_id,sybyl_atom_type,mol_id\n")

                i = 0
                try:
                    while reader.read(mol):
                        try:
                            MoleculeProcessor.perceive_mol(mol)
                            uncommon_element = False
                            for atom in mol.atoms:
                                atom_type = Chem.getSybylType(atom)
                                if atom_type not in SYBYL_ATOM_TYPE_IDX_CDPKIT:
                                    f_not_calc.write(
                                        f"{atom_type},{Chem.getSybylAtomTypeString(atom_type)},X\n"
                                    )
                                    uncommon_element = True

                            if uncommon_element:
                                continue

                            descriptor_names, property_dict = self._process_molecule(
                                mol, has_soms
                            )

                            if i == 0:
                                f_desc.write(
                                    "som_label,mol_id,atom_id,"
                                    + ",".join(descriptor_names)
                                    + "\n"
                                )
                            i += 1

                            for (mol_id, atom_id), (
                                som_label,
                                descriptors,
                            ) in property_dict.items():
                                mol_ids.append(mol_id)
                                atom_ids.append(atom_id)
                                som_labels.append(som_label)
                                descriptors_lst.append(list(descriptors))

                                f_desc.write(
                                    f"{som_label},{mol_id},{atom_id},"
                                    + ",".join(map(str, descriptors))
                                    + "\n"
                                )

                        except RuntimeError as e:
                            sys.exit(f"Error: processing of molecule failed:\n{e}")

                except RuntimeError as e:
                    sys.exit(f"Error: reading molecule failed:\n{e}")

        else:
            print(f"Reading pre-calculated descriptors from {out_descriptors}")
            with open(out_descriptors, "r", encoding="UTF-8") as f:
                next(f)
                for line in f:
                    parts = line.strip().split(",")
                    som_label, mol_id, atom_id = map(int, parts[:3])
                    descriptors = list(map(float, parts[3:]))
                    mol_ids.append(mol_id)
                    atom_ids.append(atom_id)
                    som_labels.append(som_label)
                    descriptors_lst.append(descriptors)

        if has_soms:
            return (
                np.array(mol_ids, dtype=int),
                np.array(atom_ids, dtype=int),
                np.array(som_labels, dtype=int),
                np.array(descriptors_lst, dtype=float),
            )

        return (
            np.array(mol_ids, dtype=int),
            np.array(atom_ids, dtype=int),
            None,
            np.array(descriptors_lst, dtype=float),
        )
