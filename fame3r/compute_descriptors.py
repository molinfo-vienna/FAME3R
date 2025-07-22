"""Module for computing FAME descriptors for a given molecule."""

import ast
import csv
import os
import sys
from pathlib import Path

import numpy as np
from CDPL import Chem, ForceField, MolProp

SYBYL_ATOM_TYPE_IDX_CDPKIT = [
    1, # C.3 - sp3 carbon
    2, # C.2 - sp2 carbon
    3, # C.1 - sp carbon
    4, # C.ar -  aromatic carbon
    6, # N.3 - sp3 nitrogen
    7, # N.2 - sp2 nitrogen
    8, # N.1 - sp nitrogen
    9, # N.ar - aromatic nitrogen
    10, # N.am - amide nitrogen
    11, # N.pl3 - trigonal nitrogen
    12, # N.4 - quaternary nitrogen
    13, # O.3 - sp3 oxygen
    14, # O.2 - sp2 oxygen
    15, # O.co2 - carboxylic oxygen
    18, # S.3 - sp3 sulfur
    19, # S.2 - sp2 sulfur
    20, # S.0 - sulfoxide sulfur
    21, # S.O2 - sulfone sulfur
    22, # P.3 - sp3 phosphorus
    23, # F - fluorine
    24, # H - hydrogen
    38, # Si - silicon
    47, # Cl - chlorine
    48, # Br - bromine
    49, # I - iodine
    54, # B - boron
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
        struct_data = Chem.getStructureData(mol) if Chem.hasStructureData(mol) else []
        data_dict = {}

        for entry in struct_data:
            if "soms" in entry.header:
                data_dict["soms"] = ast.literal_eval(entry.data)

        return data_dict


class DescriptorGenerator:
    """Class for generating descriptors for a given atom in a molecule."""

    def __init__(self, radius):
        """Initialize the DescriptorGenerator class."""
        self.radius = radius

    def generate_descriptors(
        self, ctr_atom: Chem.Atom, molgraph: Chem.MolecularGraph
    ) -> tuple[list[str], np.ndarray]:
        """Generate descriptors for a given atom in a molecule.

        Args:
            ctr_atom (Chem.Atom): center atom
            molgraph (Chem.MolecularGraph): molecular graph

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
        
        # Create descriptor names
        descriptor_names = []
        for prefix in names:
            if prefix == "AtomType":
                # Circular fingerprints for all radii (0 to radius)
                for j in range(self.radius + 1):
                    for i in SYBYL_ATOM_TYPE_IDX_CDPKIT:
                        for k in range(32):
                            descriptor_names.append(f"R{j}_{prefix}_{Chem.getSybylAtomTypeString(i)}_B{k}")
            else:
                # Physicochemical property descriptors only for center atom (radius 0)
                descriptor_names.append(f"{prefix}")
        
        # Calculate total descriptor size
        fingerprints_size = len(SYBYL_ATOM_TYPE_IDX_CDPKIT) * (self.radius + 1) * 32
        physchem_descriptor_size = len(properties_dict) - 1
        total_descriptor_size = fingerprints_size + physchem_descriptor_size
        
        descr = np.zeros(total_descriptor_size, dtype=float)
        
        # Get the chemical environment around the center atom
        env = Chem.Fragment()
        Chem.getEnvironment(ctr_atom, molgraph, self.radius, env)
        
        # Initialize circular fingerprints
        fingerprints = np.zeros(fingerprints_size, dtype=float)
        
        # Count atoms of each type at each distance
        atom_counts = np.zeros((len(SYBYL_ATOM_TYPE_IDX_CDPKIT), self.radius + 1), dtype=int)
        
        for atom in env.atoms:
            sybyl_type = Chem.getSybylType(atom)
            if sybyl_type not in SYBYL_ATOM_TYPE_IDX_CDPKIT:
                continue

            sybyl_type_index = SYBYL_ATOM_TYPE_IDX_CDPKIT.index(sybyl_type)
            radius = Chem.getTopologicalDistance(ctr_atom, atom, molgraph)
            atom_counts[sybyl_type_index, radius] += 1
        
        # Generate 32-bit fingerprints for each combination of atom type and distance
        fingerprint_index = 0
        for radius in range(self.radius + 1):  # Radius (R0, R1, ..., R5)
            for sybyl_type_index in range(len(SYBYL_ATOM_TYPE_IDX_CDPKIT)):  # Atom type
                for bit in range(32):  # Bit position (B0, B1, ..., B31)
                    count = atom_counts[sybyl_type_index, radius]
                    # Set bit to 1 if count > bit position
                    if count > bit:
                        fingerprints[fingerprint_index] = 1
                    fingerprint_index += 1
        
        # Copy circular fingerprints to the main descriptor array
        descr[:fingerprints_size] = fingerprints
        
        # Initialize physicochemical property descriptors (only for center atom)
        physchem_descr = np.zeros(physchem_descriptor_size, dtype=float)
        
        # Calculate physicochemical properties for the center atom
        for i, (f_, name) in enumerate(zip(fs[1:], names[1:])):  # Skip AtomType
            prop = (
                f_(ctr_atom)
                if name in ["SigmaCharge", "MMFF94Charge", "SigmaElectronegativity"]
                else f_(ctr_atom, molgraph)
            )
            physchem_descr[i] = prop
        
        # Copy physicochemical property descriptors to the main descriptor array
        descr[fingerprints_size:] = physchem_descr.flatten()

        # Add topological descriptors
        max_top_dist, max_distance_center = self._calculate_distances(
            ctr_atom,
            molgraph,
        )
        descriptor_names += [
            "longestMaxTopDistinMolecule",
            "highestMaxTopDistinMatrixRow",
            "diffSPAN",
            "refSPAN",
        ]

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
        self,
        ctr_atom: Chem.Atom,
        molgraph: Chem.MolecularGraph,
    ) -> tuple:
        """Calculate the maximum topological distance across \
            the whole molecule and the maximum distance from the center atom.

        Args:
            ctr_atom (Chem.Atom): center atom
            molgraph (Chem.MolecularGraph): molecular graph

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
        """Initialize the FAMEDescriptors class."""
        self.radius = radius
        self.descriptor_generator = DescriptorGenerator(radius)

    def _process_molecule(
        self, mol: Chem.Molecule, has_soms: bool
    ) -> tuple[tuple, dict]:
        """Process a molecule and generate descriptors for each atom.

        Args:
            mol (Chem.Molecule): molecule
            has_soms (bool): whether the input file contains site of metabolism labels

        Returns:
            tuple: (descriptor_names, property_dict)
        """
        data_dict = MoleculeProcessor.extract_structure_data(mol)
        mol_id = Chem.getName(mol)
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
    ) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray | None, np.ndarray]:
        """Compute FAME descriptors for a given molecule.

        Args:
            in_file (str): input file path
            out_folder (str): output folder path
            has_soms (bool): whether the input file contains site of metabolism labels

        Returns:
            tuple: (mol_num_ids, mol_ids, atom_ids, som_labels, descriptors_lst)
        """
        in_file_path, out_folder_path = Path(in_file), Path(out_folder)
        out_not_calculated, out_descriptors = self._get_output_file_paths(
            in_file_path, out_folder_path
        )

        if not os.path.exists(out_not_calculated) and not os.path.exists(
            out_descriptors
        ):
            return self._compute_and_save_descriptors(
                in_file, out_not_calculated, out_descriptors, has_soms
            )
        return self._read_precomputed_descriptors(out_descriptors, has_soms)

    def _get_output_file_paths(
        self, in_file_path: Path, out_folder_path: Path
    ) -> tuple[Path, Path]:
        """Generate output file paths for descriptors and not-calculated compounds."""
        out_not_calculated = (
            out_folder_path
            / f"{in_file_path.stem}_{self.radius}_not_calculated_cpds.csv"
        )
        out_descriptors = (
            out_folder_path / f"{in_file_path.stem}_{self.radius}_descriptors.csv"
        )
        return out_not_calculated, out_descriptors

    def _compute_and_save_descriptors(
        self,
        in_file: str,
        out_not_calculated: Path,
        out_descriptors: Path,
        has_soms: bool,
    ) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray | None, np.ndarray]:
        """Compute and save descriptors from molecules."""
        reader = Chem.FileSDFMoleculeReader(in_file)
        mol = Chem.BasicMolecule()

        mol_num_ids: list[int] = []
        mol_ids: list[str] = []
        atom_ids: list[int] = []
        som_labels: list[int] = []
        descriptors_lst: list[list[float]] = []

        with (
            open(out_not_calculated, "w", encoding="UTF-8") as f_not_calc_writer,
            open(out_descriptors, "w", encoding="UTF-8") as f_desc_writer,
        ):
            f_not_calc_csv = csv.writer(f_not_calc_writer)
            f_desc_csv = csv.writer(f_desc_writer)

            f_not_calc_csv.writerow(["sybyl_atom_type_id", "sybyl_atom_type", "mol_id"])

            i = 0
            try:
                while reader.read(mol):
                    try:
                        MoleculeProcessor.perceive_mol(mol)
                        if self._has_uncommon_element(mol, f_not_calc_csv):
                            continue

                        descriptor_names, property_dict = self._process_molecule(
                            mol, has_soms
                        )

                        if i == 0:
                            f_desc_csv.writerow(
                                ["mol_num_id", "mol_id", "atom_id", "som_label"]
                                + list(descriptor_names)
                            )
                        i += 1

                        for (mol_id, atom_id), (
                            som_label,
                            descriptors,
                        ) in property_dict.items():
                            mol_num_ids.append(i)
                            mol_ids.append(mol_id)
                            atom_ids.append(atom_id)
                            som_labels.append(som_label)
                            descriptors_lst.append(list(descriptors))

                            f_desc_csv.writerow(
                                [i, mol_id, atom_id, som_label] + list(descriptors)
                            )
                    except RuntimeError as e:
                        sys.exit(f"Error: processing of molecule failed:\n{e}")
            except RuntimeError as e:
                sys.exit(f"Error: reading molecule failed:\n{e}")

            return (
                np.array(mol_num_ids, int),
                mol_ids,
                np.array(atom_ids, int),
                np.array(som_labels, int) if has_soms else None,
                np.array(descriptors_lst, float),
            )

    def _has_uncommon_element(self, mol, f_not_calc) -> bool:
        """Check if a molecule contains uncommon elements and log them."""
        uncommon_element = False
        for atom in mol.atoms:
            atom_type = Chem.getSybylType(atom)
            if atom_type not in SYBYL_ATOM_TYPE_IDX_CDPKIT:
                f_not_calc.writerow([atom_type, Chem.getSybylAtomTypeString(atom_type)])
                uncommon_element = True
        return uncommon_element

    def _read_precomputed_descriptors(
        self, out_descriptors: Path, has_soms: bool
    ) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray | None, np.ndarray]:
        """Read previously computed descriptors from a file."""
        print(f"Reading pre-calculated descriptors from {out_descriptors}")

        mol_num_ids, mol_ids, atom_ids, som_labels, descriptors_lst = [], [], [], [], []

        with open(out_descriptors, "r", encoding="UTF-8") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(",")
                mol_num_id, mol_id, atom_id, som_label = parts[:4]
                descriptors = list(map(float, parts[4:]))
                mol_num_ids.append(mol_num_id)
                mol_ids.append(mol_id)
                atom_ids.append(atom_id)
                som_labels.append(som_label)
                descriptors_lst.append(descriptors)

        return (
            np.array(mol_num_ids, int),
            mol_ids,
            np.array(atom_ids, int),
            np.array(som_labels, int) if has_soms else None,
            np.array(descriptors_lst, float),
        )
