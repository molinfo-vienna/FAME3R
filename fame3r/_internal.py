# pyright: reportAttributeAccessIssue=false


import numpy as np
from CDPL import Chem, ForceField, MolProp

SYBYL_ATOM_TYPE_IDX_CDPKIT = [
    1,  ## C.3   - sp3 carbon
    2,  ## C.2   - sp2 carbon
    3,  ## C.1   - sp carbon
    4,  ## C.ar  - aromatic carbon
    6,  ## N.3   - sp3 nitrogen
    7,  ## N.2   - sp2 nitrogen
    8,  ## N.1   - sp nitrogen
    9,  ## N.ar  - aromatic nitrogen
    10,  # N.am  - amide nitrogen
    11,  # N.pl3 - trigonal nitrogen
    12,  # N.4   - quaternary nitrogen
    13,  # O.3   - sp3 oxygen
    14,  # O.2   - sp2 oxygen
    15,  # O.co2 - carboxylic oxygen
    18,  # S.3   - sp3 sulfur
    19,  # S.2   - sp2 sulfur
    20,  # S.0   - sulfoxide sulfur
    21,  # S.O2  - sulfone sulfur
    22,  # P.3   - sp3 phosphorus
    23,  # F     - fluorine
    24,  # H     - hydrogen
    38,  # Si    - silicon
    47,  # Cl    - chlorine
    48,  # Br    - bromine
    49,  # I     - iodine
    54,  # B     - boron
]


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
        _prepare_mol(molgraph)

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
                            descriptor_names.append(
                                f"R{j}_{prefix}_{Chem.getSybylAtomTypeString(i)}_B{k}"
                            )
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
        atom_counts = np.zeros(
            (len(SYBYL_ATOM_TYPE_IDX_CDPKIT), self.radius + 1), dtype=int
        )

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
        max_top_dist = _max_topological_distance(molgraph)
        max_distance_center = _max_distance_from_reference(molgraph, ctr_atom)
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


def _prepare_mol(mol: Chem.Molecule) -> None:
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


def _max_topological_distance(molgraph: Chem.MolecularGraph) -> float:
    return max(
        Chem.getTopologicalDistance(atom1, atom2, molgraph)
        for atom1 in molgraph.atoms
        for atom2 in molgraph.atoms
    )


def _max_distance_from_reference(
    molgraph: Chem.MolecularGraph, ref_atom: Chem.Atom
) -> float:
    return max(
        Chem.getTopologicalDistance(ref_atom, atom, molgraph) for atom in molgraph.atoms
    )
