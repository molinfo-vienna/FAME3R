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

    def generate_fp(self, ctr_atom: Chem.Atom, molgraph: Chem.MolecularGraph):
        _prepare_mol(molgraph)

        # Create descriptor names
        descriptor_names = []
        for radius in range(self.radius + 1):
            for atom_type in SYBYL_ATOM_TYPE_IDX_CDPKIT:
                for bit in range(32):
                    descriptor_names.append(
                        f"R{radius}_AtomType_{Chem.getSybylAtomTypeString(atom_type)}_B{bit}"
                    )

        # Calculate total descriptor size
        fingerprints_size = len(descriptor_names)

        # Get the chemical environment around the center atom
        env = Chem.Fragment()
        Chem.getEnvironment(ctr_atom, molgraph, self.radius, env)

        # Initialize circular fingerprints
        fingerprints = np.zeros(fingerprints_size, dtype=bool)

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

        return descriptor_names, fingerprints

    def generate_pchem(self, ctr_atom: Chem.Atom, molgraph: Chem.MolecularGraph):
        _prepare_mol(molgraph)

        descriptors = {
            "AtomDegree": MolProp.getHeavyAtomCount(ctr_atom),
            "HybridPolarizability": MolProp.getHybridPolarizability(ctr_atom, molgraph),
            "VSEPRgeometry": MolProp.getVSEPRCoordinationGeometry(ctr_atom, molgraph),
            "AtomValence": MolProp.calcExplicitValence(ctr_atom, molgraph),
            "EffectivePolarizability": MolProp.calcEffectivePolarizability(
                ctr_atom, molgraph
            ),
            "SigmaCharge": MolProp.getPEOESigmaCharge(ctr_atom),
            "MMFF94Charge": ForceField.getMMFF94Charge(ctr_atom),
            "PiElectronegativity": MolProp.calcPiElectronegativity(ctr_atom, molgraph),
            "SigmaElectronegativity": MolProp.getPEOESigmaElectronegativity(
                ctr_atom,
            ),
            "InductiveEffect": MolProp.calcInductiveEffect(ctr_atom, molgraph),
        }

        return list(descriptors.keys()), np.array(list(descriptors.values())).round(4)

    def generate_topo(self, ctr_atom: Chem.Atom, molgraph: Chem.MolecularGraph):
        _prepare_mol(molgraph)

        max_topo_dist = _max_topological_distance(molgraph)
        max_dist_center = _max_distance_from_reference(molgraph, ctr_atom)

        descriptors = {
            "longestMaxTopDistinMolecule": max_topo_dist,
            "highestMaxTopDistinMatrixRow": max_dist_center,
            "diffSPAN": max_topo_dist - max_dist_center,
            "refSPAN": max_dist_center / max_topo_dist if max_topo_dist != 0 else 0,
        }

        return list(descriptors.keys()), np.array(list(descriptors.values())).round(4)

    def generate_descriptors(
        self, ctr_atom: Chem.Atom, molgraph: Chem.MolecularGraph
    ) -> tuple[list[str], np.ndarray]:
        full_descriptor_names = []
        full_descriptors = np.array([], dtype=float)

        for f in [self.generate_fp, self.generate_pchem, self.generate_topo]:
            descriptor_names, descriptors = f(ctr_atom, molgraph)

            full_descriptor_names += descriptor_names
            full_descriptors = np.concatenate((full_descriptors, descriptors))

        return full_descriptor_names, full_descriptors


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
