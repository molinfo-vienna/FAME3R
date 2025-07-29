# pyright: reportAttributeAccessIssue=false

import numpy as np
import numpy.typing as npt
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


def generate_fingerprint_names(radius: int) -> list[str]:
    descriptor_names = []
    for radius in range(radius + 1):
        for atom_type in SYBYL_ATOM_TYPE_IDX_CDPKIT:
            for bit in range(32):
                descriptor_names.append(
                    f"R{radius}_AtomType_{Chem.getSybylAtomTypeString(atom_type)}_B{bit}"
                )

    return descriptor_names


def generate_fingerprints(
    ctr_atom: Chem.Atom,
    molgraph: Chem.MolecularGraph,
    radius: int,
) -> npt.NDArray[np.bool]:
    _prepare_mol(molgraph)

    # Calculate total descriptor size
    fingerprints_size = (radius + 1) * len(SYBYL_ATOM_TYPE_IDX_CDPKIT) * 32

    # Get the chemical environment around the center atom
    env = Chem.Fragment()
    Chem.getEnvironment(ctr_atom, molgraph, radius, env)

    # Initialize circular fingerprints
    fingerprints = np.zeros(fingerprints_size, dtype=bool)

    # Count atoms of each type at each distance
    atom_counts = np.zeros((len(SYBYL_ATOM_TYPE_IDX_CDPKIT), radius + 1), dtype=int)

    for atom in env.atoms:
        sybyl_type = Chem.getSybylType(atom)
        if sybyl_type not in SYBYL_ATOM_TYPE_IDX_CDPKIT:
            continue

        sybyl_type_index = SYBYL_ATOM_TYPE_IDX_CDPKIT.index(sybyl_type)
        radius = Chem.getTopologicalDistance(ctr_atom, atom, molgraph)
        atom_counts[sybyl_type_index, radius] += 1

    # Generate 32-bit fingerprints for each combination of atom type and distance
    fingerprint_index = 0
    for radius in range(radius + 1):  # Radius (R0, R1, ..., R5)
        for sybyl_type_index in range(len(SYBYL_ATOM_TYPE_IDX_CDPKIT)):  # Atom type
            for bit in range(32):  # Bit position (B0, B1, ..., B31)
                count = atom_counts[sybyl_type_index, radius]
                # Set bit to 1 if count > bit position
                if count > bit:
                    fingerprints[fingerprint_index] = 1
                fingerprint_index += 1

    return fingerprints


PHYSICOCHEMICAL_DESCRIPTOR_NAMES = [
    "AtomDegree",
    "HybridPolarizability",
    "VSEPRgeometry",
    "AtomValence",
    "EffectivePolarizability",
    "SigmaCharge",
    "MMFF94Charge",
    "PiElectronegativity",
    "SigmaElectronegativity",
    "InductiveEffect",
]


def generate_physicochemical_descriptors(
    ctr_atom: Chem.Atom,
    molgraph: Chem.MolecularGraph,
) -> npt.NDArray[np.float64]:
    _prepare_mol(molgraph)

    return np.array(
        [
            MolProp.getHeavyAtomCount(ctr_atom),
            MolProp.getHybridPolarizability(ctr_atom, molgraph),
            MolProp.getVSEPRCoordinationGeometry(ctr_atom, molgraph),
            MolProp.calcExplicitValence(ctr_atom, molgraph),
            MolProp.calcEffectivePolarizability(ctr_atom, molgraph),
            MolProp.getPEOESigmaCharge(ctr_atom),
            ForceField.getMMFF94Charge(ctr_atom),
            MolProp.calcPiElectronegativity(ctr_atom, molgraph),
            MolProp.getPEOESigmaElectronegativity(ctr_atom),
            MolProp.calcInductiveEffect(ctr_atom, molgraph),
        ],
        dtype=float,
    )


TOPOLOGICAL_DESCRIPTOR_NAMES = [
    "longestMaxTopDistinMolecule",
    "highestMaxTopDistinMatrixRow",
    "diffSPAN",
    "refSPAN",
]


def generate_topological_descriptors(
    ctr_atom: Chem.Atom,
    molgraph: Chem.MolecularGraph,
) -> npt.NDArray[np.float64]:
    _prepare_mol(molgraph)

    max_topo_dist = _max_topological_distance(molgraph)
    max_dist_center = _max_distance_from_reference(molgraph, ctr_atom)

    return np.array(
        [
            max_topo_dist,
            max_dist_center,
            max_topo_dist - max_dist_center,
            max_dist_center / max_topo_dist if max_topo_dist != 0 else 0,
        ],
        dtype=float,
    )


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

    Chem.perceiveComponents(mol, False)


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
