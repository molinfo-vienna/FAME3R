from pathlib import Path

import pytest
from CDPL import Chem

from src.compute_descriptors import FAMEDescriptors

inputs_dir: Path = Path(__file__).parent.joinpath("inputs")


@pytest.fixture(
    params=[
        inputs_dir.joinpath("IMATINIB.sdf"),
    ]
)
def mol(request) -> Chem.BasicMolecule:
    read_mol = Chem.BasicMolecule()
    reader = Chem.MoleculeReader(request.param.as_posix())
    reader.read(read_mol)

    return read_mol


def test_descriptor_dimensions(mol):
    labels, desc = FAMEDescriptors(radius=2)._process_molecule(mol, has_soms=False)
    non_hydrogen_atoms = [atom for atom in mol.atoms if Chem.getSybylType(atom) != 24]

    assert len(desc) == len(
        non_hydrogen_atoms
    ), "descriptors are generated for each non-hydrogen atom"

    assert all(
        [len(atom_desc) == len(labels) for _, atom_desc in desc.values()]
    ), "each atom has the correct amount of descriptors"


@pytest.mark.parametrize("radius", [0, 1, 2, 3, 4, 5])
def test_different_radius_values(mol, radius):
    expected_length = 286 * (radius + 1) + 4
    labels, _ = FAMEDescriptors(radius=radius)._process_molecule(mol, has_soms=False)

    assert (
        len(labels) == expected_length
    ), f"the correct amount of descriptors is generated for {radius=}"
