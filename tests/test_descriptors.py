from pathlib import Path

import numpy as np
import pytest
from CDPL import Chem

from src.compute_descriptors import DescriptorGenerator, FAMEDescriptors

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


def test_molecule_and_atom_generators_agree(mol):
    _, desc = FAMEDescriptors(radius=2)._process_molecule(mol, has_soms=False)
    atom_desc_generator = DescriptorGenerator(radius=2)

    for (_, atom_id), (_, atom_desc_from_mol) in desc.items():
        atom = mol.getAtom(atom_id)
        _, atom_desc_from_atom = atom_desc_generator.generate_descriptors(atom, mol)

        assert np.array_equal(
            atom_desc_from_atom, atom_desc_from_mol
        ), "atom and molecule descriptor generators produce the same results"
