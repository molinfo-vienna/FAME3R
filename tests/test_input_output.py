# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=missing-module-docstring,missing-function-docstring

from pathlib import Path

import numpy as np

from fame3r.compute_descriptors import FAMEDescriptors

in_file_path: Path = Path(__file__).parent.joinpath("inputs/COMBINED.sdf")


def test_cached_descriptors(tmp_path):
    desc_generator = FAMEDescriptors(radius=2)

    mol_num_ids, mol_ids, atom_ids, som_labels, descriptors_lst = (
        desc_generator.compute_fame_descriptors(
            in_file=in_file_path.as_posix(),
            out_folder=tmp_path.as_posix(),
            has_soms=False,
        )
    )

    _, out_descriptors_path = desc_generator._get_output_file_paths(
        in_file_path=in_file_path, out_folder_path=tmp_path
    )
    assert (
        out_descriptors_path.exists()
    ), "output descriptor file is generated during calculation"

    (
        mol_num_ids_cache,
        mol_ids_cache,
        atom_ids_cache,
        som_labels_cache,
        descriptors_lst_cache,
    ) = desc_generator._read_precomputed_descriptors(
        out_descriptors=out_descriptors_path, has_soms=False
    )

    assert (
        np.array_equal(mol_num_ids, mol_num_ids_cache)
        and mol_ids == mol_ids_cache
        and np.array_equal(atom_ids, atom_ids_cache)
        and som_labels == som_labels_cache
        and np.array_equal(descriptors_lst, descriptors_lst_cache)
    ), "cached results match newly generated ones exactly"
