import pytest

import shutil
from pathlib import Path

import numpy as np

from trigram_tokenizer.data import MemoryMap, MemoryMapBuilder

from ..utils import get_empty_cache_dir


def test_memory_map():
    """
    tests the creation and read of a memory map dataset
    """

    tmp_path = get_empty_cache_dir("test_memory_map")

    prefix_path = tmp_path / "data_set_prefix"

    data_items = [
        [1, 2, 3, 4, 5],
        [1, 2, 5],
        [45, 1, 20, 303, 30203],
    ]

    # instantiate a builder and write data
    builder = MemoryMapBuilder(prefix_path=prefix_path)

    for data_item in data_items:
        builder.add(np_array=np.array(data_item))
    builder.finalize()

    # make sure an error is raised if the dataset already exist
    with pytest.raises(AssertionError):
        builder = MemoryMapBuilder(prefix_path=prefix_path)

    # load the dataset
    dataset = MemoryMap(prefix_path=prefix_path)
    assert len(dataset) == len(data_items)

    # compare all data items to ground truth
    for data_item, data_item_truth in zip(dataset, data_items):
        assert (np.array(data_item) == np.array(data_item_truth)).all()


def test_fineweb_memory_map():
    mmap = MemoryMap(
        prefix_path=Path(__file__).parent / "data_fineweb" / "CC-MAIN-2013-20",
        load_index_to_memory=False,
    )
    assert len(mmap) == 101, "mmap does not have expected length"

    first_item = mmap[0].tobytes().decode("utf-8")
    last_item = mmap[100].tobytes().decode("utf-8")
