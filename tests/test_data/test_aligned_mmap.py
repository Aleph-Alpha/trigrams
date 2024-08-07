import shutil
from pathlib import Path
from trigram_tokenizer.data.memory_map import MemoryMap, MemoryMapBuilder
from trigram_tokenizer.data.aligned_mmap import AlignedMMap
import numpy as np


def test_aligned_mmap():
    # create cache dir
    cache_dir = Path(__file__).parent.absolute() / "tmp_test_aligned_mmap"
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)

    # write some data
    max_bytes = 20000
    tolerance_bytes = 100
    chars = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    ]  # lets ignore the bs for now , "🇩🇪" , "🤷🏼‍♀️"
    prefix_path = cache_dir / "tmp_data"
    builder = MemoryMapBuilder(
        prefix_path=prefix_path, dtype=np.uint8, index_dtype=np.uint32
    )
    for i, char in enumerate(chars):  # emojies have multiple bytes
        if i % 2 == 0:
            sample_text = f"{char} " * int(max_bytes * 1.5 / 2)
        else:
            sample_text = f"{char} " * (max_bytes // 2)
        encoded = sample_text.encode(encoding="utf-8")
        byte_array = np.frombuffer(encoded, dtype=np.uint8)
        builder.add(np_array=byte_array)
    builder.finalize()

    # load text mmap
    mmap = MemoryMap(
        prefix_path=prefix_path,
        load_index_to_memory=False,
    )
    assert len(mmap) == len(chars), "mmap does not have expected length"

    # index
    AlignedMMap.assert_index_exists(
        prefix_path=prefix_path, max_bytes=max_bytes, tolerance_bytes=tolerance_bytes
    )

    # load
    aligned_mmap = AlignedMMap(file_path=prefix_path)

    char_index = 0
    for idx in range(len(aligned_mmap)):
        offsets, lengths, texts = aligned_mmap[idx]
        assert isinstance(texts, list)  # we get one item per doc

        # assert len of texts
        total_len_in_bytes = sum(len(text.encode("utf-8")) for text in texts)
        assert (
            (max_bytes - tolerance_bytes)
            <= total_len_in_bytes
            <= (max_bytes + tolerance_bytes)
        )

        # each text should only contain one char (except whitespaces)
        for text in texts:
            text_chars = set(text.replace(" ", ""))
            assert len(text_chars) == 1
            assert (
                chars[char_index] in text_chars or chars[char_index + 1] in text_chars
            )
            if chars[char_index + 1] in text_chars:
                char_index += 1
