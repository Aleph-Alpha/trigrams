from typing import Union, List
from pathlib import Path
from tqdm import tqdm
import numpy as np

from trigram_tokenizer.logging import logger

from .memory_map import MemoryMap


class AlignedMMap:
    def __init__(self, file_path):
        self.ind_arr = np.memmap(
            f"{file_path}.indices.bin", mode="r", order="C", dtype=np.uint32
        )

        self.document_count = self.ind_arr.max()

        self.len_arr_buff = np.memmap(
            f"{file_path}.lengths.bin", mode="r", order="C", dtype=np.uint16
        )
        self.off_arr_buff = np.memmap(
            f"{file_path}.offsets.bin", mode="r", order="C", dtype=np.uint64
        )

        self.data_arr_buff = np.memmap(
            f"{file_path}.bin", mode="r", order="C", dtype=np.uint8
        )

    def __len__(self):
        return self.document_count

    def __getitem__(self, idx: int):
        assert idx < len(
            self
        ), f"cannot retrieve document idx {idx} from {len(self)} documents"

        indices = np.where(self.ind_arr == idx)

        assert len(indices) == 1
        indices = indices[0].tolist()

        size = len(indices)
        start_index = indices[0]

        offsets = np.frombuffer(
            self.off_arr_buff,
            dtype=np.uint64,
            count=int(size),
            offset=int(start_index) * 8,  # np.uint64.itemsize,
        )

        lengths = np.frombuffer(
            self.len_arr_buff,
            dtype=np.uint16,
            count=int(size),
            offset=int(start_index) * 2,  # np.uint16.itemsize,
        )

        all_data = []
        for o, l in zip(offsets, lengths):
            data = np.frombuffer(
                self.data_arr_buff,
                dtype=np.uint8,
                count=l,
                offset=int(o),
            )
            all_data.append(data.tobytes().decode("utf-8"))
        return offsets, lengths, all_data

    @staticmethod
    def index_text_mmap_file(
        prefix_path: Union[str, Path],
        file_path_indices: Union[str, Path],
        file_path_lengths: Union[str, Path],
        file_path_offsets: Union[str, Path],
        all_splittables: str = " .,!?",
        max_bytes: int = 20_000,
        tolerance_bytes: int = 100,
    ):
        # filenames do not contain parameters
        # we assert to make sure that changing the config does not results in unexpected behavior
        assert all_splittables == " .,!?"
        assert max_bytes == 20_000
        assert tolerance_bytes == 100

        # use mmap to inherit changes and not duplicate code
        # Index is loaded to memory because we will need all items anyways
        mmap = MemoryMap(prefix_path=Path(prefix_path), load_index_to_memory=True)

        new_index: List[List[List[int]]] = []
        append_new = True
        ditched_count = 0
        for i in tqdm(range(len(mmap))):
            start, size = mmap.get_start_index_and_size(i)
            if size < tolerance_bytes:
                continue

            j = 0
            offset = 0

            while offset < size:
                force_new = False
                cur_length = size - offset
                if cur_length <= tolerance_bytes:
                    break

                cur_offset = int(start + offset)
                cur_length = int(min(max_bytes, cur_length))
                if not append_new:
                    remaining = max_bytes - sum([x[1] for x in new_index[-1]])

                    if remaining > tolerance_bytes:
                        cur_length = int(min(remaining, cur_length))
                        if cur_length == remaining:
                            force_new = True
                    else:
                        append_new = True

                offset_count = 0
                while True:
                    offset_count += 1

                    cur_bytes = mmap.read_from_buffer(
                        start_index=cur_offset + cur_length - offset_count, size=1
                    )

                    # decode sometimes fails on single-special unicode bytes
                    # don't want to split at random letter
                    try:
                        if cur_bytes.tobytes().decode("utf-8") in all_splittables:
                            break
                    except Exception as e:
                        pass

                if cur_length - offset_count <= 0:
                    if append_new:
                        # no_ws_string = (
                        #     mmap.read_from_buffer(
                        #         start_index=cur_offset + cur_length - offset_count,
                        #         size=offset_count,
                        #     )
                        #     .tobytes()
                        #     .decode("utf-8")
                        # )
                        # print(f"no ws in {no_ws_string}")
                        ditched_count += 1
                        break
                    else:
                        append_new = True
                        continue

                cur_index = [[cur_offset, cur_length - offset_count]]

                # test that the item is readable
                cur_bytes = mmap.read_from_buffer(
                    start_index=cur_offset, size=cur_length - offset_count
                )
                try:
                    text = cur_bytes.tobytes().decode("utf-8")
                except Exception as e:
                    offset += cur_index[0][1]
                    continue

                if append_new:
                    new_index.append(cur_index)
                else:
                    append_new = True
                    new_index[-1].extend(cur_index)

                if (
                    sum([x[1] for x in new_index[-1]]) < (max_bytes - tolerance_bytes)
                    and not force_new
                ):
                    append_new = False

                j += 1
                offset += cur_index[0][1]

        new_index = new_index[:-1]

        all_indices = []
        all_offsets = []
        all_lengths = []
        for i, ind in enumerate(new_index):
            for off, length in ind:
                all_indices.append(i)
                all_offsets.append(off)
                all_lengths.append(length)

        ind_arr = np.array(all_indices, dtype=np.uint32)
        len_arr = np.array(all_lengths, dtype=np.uint16)
        off_arr = np.array(all_offsets, dtype=np.uint64)

        file = open(file_path_indices, "wb")
        file.write(ind_arr.tobytes(order="C"))
        file = open(file_path_lengths, "wb")
        file.write(len_arr.tobytes(order="C"))
        file = open(file_path_offsets, "wb")
        file.write(off_arr.tobytes(order="C"))
        file.close()

        logger.info(
            f"index_text_mmap_file dichted parts of {ditched_count} samples for {prefix_path}"
        )

    @staticmethod
    def assert_index_exists(
        prefix_path: Union[str, Path],
        all_splittables: str = " .,!?",
        max_bytes: int = 20_000,
        tolerance_bytes: int = 100,
    ):
        file_path_indices = f"{prefix_path}.indices.bin"
        file_path_lengths = f"{prefix_path}.lengths.bin"
        file_path_offsets = f"{prefix_path}.offsets.bin"

        if (
            Path(file_path_indices).is_file()
            and Path(file_path_lengths).is_file()
            and Path(file_path_offsets).is_file()
        ):
            return

        logger.info(f"computing text index for {prefix_path}")
        AlignedMMap.index_text_mmap_file(
            prefix_path=prefix_path,
            file_path_indices=file_path_indices,
            file_path_lengths=file_path_lengths,
            file_path_offsets=file_path_offsets,
            all_splittables=all_splittables,
            max_bytes=max_bytes,
            tolerance_bytes=tolerance_bytes,
        )

    def close(self):
        pass

    def open(self):
        pass
