import logging
import numpy as np
import shelve
import multiprocessing as mp

from pathlib import Path
from datasets import load_dataset
from filelock import FileLock


from trigram_tokenizer.data.memory_map import MemoryMap
from trigram_tokenizer.data.memory_map import MemoryMapBuilder


# Huggingface Dataset repository
HF_DATASET = "HuggingFaceFW/fineweb-edu"
# Path to download and index data to.
DATA_DIRECTORY = Path("<your path to happiness>")
# List of dataset names to be downloaded as newline-separated textfile.
DATASET_LIST = Path(".data/cc_datasets.txt")
# Path of DB with metadata for completed/incomplete downloads.
STATUS_DB = Path(".data/status")
# Validates the correctness of memory map for each 'SAMPLE_INTERVAL' samples.
SAMPLE_INTERVAL = 10000
# Number of parallel processes downloading files.
PARALLEL_PROCESSES = 8


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class DownloadStatus:
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"


def validate(mmap_path: str, valid_samples: list[tuple[int, str]]):
    mmap = MemoryMap(mmap_path)
    for i, sample in valid_samples:
        doc = mmap[i]
        decoded = doc.tobytes().decode("utf-8")
        if decoded != sample:
            raise ValueError("MemoryMap validation failed!")
    return True


def download_and_index(ds_name):
    logger.info(f"Indexing '{ds_name}'.")
    ds = load_dataset(HF_DATASET, name=ds_name, split="train", streaming=True)
    mmap_path = Path(DATA_DIRECTORY, ds_name)
    mmap_builder = MemoryMapBuilder(
        prefix_path=mmap_path,
        index_dtype=np.uint64,
        dtype=np.uint8,
    )

    valid_samples = []
    sample_index = 0

    try:
        for sample in ds:
            sample_text = sample["text"]
            encoded = sample_text.encode(encoding="utf-8")
            byte_array = np.frombuffer(encoded, dtype=np.uint8)
            mmap_builder.add(np_array=byte_array)

            if not sample_index % SAMPLE_INTERVAL:
                if sample_index != 0:
                    logger.info(f"Indexed {SAMPLE_INTERVAL} samples in '{ds_name}'.")
                valid_samples.append((sample_index, sample_text))
            sample_index += 1

        # Append last sample.
        valid_samples.append((sample_index - 1, sample_text))

        mmap_builder.finalize()
        if validate(mmap_path, valid_samples):
            with FileLock(str(STATUS_DB) + ".lock"):
                with shelve.open(str(STATUS_DB)) as db:
                    entry = db[ds_name]
                    entry["status"] = DownloadStatus.COMPLETE
                    db[ds_name] = entry
            logger.info(f"'{ds_name}' successfully indexed.")

    except Exception as e:
        logger.error(f"Error indexing '{ds_name}': {e}")
        with FileLock(str(STATUS_DB) + ".lock"):
            with shelve.open(str(STATUS_DB)) as db:
                entry = db[ds_name]
                entry["status"] = DownloadStatus.FAILED
                db[ds_name] = entry


if __name__ == "__main__":
    incomplete_ds_dl = []
    with open(DATASET_LIST, "r") as f:
        ds_names = [line.strip() for line in f.readlines()]

    with FileLock(str(STATUS_DB) + ".lock"):
        with shelve.open(str(STATUS_DB)) as db:
            for ds_name in ds_names:
                if not db.get(ds_name):
                    entry = {}
                    entry["dataset"] = ds_name
                    entry["status"] = DownloadStatus.INCOMPLETE
                    db[ds_name] = entry
                if db[ds_name]["status"] != DownloadStatus.COMPLETE:
                    incomplete_ds_dl.append(ds_name)

    with mp.Pool(PARALLEL_PROCESSES) as pool:
        pool.map(download_and_index, incomplete_ds_dl)