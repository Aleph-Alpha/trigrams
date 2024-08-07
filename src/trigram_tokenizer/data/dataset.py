from typing import Optional, List
import torch
from pydantic import Field
from pathlib import Path
from tokenizers import Tokenizer
import numpy as np
import json

from ..config import BaseConfig

from .memory_map import MemoryMap, MemoryMapBuilder
from .aligned_mmap import AlignedMMap

from ..tokenizer import TrigramTokenizer, EncodingTraining, TrigramTokenizerConfig
from trigram_tokenizer.logging import logger


class TextDatasetConfig(BaseConfig):
    seed: int = Field(42, description="")

    sequence_length: int = Field(
        2048,
        description="Sequence length in number of tokens in one sample on which a train job is run; at inference time the seqence length of a sample should (usually) not be exceeded.",
    )

    prefix_paths: List[Path] = Field(description="")

    pretraining: bool = Field(True, description="")

    prefix_path_tokenizer_file: Path = Field(description="")

    reset_position_ids: bool = Field(True, description="")


class TextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: TextDatasetConfig,
        tokenizer_config: TrigramTokenizerConfig,
        tokenizer: Optional[TrigramTokenizer] = None,
    ):
        self.config = config
        if self.config.pretraining:
            for prefix_path_index, prefix_path in enumerate(self.config.prefix_paths):
                compute_index_on_rank = False
                if torch.distributed.is_initialized():
                    compute_index_on_rank = (
                        prefix_path_index % torch.distributed.get_world_size()
                        == torch.distributed.get_rank()
                    )
                else:
                    compute_index_on_rank = True

                if compute_index_on_rank:
                    AlignedMMap.assert_index_exists(prefix_path=prefix_path)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            _indexed_texts = list()
            for prefix_path in self.config.prefix_paths:
                indexed_text = AlignedMMap(file_path=prefix_path)
                _indexed_texts.append(indexed_text)

            self.indexed_texts = _indexed_texts
            self.storage_sizes = [len(it) for it in self.indexed_texts]
        else:
            self.memory_maps = [
                MemoryMap(prefix_path=prefix_path, load_index_to_memory=True)
                for prefix_path in self.config.prefix_paths
            ]
            self.storage_sizes = [len(mmap) for mmap in self.memory_maps]

        self.seed: Optional[int] = None
        self.random_index: np.ndarray
        self.set_seed(seed=self.config.seed)

        if Path(self.config.prefix_path_tokenizer_file).is_file():
            self.prefix_path_tokenizer = Tokenizer.from_file(
                str(self.config.prefix_path_tokenizer_file)
            )

        self.tokenizer = tokenizer or TrigramTokenizer.init(config=tokenizer_config)

    def close_all(self):
        if self.config.pretraining:
            for it in self.indexed_texts:
                it.close()

    def open_all(self):
        if self.config.pretraining:
            for it in self.indexed_texts:
                it.open()

    def __len__(self):
        return len(self.random_index)

    def read_aligned_mmap_texts(self, idx: int):
        idx = int(self.random_index[idx % len(self.random_index)])
        for indexed_text, indexed_text_size in zip(
            self.indexed_texts, self.storage_sizes
        ):
            if idx > indexed_text_size - 1:
                idx -= indexed_text_size
                continue
            all_texts = indexed_text[idx][2]
            return all_texts

    def __getitem__(self, idx: int) -> EncodingTraining:
        eot = None
        if self.config.pretraining:
            all_encodings: List[EncodingTraining] = []
            num_words = 0
            while num_words < self.config.sequence_length:
                texts = self.read_aligned_mmap_texts(idx + len(all_encodings))
                for text in texts:
                    encoding = self.tokenizer.encode_training(
                        text=text + "<|endoftext|>", pad_to_seq_len=False
                    )  # do not pad here, this will be done in the concat!
                    num_words += len(encoding)
                    all_encodings.append(encoding)

            encoding = EncodingTraining.concat_encodings(
                all_encodings,
                seq_len=self.config.sequence_length,
                eot=eot,
                reset_position_ids=self.config.reset_position_ids,
            )

            return encoding
        else:
            idx = int(self.random_index[idx])
            for memory_map_size, memory_map in zip(
                self.storage_sizes, self.memory_maps
            ):
                if idx > memory_map_size - 1:
                    idx -= memory_map_size
                    continue

                finetuning_bytes = memory_map[idx]
                finetuning_str = finetuning_bytes.tobytes().decode("utf-8")
                finetuning_list = json.loads(finetuning_str)
                finetuning_texts: List[str] = list()
                finetuning_loss_weights: List[float] = list()
                for i in finetuning_list:
                    assert i["type"] == "text"
                    finetuning_texts.append(i["content"])
                    finetuning_loss_weights.append(1.0 if i["has_loss"] else 0.0)

                assert not self.tokenizer.config.do_classic_tokenization
                encoding = EncodingTraining.concat_encodings(
                    [
                        self.tokenizer.encode_training_trigram(
                            text=finetuning_texts,
                            pad_to_seq_len=False,
                            text_loss_weights=finetuning_loss_weights,
                        )
                    ],
                    seq_len=self.config.sequence_length,
                )

                return encoding

        raise RuntimeError("should not be here as one index should have fit")

    def save(self, dirname: Path):
        dirname = Path(dirname)
        dirname.mkdir(parents=True)
        self.config.save(dirname / "config.yaml")
        self.tokenizer.save(dirname=dirname / "tokenizer")

    @classmethod
    def load(cls, dirname: Path):
        dirname = Path(dirname)
        config = TextDatasetConfig.from_yaml(dirname / "config.yaml")
        tokenizer = TrigramTokenizer.load(dirname=dirname / "tokenizer")

        return cls(config, tokenizer.config, tokenizer)

    def set_seed(self, seed: int):
        if self.seed is not None and self.seed == seed:
            return

        self.seed = seed

        np_rng = np.random.RandomState(seed=seed)
        random_index = np.arange(sum(self.storage_sizes))
        np_rng.shuffle(random_index)
        self.random_index = random_index

    @staticmethod
    def convert_finetuning_dataset_to_mmap(source_file: str, prefix_path: str):
        mmap_builder = MemoryMapBuilder(
            prefix_path=Path(prefix_path),
            index_dtype=np.uint64,
            dtype=np.uint8,
        )

        with open(source_file, "r", encoding="UTF-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue

                encoded = line.encode(encoding="utf-8")
                byte_array = np.frombuffer(encoded, dtype=np.uint8)
                mmap_builder.add(np_array=byte_array)

        mmap_builder.finalize()
