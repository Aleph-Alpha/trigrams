from typing import Set, List, Optional, Tuple, Union, Dict, NamedTuple
from pathlib import Path
from tqdm import tqdm
import collections
from pydantic import Field
import hashlib
import pickle
import torch
import copy
import collections
from tokenizers import Tokenizer
from trigram_tokenizer.logging import logger
from ..config import BaseConfig

from .encoding_inference import EncodingInference
from .encoding_training import EncodingTraining
from .wordsplit import text_to_words

import string
import itertools


class DecodeResult(NamedTuple):
    words: List[str]
    word_indices: List[int]
    word_ps: List[float]
    logits: List[Optional[torch.Tensor]]
    log_probs: Optional[List[Dict[str, float]]]
    words_count_training: List[Optional[int]]


class TrigramTokenizerConfig(BaseConfig):
    lowercase: bool = Field(False, description="")
    vocab_size: int = Field(description="")
    sequence_length: int = Field(4096, description="")
    vocab_population: int = Field(description="")
    vocab_population_partial_lowercase: int = Field(0, description="")
    do_classic_tokenization: bool = Field(False, description="")
    prefix_path_tokenizer_file: str = Field("")
    seed: int = Field(42, description="")
    initialize: str = Field("hash", description="hash or orthogonal")
    entire_words: bool = Field(
        False, description="use trigramify (False) or hash word directly"
    )
    end_of_text: str = Field("<|endoftext|>", description="")

    word_edge_weight: int = Field(1, description="")

    cache_dir: Optional[Path] = Field(None, description="")


class TrigramTokenizer:
    def __init__(
        self,
        config: TrigramTokenizerConfig,
        trigram_to_vocab: Optional[torch.Tensor] = None,
        words_to_weights_word_indices: Optional[torch.Tensor] = None,
        words_to_weights_token_indices: Optional[torch.Tensor] = None,
        words_to_weights_word_indices_edges: Optional[torch.Tensor] = None,
        words_to_weights_token_indices_edges: Optional[torch.Tensor] = None,
        word_trigram_counts: Optional[torch.Tensor] = None,
        word_counts: Optional[torch.Tensor] = None,
        words: Optional[List[str]] = None,
        word_counter_full: Optional[Dict[str, int]] = None,
    ):
        logger.info(
            f"TrigramTokenizer initializing",
        )
        self.config = config

        assert (
            config.word_edge_weight < 2**8
        ), "counts are represented in int 8, dont make the weight too big!"

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(self.config.seed)

        if self.config.do_classic_tokenization:
            logger.info(
                f"TrigramTokenizer file: {self.config.prefix_path_tokenizer_file}"
            )
            assert self.config.prefix_path_tokenizer_file != ""
            assert Path(
                self.config.prefix_path_tokenizer_file
            ).exists(), f"tokenizer file does not exist {self.config.prefix_path_tokenizer_file}"
            self.classic_tokenizer = Tokenizer.from_file(
                self.config.prefix_path_tokenizer_file
            )
            self.trigram_to_vocab = None  # torch.tensor(range(len(self.classic_tokenizer))).unsqueeze(dim=-1).to(torch.long)
        else:
            if self.config.initialize == "hash":
                self.trigram_to_vocab = (
                    self.initialize_hash_weight(
                        cache_dir=self.config.cache_dir,
                        trigram_count=256 * 256 * 256,
                        vocab_size=self.config.vocab_size,
                        vocab_population=self.config.vocab_population,
                        seed=self.config.seed,
                    )
                    if trigram_to_vocab is None
                    else trigram_to_vocab
                )
            else:
                assert self.config.initialize == "orthogonal"

                self.trigram_to_vocab = (
                    self.initialize_orthogonal_weight(
                        cache_dir=self.config.cache_dir,
                        trigram_count=256 * 256 * 256,
                        vocab_size=self.config.vocab_size,
                        vocab_population=self.config.vocab_population,
                        seed=self.config.seed,
                    )
                    if trigram_to_vocab is None
                    else trigram_to_vocab
                )

        if self.config.vocab_population_partial_lowercase > 0:
            logger.info(
                f"replacing lower-cased vocabs - {self.config.vocab_population_partial_lowercase} ",
            )

            all_chars = list(string.ascii_lowercase + " ")

            for pattern in itertools.product(all_chars, repeat=3):
                lower_pattern = "".join(pattern)
                for num in range(8):
                    cur_pattern = list(lower_pattern)
                    cur_num = num
                    for i in range(3):
                        if cur_num % 2 == 1:
                            cur_pattern[i] = cur_pattern[i].upper()
                        cur_num = cur_num >> 1
                    cur_pattern_joined = "".join(cur_pattern)
                    if lower_pattern == cur_pattern_joined:
                        continue

                    assert self.trigram_to_vocab is not None
                    self.trigram_to_vocab[
                        self.trigram_to_id(str.encode(cur_pattern_joined))
                    ][
                        -self.config.vocab_population_partial_lowercase :
                    ] = self.trigram_to_vocab[
                        self.trigram_to_id(str.encode(lower_pattern))
                    ][
                        -self.config.vocab_population_partial_lowercase :
                    ]

            logger.info(
                f"end replacing lower-cased vocabs",
            )

        self.words_to_weights_word_indices: Optional[
            torch.Tensor
        ] = words_to_weights_word_indices
        self.words_to_weights_token_indices: Optional[
            torch.Tensor
        ] = words_to_weights_token_indices
        self.words_to_weights_word_indices_edges: Optional[
            torch.Tensor
        ] = words_to_weights_word_indices_edges
        self.words_to_weights_token_indices_edges: Optional[
            torch.Tensor
        ] = words_to_weights_token_indices_edges
        self.word_trigram_counts: Optional[torch.Tensor] = word_trigram_counts
        self.word_counts: Optional[torch.Tensor] = word_counts
        self.words: List[str] = list() if words is None else words
        self.word_to_index = {w: i for i, w in enumerate(self.words)}

        self.word_counter_full = word_counter_full or dict()

        logger.info(
            f"end TrigramTokenizer initializing",
        )

    @staticmethod
    def trigramify(
        text: str,
        lowercase: bool,
        strip: bool = False,
    ) -> Tuple[List[Set[bytes]], List[str]]:
        if lowercase:
            text = text.lower()

        if strip:
            text = text.strip()

        words = text_to_words(text=text)

        trigram_sets = list()
        words_ = list()
        for word in words:
            whitespaced_word = f" {word} "
            whitespaced_word_bytes = str.encode(whitespaced_word)
            trigram_set = set()
            for i in range(len(whitespaced_word_bytes) - 2):
                trigram = whitespaced_word_bytes[i : i + 3]
                trigram_set.add(trigram)

            if len(trigram_set) > 0:
                trigram_sets.append(trigram_set)
                words_.append(word)

        return trigram_sets, words_

    @staticmethod
    def trigram_to_id(word_trigram: bytes) -> int:
        assert len(word_trigram) == 3
        trigram_id = 0
        for i, b in enumerate(word_trigram):
            trigram_id += b * pow(256, i)

        return trigram_id

    @staticmethod
    def id_to_trigram(trigram_id: int) -> bytes:
        assert trigram_id < pow(256, 3)

        byte_1 = trigram_id % 256
        trigram_id -= byte_1
        trigram_id = trigram_id // 256
        byte_2 = trigram_id % 256
        trigram_id -= byte_2
        byte_3 = trigram_id // 256
        return bytes([byte_1, byte_2, byte_3])

    def encode_training(
        self, text: str, pad_to_seq_len: bool = True
    ) -> EncodingTraining:
        if self.config.do_classic_tokenization:
            return self.encode_training_token(text, pad_to_seq_len=pad_to_seq_len)
        else:
            return self.encode_training_trigram(text, pad_to_seq_len=pad_to_seq_len)

    def encode_training_token(
        self, text: str, pad_to_seq_len: bool = True
    ) -> EncodingTraining:
        assert not pad_to_seq_len
        tokenized_text = self.classic_tokenizer.encode(text).ids

        trigram_set_position_ids = list()
        trigram_token_ids = list()
        trigram_sets_input = list()
        trigram_sets_targets = list()
        position_ids = list()
        trigram_set_target_is_eot = list()
        words_targets: List[Optional[str]] = list()

        eot_token = self.classic_tokenizer.encode(self.config.end_of_text).ids
        seq_len = len(tokenized_text) - 1
        targets = torch.zeros((seq_len, 1), dtype=torch.int32)  # CAN BE MEMORY INTENSE!
        for position_index, (trigram_set, word) in enumerate(
            zip(
                [[i] for i in tokenized_text],
                [
                    self.classic_tokenizer.decode([i], skip_special_tokens=False)
                    for i in tokenized_text
                ],
            )
        ):
            trigram_set_tensor = torch.tensor(trigram_set)
            if position_index < seq_len:
                trigram_sets_input.append(set(trigram_set))
                position_ids.append(position_index)
            if position_index > 0 and position_index <= (seq_len):
                trigram_sets_targets.append(set(trigram_set))
                trigram_set_target_is_eot.append(
                    trigram_set_tensor.tolist() == eot_token
                )
                words_targets.append(word)
            for word_trigram in [trigram_set_tensor]:
                vocab_ids = word_trigram
                if (position_index < seq_len) and position_index <= (seq_len - 1):
                    trigram_set_position_ids.append(position_index)
                    trigram_token_ids.append(vocab_ids)
                if position_index > 0 and position_index <= seq_len:
                    targets[position_index - 1] = vocab_ids[0]

        return EncodingTraining(
            trigram_set_position_ids=torch.tensor(
                trigram_set_position_ids, dtype=torch.int32
            ),
            trigram_token_ids=torch.stack(trigram_token_ids),
            targets=targets,
            trigram_sets_input=trigram_sets_input,
            trigram_sets_targets=trigram_sets_targets,
            position_ids=torch.tensor(position_ids, dtype=torch.int32),
            trigram_set_target_is_eot=torch.tensor(
                trigram_set_target_is_eot, dtype=torch.bool
            ),
            trigram_set_input_is_padding=torch.zeros(
                (len(trigram_set_target_is_eot),), dtype=torch.bool
            ),
            loss_weights=torch.ones(
                (len(trigram_set_target_is_eot),), dtype=torch.float
            ),
            words_targets=words_targets,
        )

    @classmethod
    def encode_word_to_vocab_ids(
        cls, config: TrigramTokenizerConfig, word: str
    ) -> torch.Tensor:
        vocab_ids = []
        for i in range(
            config.vocab_population - config.vocab_population_partial_lowercase
        ):
            vocab_ids.append(
                int(
                    hashlib.md5(str.encode(f"{word}_{i}")).hexdigest(),
                    16,
                )
                % config.vocab_size
            )
        for i in range(config.vocab_population_partial_lowercase):
            vocab_ids.append(
                int(
                    hashlib.md5(str.encode(f"{word.lower()}_lower_{i}")).hexdigest(),
                    16,
                )
                % config.vocab_size
            )
        vocab_ids_tensor = torch.tensor(vocab_ids)

        return vocab_ids_tensor

    def encode_training_trigram(
        self,
        text: Union[str, List[str]],
        pad_to_seq_len=True,
        text_loss_weights: Optional[List[float]] = None,
    ) -> EncodingTraining:
        # always look at a list from now on
        if isinstance(text, str):
            assert text_loss_weights is None
            text = [text]
            text_loss_weights = [1.0]
        elif isinstance(text, list):
            assert text_loss_weights is not None
            assert len(text) == len(text_loss_weights)
        else:
            raise NotImplementedError

        assert isinstance(text, list)
        assert isinstance(text_loss_weights, list)

        trigram_sets = list()
        words = list()
        trigram_sets_loss_weights = list()
        for t, lw in zip(text, text_loss_weights):
            trigram_sets_, words_ = self.trigramify(
                text=t,
                lowercase=self.config.lowercase,
                strip=True,
            )
            trigram_sets.extend(trigram_sets_)
            words.extend(words_)
            trigram_sets_loss_weights.extend([lw for _ in range(len(trigram_sets_))])

        assert len(trigram_sets) == len(words)
        assert len(trigram_sets) == len(trigram_sets_loss_weights)

        trigram_set_position_ids = list()
        trigram_token_ids = list()
        trigram_sets_input = list()
        trigram_sets_targets = list()
        position_ids = list()
        trigram_set_target_is_eot = list()
        words_targets: List[Optional[str]] = list()

        seq_len = (
            self.config.sequence_length if pad_to_seq_len else len(trigram_sets) - 1
        )
        targets = torch.zeros(
            (seq_len, self.config.vocab_size), dtype=torch.int8
        )  # CAN BE MEMORY INTENSE!
        loss_weights = torch.zeros(
            (seq_len,), dtype=torch.float32
        )  # CAN BE MEMORY INTENSE!
        for position_index, (trigram_set, word, lw) in enumerate(
            zip(trigram_sets, words, trigram_sets_loss_weights)
        ):
            hash_word = self.config.entire_words or "<|" in word
            if hash_word:
                trigram_set = set([word.encode()])
            if position_index < seq_len:
                trigram_sets_input.append(trigram_set)
                position_ids.append(position_index)
            if position_index > 0 and position_index <= seq_len:
                trigram_sets_targets.append(trigram_set)
                trigram_set_target_is_eot.append(word == self.config.end_of_text)
                words_targets.append(word)
            for word_trigram in trigram_set:
                if hash_word:
                    assert len(trigram_set) == 1
                    vocab_ids = self.encode_word_to_vocab_ids(self.config, word)
                else:
                    trigram_id = self.trigram_to_id(word_trigram)
                    assert self.trigram_to_vocab is not None
                    vocab_ids = self.trigram_to_vocab[trigram_id]

                if position_index < seq_len:
                    trigram_set_position_ids.append(position_index)
                    trigram_token_ids.append(vocab_ids)
                if position_index > 0 and position_index <= seq_len:
                    targets[
                        position_index - 1, vocab_ids
                    ] += 1  # we increment to allow for loss fn experiments around the magnitude of the sparse hits. will bo converted to bool in loss atm

                    loss_weights[position_index - 1] = lw * (
                        self.config.word_edge_weight
                        if word_trigram.startswith(b" ") or word_trigram.endswith(b" ")
                        else 1
                    )

            if position_index == seq_len:
                break

        return EncodingTraining(
            trigram_set_position_ids=torch.tensor(
                trigram_set_position_ids, dtype=torch.int32
            ),
            trigram_token_ids=torch.stack(trigram_token_ids),
            targets=targets,
            trigram_sets_input=trigram_sets_input,
            trigram_sets_targets=trigram_sets_targets,
            position_ids=torch.tensor(position_ids, dtype=torch.int32),
            trigram_set_target_is_eot=torch.tensor(
                trigram_set_target_is_eot, dtype=torch.bool
            ),
            trigram_set_input_is_padding=torch.zeros(
                (len(trigram_set_target_is_eot),), dtype=torch.bool
            ),
            loss_weights=loss_weights,
            words_targets=words_targets,
        )

    def encode(self, text: str, strip: bool = False) -> EncodingInference:
        if self.config.do_classic_tokenization:
            return self.encode_token(text)
        else:
            return self.encode_trigram(text, strip)

    def encode_token(self, text: str) -> EncodingInference:
        tokenized_text = self.classic_tokenizer.encode(text).ids

        trigram_set_position_ids = list()
        trigram_token_ids = list()
        trigram_sets_input = list()
        position_ids = list()

        all_words = []
        for position_index, (trigram_set, word) in enumerate(
            zip(
                [set([i]) for i in tokenized_text],
                [self.classic_tokenizer.decode([i]) for i in tokenized_text],
            )
        ):
            all_words.append(word)
            trigram_set_tensor = torch.tensor(list(trigram_set))
            trigram_sets_input.append(trigram_set)
            position_ids.append(position_index)

            for word_trigram in [trigram_set_tensor]:
                vocab_ids = word_trigram
                trigram_set_position_ids.append(position_index)
                trigram_token_ids.append(vocab_ids)

        return EncodingInference(
            trigram_set_position_ids=torch.tensor(
                trigram_set_position_ids, dtype=torch.long
            ),
            trigram_token_ids=torch.stack(trigram_token_ids),
            trigram_sets_input=trigram_sets_input,
            position_ids=torch.tensor(position_ids, dtype=torch.long),
            words=all_words,
        )

    def encode_trigram(self, text: str, strip: bool = False) -> EncodingInference:
        trigram_sets, words = self.trigramify(
            text=text,
            lowercase=self.config.lowercase,
            strip=strip,
        )

        trigram_set_position_ids = list()
        trigram_token_ids = list()
        trigram_sets_input = list()
        position_ids = list()
        for position_index, (trigram_set, word) in enumerate(zip(trigram_sets, words)):
            hash_word = self.config.entire_words or "<|" in word
            if hash_word:
                trigram_set = set([word.encode()])
            trigram_sets_input.append(trigram_set)
            position_ids.append(position_index)
            for word_trigram in trigram_set:
                if hash_word:
                    assert len(trigram_set) == 1
                    vocab_ids = self.encode_word_to_vocab_ids(self.config, word)
                else:
                    trigram_id = self.trigram_to_id(word_trigram)
                    assert self.trigram_to_vocab is not None
                    vocab_ids = self.trigram_to_vocab[trigram_id]
                trigram_set_position_ids.append(position_index)
                trigram_token_ids.append(vocab_ids)

        return EncodingInference(
            trigram_set_position_ids=torch.tensor(
                trigram_set_position_ids, dtype=torch.int32
            ),
            trigram_token_ids=torch.stack(trigram_token_ids),
            trigram_sets_input=trigram_sets_input,
            position_ids=torch.tensor(position_ids, dtype=torch.int32),
            words=words,
        )

    def decode(
        self,
        logits: torch.Tensor,
        log_probs: Optional[int] = None,
        target_words: Optional[List[str]] = None,
        more_words: Optional[str] = None,
        blacklist_words: Optional[List[str]] = None,
        word_edge_weight: float = 1.0,
        softmax: bool = True,
    ) -> DecodeResult:
        assert logits.ndim == 2, "expecting two dimensions [seq, vocab]"
        if self.config.do_classic_tokenization:
            assert blacklist_words is None
            assert word_edge_weight == 1.0
            return self.decode_token(logits, log_probs, target_words)
        else:
            return self.decode_trigram(
                logits,
                log_probs,
                target_words,
                more_words=more_words,
                blacklist_words=blacklist_words,
                word_edge_weight=word_edge_weight,
                softmax=softmax,
            )

    def decode_token(
        self,
        logits: torch.Tensor,
        log_probs: Optional[int] = None,
        target_words: Optional[List[str]] = None,
    ) -> DecodeResult:
        argmax_indices = logits.argmax(-1).tolist()
        argmax_words = [self.classic_tokenizer.decode([i]) for i in argmax_indices]
        log_probs_ = logits.log_softmax(-1)
        probs_ = logits.softmax(-1)

        log_probs_result: Optional[List[Dict[str, float]]] = None
        if log_probs is not None:
            log_probs_result = list()
            for _seq_idx, word_logprobs in enumerate(log_probs_):
                log_probs_word_result = dict()

                top_values, top_indices = word_logprobs.topk(log_probs)
                for v, i in zip(top_values, top_indices):
                    log_probs_word_result[
                        self.classic_tokenizer.decode([i.item()])
                    ] = float(v.item())

                log_probs_result.append(log_probs_word_result)

        return DecodeResult(
            words=argmax_words,
            word_indices=argmax_indices,
            word_ps=[
                float(probs_[indx, i].item()) for indx, i in enumerate(argmax_indices)
            ],
            logits=[l for l in logits],
            log_probs=log_probs_result,
            words_count_training=[None for _ in argmax_words],
        )

    @classmethod
    def get_sparse_word_list_from_words(
        cls,
        words: List[str],
        index_base: int,
        trigram_to_vocab: torch.Tensor,
        config: TrigramTokenizerConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        words_to_weights_word_indices = list()
        words_to_weights_token_indices = list()
        words_to_weights_word_indices_edges = list()
        words_to_weights_token_indices_edges = list()

        word_trigram_counts = list()

        for idx, word in enumerate(words):
            (token_ids, token_ids_edge, trigram_count) = cls.process_word(
                word=word,
                trigram_to_vocab=trigram_to_vocab,
                config=config,
            )

            for t_id in token_ids:
                words_to_weights_word_indices.append(index_base + idx)
                words_to_weights_token_indices.append(t_id)

            for t_id in token_ids_edge:
                words_to_weights_word_indices_edges.append(index_base + idx)
                words_to_weights_token_indices_edges.append(t_id)

            word_trigram_counts.append(trigram_count)

        return (
            torch.tensor(words_to_weights_word_indices),
            torch.tensor(words_to_weights_token_indices),
            torch.tensor(words_to_weights_word_indices_edges),
            torch.tensor(words_to_weights_token_indices_edges),
            torch.tensor(word_trigram_counts),
        )

    def decode_trigram(
        self,
        logits: torch.Tensor,
        log_probs: Optional[int] = None,
        target_words: Optional[List[str]] = None,
        more_words: Optional[str] = None,
        blacklist_words: Optional[List[str]] = None,
        word_edge_weight: float = 1.0,
        softmax: bool = True,
    ) -> DecodeResult:
        # prepare logits and scores
        assert self.words_to_weights_word_indices is not None

        logits_activated = (
            (logits.float() ).sigmoid().to(self.words_to_weights_word_indices.device)
        )

        all_words = copy.deepcopy(self.words)  # TODO no deepcopy needed
        words_to_weights_word_indices = self.words_to_weights_word_indices
        words_to_weights_token_indices = self.words_to_weights_token_indices
        words_to_weights_word_indices_edges = self.words_to_weights_word_indices_edges
        words_to_weights_token_indices_edges = self.words_to_weights_token_indices_edges
        word_trigram_counts = self.word_trigram_counts

        if more_words:
            more_words_split = text_to_words(more_words)
            more_words_split = [
                w for w in more_words_split if w not in self.word_to_index
            ]
            assert self.trigram_to_vocab is not None
            (
                more_words_to_weights_word_indices,
                more_words_to_weights_token_indices,
                more_words_to_weights_word_indices_edges,
                more_words_to_weights_token_indices_edges,
                more_word_trigram_counts,
            ) = self.get_sparse_word_list_from_words(
                more_words_split,
                index_base=len(self.words),
                trigram_to_vocab=self.trigram_to_vocab,
                config=self.config,
            )

            if len(more_words_split) > 0:
                assert words_to_weights_word_indices is not None
                assert words_to_weights_token_indices is not None
                assert words_to_weights_word_indices_edges is not None
                assert words_to_weights_token_indices_edges is not None
                assert word_trigram_counts is not None

                # logger.warning(f"Added {len(more_words_split)} new words to decoding.")
                all_words.extend(more_words_split)
                words_to_weights_word_indices = torch.cat(
                    [
                        words_to_weights_word_indices,
                        more_words_to_weights_word_indices.to(
                            words_to_weights_word_indices.device
                        ),
                    ]
                )
                words_to_weights_token_indices = torch.cat(
                    [
                        words_to_weights_token_indices,
                        more_words_to_weights_token_indices.to(
                            words_to_weights_token_indices.device
                        ),
                    ]
                )
                words_to_weights_word_indices_edges = torch.cat(
                    [
                        words_to_weights_word_indices_edges,
                        more_words_to_weights_word_indices_edges.to(
                            words_to_weights_word_indices_edges.device
                        ),
                    ]
                )
                words_to_weights_token_indices_edges = torch.cat(
                    [
                        words_to_weights_token_indices_edges,
                        more_words_to_weights_token_indices_edges.to(
                            words_to_weights_token_indices_edges.device
                        ),
                    ]
                )

                word_trigram_counts = torch.cat(
                    [
                        word_trigram_counts,
                        more_word_trigram_counts.to(word_trigram_counts.device),
                    ]
                )

        # initialize scores
        assert word_trigram_counts is not None
        scores = torch.zeros(
            (logits_activated.shape[0] * word_trigram_counts.shape[0],),
            device=logits_activated.device,
            dtype=logits_activated.dtype,
        )

        # add logits
        assert words_to_weights_word_indices is not None
        assert words_to_weights_token_indices is not None
        if logits_activated.shape[0] > 1:
            words_to_weights_token_indices = words_to_weights_token_indices.repeat(
                logits_activated.shape[0]
            )
            words_to_weights_word_indices = torch.cat(
                [
                    (
                        words_to_weights_word_indices
                        + int(sequence_index * word_trigram_counts.shape[0])
                    )
                    for sequence_index in range(logits_activated.shape[0])
                ],
                0,
            )
        scores.index_add_(
            0,
            words_to_weights_word_indices,
            logits_activated.flatten()[words_to_weights_token_indices],
        )

        # add logits for edges
        assert words_to_weights_word_indices_edges is not None
        assert words_to_weights_token_indices_edges is not None
        if logits_activated.shape[0] > 1:
            words_to_weights_token_indices_edges = (
                words_to_weights_token_indices_edges.repeat(logits_activated.shape[0])
            )
            words_to_weights_word_indices_edges = torch.cat(
                [
                    (
                        words_to_weights_word_indices_edges
                        + int(sequence_index * word_trigram_counts.shape[0])
                    )
                    for sequence_index in range(logits_activated.shape[0])
                ],
                0,
            )
        scores.index_add_(
            0,
            words_to_weights_word_indices_edges,
            word_edge_weight
            * logits_activated.flatten()[words_to_weights_token_indices_edges],
        )

        # view
        scores = (
            scores.view(logits_activated.shape[0], -1)
            # / word_trigram_counts
            # / self.config.vocab_population
        )

        #sigmoid version:
        word_trigram_counts_others = self.config.vocab_size-word_trigram_counts*self.config.vocab_population
        scores = scores/(self.config.vocab_population*word_trigram_counts) + (scores-logits_activated.sum(-1).repeat(scores.shape[-1]).view((scores.shape[-1],-1)).t())/word_trigram_counts_others

        if softmax:
            scores = scores.softmax(-1)

        # greedy sampling
        argmax_indices = scores.argmax(1).tolist()
        argmax_words = [all_words[word_index] for word_index in argmax_indices]
        argmax_scores = [
            float(scores[sequence_index, word_index].item())
            for sequence_index, word_index in enumerate(argmax_indices)
        ]

        if blacklist_words is not None:
            assert logits.shape[0] == 1, "blacklist words not implemented for echo"
            assert len(argmax_words) == 1, "can only have one argmax word without echo"

            # resample
            k = 1
            blacklist_indices = set(
                [self.word_to_index.get(w, -1) for w in blacklist_words]
            )
            while argmax_indices[0] in blacklist_indices:
                k += 1
                argmax_indices = [scores.topk(k).indices[0, -1].item()]

            # update
            if k > 1:
                argmax_words = [all_words[word_index] for word_index in argmax_indices]
                argmax_scores = [
                    float(scores[sequence_index, word_index].item())
                    for sequence_index, word_index in enumerate(argmax_indices)
                ]

        # get logprobs
        log_probs_result: Optional[List[Dict[str, float]]] = None
        if log_probs is not None:
            # make sure we are not getting more log probs than we have words
            log_probs = min(log_probs, scores.shape[-1])

            # initialize log probs with the actual generations
            # sigmoid values don't sum up to 1 (i.p. not * random activation patterns)
            log_probs_result = [
                {word: float(torch.tensor(score).log().item())}
                for word, score in zip(argmax_words, argmax_scores)
            ]

            # add topk tokens to logprobs
            if log_probs > 0:
                top_k_values, top_k_indices = scores.topk(log_probs, dim=-1)

                for seq_idx in range(top_k_values.shape[0]):
                    for word_p, word_index in zip(
                        top_k_values[seq_idx].tolist(), top_k_indices[seq_idx].tolist()
                    ):
                        word = all_words[word_index]
                        log_probs_result[seq_idx][word] = float(
                            torch.tensor(word_p).log().item()
                        )
                        if len(log_probs_result[seq_idx]) == log_probs:
                            break

            # add target words to logprobs
            if target_words is not None:
                # assert False, "not yet ready?"+
                word_to_index: Optional[Dict[str, int]] = None
                assert len(target_words) == len(log_probs_result)
                for seq_idx, target_word in enumerate(target_words):
                    word_index = self.word_to_index.get(target_word)
                    if word_index is None:
                        if word_to_index is None:
                            word_to_index = {
                                word: idx for idx, word in enumerate(all_words)
                            }
                        word_index = word_to_index.get(target_word)
                        assert (
                            word_index is not None
                        ), f" BROKEN INDEX FOR {target_word}"
                        log_probs_result[seq_idx][target_word] = (
                            scores[seq_idx, word_index].log().item()
                        )
                    else:
                        log_probs_result[seq_idx][target_word] = float(
                            scores[seq_idx, word_index].log().item()
                        )

        return DecodeResult(
            words=argmax_words,
            word_indices=argmax_indices,
            word_ps=argmax_scores,
            logits=logits.float(),
            log_probs=log_probs_result,
            words_count_training=[
                self.word_counter_full.get(w, 0) for w in argmax_words
            ],
        )

    def __len__(self):
        return self.config.vocab_size

    def save(self, dirname: Union[str, Path]):
        dirname = Path(dirname)
        dirname.mkdir(exist_ok=True, parents=True)
        torch.save(
            {
                "trigram_to_vocab": self.trigram_to_vocab,
                "words_to_weights_word_indices": self.words_to_weights_word_indices,
                "words_to_weights_token_indices": self.words_to_weights_token_indices,
                "words_to_weights_word_indices_edges": self.words_to_weights_word_indices_edges,
                "words_to_weights_token_indices_edges": self.words_to_weights_token_indices_edges,
                "word_trigram_counts": self.word_trigram_counts,
                "word_counts": self.word_counts,
                "words": self.words,
                "word_counter_full": self.word_counter_full,
            },
            str(dirname / "tokenizer.pt"),
        )
        self.config.save(dirname / "tokenizer_config.yaml")

        logger.info(
            f"TrigramTokenizer saved checkpoint to {dirname}",
        )

    @classmethod
    def process_word(
        cls,
        word: str,
        trigram_to_vocab: torch.Tensor,
        config: TrigramTokenizerConfig,
    ):
        trigram_sets, words = cls.trigramify(
            text=word,
            lowercase=config.lowercase,
        )
        assert len(words) == 1, f"{word}, {words}"

        token_ids = set()
        token_ids_edge = set()
        hash_word = config.entire_words or "<|" in word
        if hash_word:
            vocab_ids = cls.encode_word_to_vocab_ids(config, word=word)
            for t_id in vocab_ids.tolist():
                token_ids.add(t_id)

            trigram_count = 1
        else:
            assert len(trigram_sets) == 1
            for trigram in trigram_sets[0]:
                trigram_id = cls.trigram_to_id(trigram)
                vocab_ids = trigram_to_vocab[trigram_id]

                # for now we are just using a set, compare loss_fn
                for t_id in vocab_ids.tolist():
                    if trigram.startswith(b" ") or trigram.endswith(b" "):
                        token_ids_edge.add(t_id)
                    else:
                        token_ids.add(t_id)

            trigram_count = len(trigram_sets[0])

        return token_ids, token_ids_edge, trigram_count

    def to(self, device):
        if self.words_to_weights_word_indices is not None:
            self.words_to_weights_word_indices = self.words_to_weights_word_indices.to(
                device
            )

        if self.words_to_weights_token_indices is not None:
            self.words_to_weights_token_indices = (
                self.words_to_weights_token_indices.to(device)
            )

        if self.words_to_weights_word_indices_edges is not None:
            self.words_to_weights_word_indices_edges = (
                self.words_to_weights_word_indices_edges.to(device)
            )

        if self.words_to_weights_token_indices_edges is not None:
            self.words_to_weights_token_indices_edges = (
                self.words_to_weights_token_indices_edges.to(device)
            )

        if self.word_trigram_counts is not None:
            self.word_trigram_counts = self.word_trigram_counts.to(device)

        return self

    @classmethod
    def load(
        cls,
        dirname: Union[str, Path],
        words_target_dir: Optional[Union[str, Path]] = None,
        reduce_tokenizer_words_to: Optional[int] = None,
        top_word_dict: Optional[Union[str, Path]] = None,
        top_word_list: Optional[List[Tuple[str, int]]] = None,
    ):
        if top_word_list is not None:
            assert top_word_dict is None
            assert words_target_dir is None

        dirname = Path(dirname)
        tokenizer_dict = torch.load(str(dirname / "tokenizer.pt"))
        config = TrigramTokenizerConfig.from_yaml(dirname / "tokenizer_config.yaml")

        # load words if defined
        word_counts_combined: collections.Counter = collections.Counter()
        if words_target_dir is not None:
            logger.info(
                f"TrigramTokenizer loading {words_target_dir}",
            )
            words_target_dir = Path(words_target_dir)

            # load all dp files
            for words_file in sorted(list(words_target_dir.glob("*.pt"))):
                logger.info(
                    f"TrigramTokenizer loading {words_file}",
                )
                words = torch.load(words_file)
                assert isinstance(words, collections.Counter)
                word_counts_combined += words

        if top_word_dict is not None:
            word_counts_combined_loaded = pickle.load(open(top_word_dict, "rb"))
            assert isinstance(word_counts_combined_loaded, collections.Counter)
            word_counts_combined = word_counts_combined + word_counts_combined_loaded

        if len(word_counts_combined):
            # add special words
            # add large count because we want to keep them
            word_counts_combined["<|\n|>"] += 0
            word_counts_combined["<|endoftext|>"] += 0
            word_counts_combined["<|no_ws|>"] += 0
            word_counts_combined["<|ws|>"] += 0
            word_counts_combined["<|2<-ws->|>"] += 0
            word_counts_combined["<|4<-ws->|>"] += 0
            word_counts_combined["<|6<-ws->|>"] += 0
            word_counts_combined["<|8<-ws->|>"] += 0
            word_counts_combined["<|2<-\n->|>"] += 0
            word_counts_combined["<|4<-\n->|>"] += 0
            word_counts_combined["<|6<-\n->|>"] += 0
            word_counts_combined["<|8<-\n->|>"] += 0

        if len(word_counts_combined) or top_word_list is not None:
            logger.info("START converting words to weights")

            if top_word_list is None:
                word_counts_combined_list = sorted(
                    [(word, count) for (word, count) in word_counts_combined.items()],
                    key=lambda i: i[1],
                    reverse=True,
                )
            else:
                word_counts_combined_list = top_word_list

            logger.info(
                f"TrigramTokenizer loaded {len(word_counts_combined_list)} words",
            )

            all_words = []
            for word_index, (word, _) in enumerate(word_counts_combined_list):
                all_words.append((word_index, word))

            # convert to weights
            (
                words_to_weights_word_indices,
                words_to_weights_token_indices,
                words_to_weights_word_indices_edges,
                words_to_weights_token_indices_edges,
                word_trigram_counts,
            ) = cls.get_sparse_word_list_from_words(
                [i[1] for i in all_words],
                index_base=0,
                trigram_to_vocab=tokenizer_dict["trigram_to_vocab"],
                config=config,
            )
            logger.info("END converting words to weights")

            # store in state_dict
            tokenizer_dict[
                "words_to_weights_word_indices"
            ] = words_to_weights_word_indices
            tokenizer_dict[
                "words_to_weights_token_indices"
            ] = words_to_weights_token_indices
            tokenizer_dict[
                "words_to_weights_word_indices_edges"
            ] = words_to_weights_word_indices_edges
            tokenizer_dict[
                "words_to_weights_token_indices_edges"
            ] = words_to_weights_token_indices_edges
            tokenizer_dict["word_trigram_counts"] = word_trigram_counts
            tokenizer_dict["word_counts"] = torch.tensor(
                [c for (w, c) in word_counts_combined_list]
            )
            tokenizer_dict["words"] = [w for (w, c) in word_counts_combined_list]
            tokenizer_dict["word_counter_full"] = {
                word: count for word, count in word_counts_combined_list
            }

        # TODO legacy
        if "words_to_weights" in tokenizer_dict:
            del tokenizer_dict["words_to_weights"]

        if reduce_tokenizer_words_to is not None:
            logger.info(
                f"reduce_tokenizer_words_to {reduce_tokenizer_words_to} from {len(tokenizer_dict['word_trigram_counts'])}"
            )
            mask = (
                tokenizer_dict["words_to_weights_word_indices"]
                < reduce_tokenizer_words_to
            )
            tokenizer_dict["words_to_weights_word_indices"] = tokenizer_dict[
                "words_to_weights_word_indices"
            ][mask]
            tokenizer_dict["words_to_weights_token_indices"] = tokenizer_dict[
                "words_to_weights_token_indices"
            ][mask]

            mask = (
                tokenizer_dict["words_to_weights_word_indices_edges"]
                < reduce_tokenizer_words_to
            )
            tokenizer_dict["words_to_weights_word_indices_edges"] = tokenizer_dict[
                "words_to_weights_word_indices_edges"
            ][mask]
            tokenizer_dict["words_to_weights_token_indices_edges"] = tokenizer_dict[
                "words_to_weights_token_indices_edges"
            ][mask]

            tokenizer_dict["word_trigram_counts"] = tokenizer_dict[
                "word_trigram_counts"
            ][:reduce_tokenizer_words_to]
            tokenizer_dict["word_counts"] = tokenizer_dict["word_counts"][
                :reduce_tokenizer_words_to
            ]
            tokenizer_dict["words"] = tokenizer_dict["words"][
                :reduce_tokenizer_words_to
            ]

        return cls(config=config, **tokenizer_dict)

    @classmethod
    def init(cls, config: TrigramTokenizerConfig):
        return cls(config=config)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @staticmethod
    def initialize_hash_weight(
        cache_dir: Optional[Path],
        trigram_count: int,
        vocab_size: int,
        vocab_population: int,
        seed: int,
    ) -> torch.Tensor:
        logger.info(
            f"TrigramTokenizer (hash) initializing (no seed needed)",
        )

        cache_file: Optional[Path] = None
        if cache_dir is not None:
            cache_file = (
                cache_dir
                / f"hash_weight_{trigram_count}_{vocab_size}_{vocab_population}.pt"
            )
            if cache_file.is_file():
                logger.info(
                    f"TrigramTokenizer loading from cache {cache_file}",
                )
                final_weight = torch.load(str(cache_file))
                return final_weight

        final_weight = torch.empty((trigram_count, vocab_population), dtype=torch.long)

        for i in range(trigram_count):
            cur_set: Set[int] = set()
            for k in range(vocab_population):
                cur_set_len = len(cur_set)
                j = 0
                while len(cur_set) == cur_set_len:
                    cur_set.add(
                        int(hashlib.md5(str.encode(f"{i}_{k}_{j}")).hexdigest(), 16)
                        % vocab_size
                    )
                    j += 1

            final_weight[i, :] = torch.tensor(sorted(list(cur_set)), dtype=torch.long)

        if cache_file is not None and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            if not cache_file.parent.is_dir():
                cache_file.parent.mkdir(parents=True)
            logger.info(
                f"TrigramTokenizer saving to cache {cache_file}",
            )
            torch.save(final_weight, str(cache_file))

        return final_weight

    @staticmethod
    def initialize_orthogonal_weight(
        cache_dir: Optional[Path],
        trigram_count: int,
        vocab_size: int,
        vocab_population: int,
        seed: int,
    ) -> torch.Tensor:
        logger.info(
            f"TrigramTokenizer (orthogonal) initializing",
        )

        cache_file: Optional[Path] = None
        if cache_dir is not None:
            cache_file = (
                cache_dir
                / f"orthogonal_weight_{trigram_count}_{vocab_size}_{vocab_population}_{seed}.pt"
            )
            if cache_file.is_file():
                logger.info(
                    f"TrigramTokenizer loading from cache {cache_file}",
                )
                final_weight = torch.load(str(cache_file))
                return final_weight

        # initialize orthogonal and cast to long
        # make sure to use the seed and do not alter the rng state otherwise
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        cuda_rng_state = None
        if torch.cuda.is_initialized():
            cuda_rng_state = torch.cuda.get_rng_state_all()
            torch.cuda.manual_seed_all(seed)
        weights = torch.empty((trigram_count, vocab_population), dtype=torch.float32)
        torch.nn.init.orthogonal_(weights)
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

        # shift weights
        weights = weights - weights.min()
        weights = weights / weights.max()
        weights = vocab_size * weights
        weights = weights.long()
        weights[weights >= vocab_size] = 0
        assert weights.max() < vocab_size

        final_weight = torch.empty((trigram_count, vocab_population), dtype=torch.long)
        for vocab_id, weight in enumerate(
            tqdm(weights, disable=torch.distributed.is_initialized())
        ):
            # get indices with lowes population
            weight_set = set()
            for w in weight.tolist():
                while w in weight_set:
                    w += 1
                    if w >= vocab_size:
                        w = 0
                weight_set.add(w)

            assert len(weight_set) == vocab_population
            final_weight[vocab_id, :] = torch.tensor(
                sorted(list(weight_set)), dtype=torch.long
            )

        if cache_file is not None and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            if not cache_file.parent.is_dir():
                cache_file.parent.mkdir(parents=True)
            logger.info(
                f"TrigramTokenizer saving to cache {cache_file}",
            )
            torch.save(final_weight, str(cache_file))

        return final_weight
