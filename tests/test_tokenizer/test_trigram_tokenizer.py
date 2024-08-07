import pytest
from pathlib import Path

from trigram_tokenizer.tokenizer import (
    TrigramTokenizer,
)

from ..utils import get_empty_cache_dir

from .tokenizer_fixture import trigram_tokenizer


def test_trigram_tokenizer(trigram_tokenizer: TrigramTokenizer):
    tmp_path = get_empty_cache_dir("test_trigram_tokenizer")

    encoding = trigram_tokenizer.encode("Hello world!")
    assert (
        len(encoding.trigram_sets_input) == 3
    ), f"did not split into words and special characters: {encoding.trigram_sets_input}"

    encoding_training = trigram_tokenizer.encode_training(
        "Hello world!", pad_to_seq_len=False
    )
    assert (
        len(encoding_training.trigram_sets_input) == 2
    ), f"did not split into words and special characters AND shift for targets: {encoding_training.trigram_sets_input}"

    assert (
        encoding_training.targets is not None
    ), "returned targets although not in train mode"
    assert (
        encoding_training.trigram_sets_targets is not None
    ), "returned trigrams_targets although not in train mode"

    assert len(encoding_training.trigram_sets_input) == len(
        encoding_training.trigram_sets_targets
    )

    trigram_tokenizer.save(tmp_path / "tokenizer" / "tokenizer.pt")

    tokenizer_ = TrigramTokenizer.load(tmp_path / "tokenizer" / "tokenizer.pt")
    encoding_training_ = tokenizer_.encode_training(
        "Hello world!", pad_to_seq_len=False
    )

    assert encoding_training.trigram_sets_input == encoding_training_.trigram_sets_input
    assert (
        encoding_training.trigram_sets_targets
        == encoding_training_.trigram_sets_targets
    )
