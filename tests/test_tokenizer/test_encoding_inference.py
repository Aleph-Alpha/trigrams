import pytest
from pathlib import Path

from trigram_tokenizer.tokenizer import EncodingInference, TrigramTokenizer

from .tokenizer_fixture import trigram_tokenizer


def test_concat_encodings(trigram_tokenizer: TrigramTokenizer):
    encodings = [
        trigram_tokenizer.encode("Hello World!"),
        trigram_tokenizer.encode("This is a test!"),
        trigram_tokenizer.encode("12345"),
    ]

    encoding = EncodingInference.concat_encodings(encodings, seq_len=None)
    assert len(encoding.position_ids) == 13
    assert len(encoding.words) == 13

    encoding = EncodingInference.concat_encodings(encodings, seq_len=4)
    assert len(encoding.position_ids) == 4
    assert len(encoding.words) == 4

    encoding = EncodingInference.concat_encodings(encodings, seq_len=2048)

    assert len(encoding.position_ids) == 2048
    assert len(encoding.words) == 2048


def test_index_encodings(trigram_tokenizer: TrigramTokenizer):
    encodings = [
        trigram_tokenizer.encode("Hello World!"),
        trigram_tokenizer.encode("This is a test!"),
        trigram_tokenizer.encode("12345"),
    ]

    encoding = EncodingInference.concat_encodings(encodings, seq_len=2048)

    first_item = encoding[0]
    assert len(first_item) == 1
    second_item = encoding[1]
    assert len(second_item) == 1
    first_two_items = EncodingInference.concat_encodings([first_item, second_item])
    first_two_items_ = encoding[:2]
    assert first_two_items == first_two_items_

    full_slice = encoding[:99999]
    assert full_slice == encoding
