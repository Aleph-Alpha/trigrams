import pytest
from typing import List
from pathlib import Path
import torch
from trigram_tokenizer.tokenizer import (
    EncodingTraining,
    TrigramTokenizer,
    EncodingBatchTraining,
)

from .tokenizer_fixture import trigram_tokenizer


def test_encode_eot(trigram_tokenizer: TrigramTokenizer):
    encoding = trigram_tokenizer.encode_training(
        "Hello World!"
        + trigram_tokenizer.config.end_of_text
        + "42"
        + trigram_tokenizer.config.end_of_text,
        pad_to_seq_len=False,
    )
    assert len(encoding.targets) == 6
    assert len(encoding.position_ids) == 6
    assert len(encoding.loss_weights) == 6
    assert len(encoding.trigram_set_target_is_eot) == 6
    assert encoding.trigram_set_target_is_eot[2] == True
    assert encoding.trigram_set_target_is_eot[5] == True

    batch = EncodingBatchTraining.from_encodings([encoding])

    assert (
        batch.attention_mask
        == torch.tensor(
            [
                [
                    [
                        [False, True, True, True, True, True],
                        [False, False, True, True, True, True],
                        [False, False, False, True, True, True],
                        [True, True, True, False, True, True],
                        [True, True, True, False, False, True],
                        [True, True, True, False, False, False],
                    ]
                ]
            ]
        )
    ).all()


def test_concat_encodings(trigram_tokenizer: TrigramTokenizer):
    encodings = [
        trigram_tokenizer.encode_training("Hello World!", pad_to_seq_len=False),
        trigram_tokenizer.encode_training("This is a test!", pad_to_seq_len=False),
        trigram_tokenizer.encode_training("12345", pad_to_seq_len=False),
    ]

    encoding = EncodingTraining.concat_encodings(encodings, seq_len=None)
    assert len(encoding.targets) == 10
    assert len(encoding.position_ids) == 10
    assert len(encoding.loss_weights) == 10
    assert len(encoding.trigram_set_target_is_eot) == 10

    encoding = EncodingTraining.concat_encodings(encodings, seq_len=4)
    assert len(encoding.targets) == 4
    assert len(encoding.position_ids) == 4
    assert len(encoding.loss_weights) == 4
    assert len(encoding.trigram_set_target_is_eot) == 4

    encoding = EncodingTraining.concat_encodings(encodings, seq_len=2048)

    assert len(encoding.targets) == 2048
    assert len(encoding.position_ids) == 2048
    assert len(encoding.loss_weights) == 2048
    assert len(encoding.trigram_set_target_is_eot) == 2048


def test_index_encodings(trigram_tokenizer: TrigramTokenizer):
    encodings = [
        trigram_tokenizer.encode_training("Hello World!", pad_to_seq_len=False),
        trigram_tokenizer.encode_training("This is a test!", pad_to_seq_len=False),
        trigram_tokenizer.encode_training("12345", pad_to_seq_len=False),
    ]

    encoding = EncodingTraining.concat_encodings(
        encodings, seq_len=2048, reset_position_ids=False
    )

    first_item = encoding[0]
    assert len(first_item) == 1
    second_item = encoding[1]
    assert len(second_item) == 1
    first_two_items = EncodingTraining.concat_encodings(
        [first_item, second_item], reset_position_ids=False
    )
    first_two_items_ = encoding[:2]
    assert first_two_items == first_two_items_

    full_slice = encoding[:99999]
    assert full_slice == encoding


def test_encode_finetuning(trigram_tokenizer: TrigramTokenizer):
    data = [
        {
            "has_loss": False,
            "type": "text",
            "content": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Provide a poem<|endoftext|><|start_header_id|>assistant<|end_header_id|>",
        },
        {
            "has_loss": True,
            "type": "text",
            "content": "The sun is shining.<|endoftext|>",
        },
    ]

    text: List[str] = list()
    for i in data:
        assert isinstance(i["content"], str)
        text.append(i["content"])
    text_loss_weights: List[float] = [1.0 if i["has_loss"] else 0.0 for i in data]

    encoding = trigram_tokenizer.encode_training_trigram(
        text=text, pad_to_seq_len=False, text_loss_weights=text_loss_weights
    )

    assert encoding.words_targets == [
        "<|start_header_id|>",
        "user",
        "<|end_header_id|>",
        "Provide",
        "a",
        "poem",
        "<|endoftext|>",
        "<|start_header_id|>",
        "assistant",
        "<|end_header_id|>",
        "The",
        "sun",
        "is",
        "shining",
        ".",
        "<|endoftext|>",
    ]
    assert len(encoding.targets) == 16
    assert len(encoding.position_ids) == 16
    assert len(encoding.loss_weights) == 16
    assert encoding.loss_weights.tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert len(encoding.trigram_set_target_is_eot) == 16
    assert encoding.trigram_set_target_is_eot[15] == True

    # concat is in dataset to assert loss weights for padding
    encoding_padded = EncodingTraining.concat_encodings(
        [encoding],
        seq_len=32,
    )

    assert encoding_padded.words_targets == encoding.words_targets + [None] * 16
    assert len(encoding_padded.targets) == 32
    assert len(encoding_padded.position_ids) == 32
    assert len(encoding_padded.loss_weights) == 32
    assert (
        encoding_padded.loss_weights.tolist()
        == encoding.loss_weights.tolist() + [0.0] * 16
    )
    assert len(encoding_padded.trigram_set_target_is_eot) == 32
    assert encoding_padded.trigram_set_target_is_eot[15] == True

    # convert to batch
    batch = EncodingBatchTraining.from_encodings(
        [encoding_padded], reset_attention_mask=False
    )
    expected_attention_mask = ~torch.tril(torch.ones((32, 32), dtype=torch.bool)).view(
        (1, 1, 32, 32)
    )

    assert (batch.attention_mask == expected_attention_mask).all()
