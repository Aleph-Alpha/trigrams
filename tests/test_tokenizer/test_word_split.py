import pytest
from pathlib import Path
import json
from typing import List, Optional
from trigram_tokenizer.tokenizer.wordsplit import (
    text_to_words,
    words_to_text,
    _do_compress_whitespaces,
    _undo_compress_whitespaces,
    _do_create_factorial_tokens,
    _undo_create_factorial_tokens,
)
from trigram_tokenizer.tokenizer.trigram_tokenizer import TrigramTokenizer

import string
import random


def random_string_with_special_chars(length=20):
    password_characters = (
        string.ascii_letters + string.digits + string.punctuation + "\n" + " "
    )
    return "".join(random.choice(password_characters) for _ in range(length))


def load_more_data():
    data = list()
    with open(Path(__file__).parent / "wiki-dev.jsonl", "r") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                d = json.loads(line)
                data.append((d["summary"], None))

    return data


@pytest.mark.parametrize(
    "text,words_expected",
    [
        ("", []),
        ("Hello World!", ["Hello", "World", "!"]),
        (
            "Hello World!<|endoftext|>abc",
            ["Hello", "World", "!", "<|endoftext|>", "abc"],
        ),
        (
            "Hello<|startoftext|> World!",
            ["Hello", "<|startoftext|>", "<|ws|>", "World", "!"],
        ),
        (
            "Hello\nWorld!",
            ["Hello", "<|\n|>", "World", "!"],
        ),
        (
            "This is a <|test with some numbers 1234.",
            [
                "This",
                "is",
                "a",
                "<",
                "|",
                "<|no_ws|>",
                "test",
                "with",
                "some",
                "numbers",
                "<|ws|>",
                "1",
                "2",
                "3",
                "4",
                ".",
            ],
        ),
    ]
    + load_more_data()
    + [(random_string_with_special_chars(1_000), None)],
)
def test_split(text: str, words_expected: Optional[List[str]]):
    words = text_to_words(text=text)
    if words_expected is not None:
        assert (
            words == words_expected
        ), f"words_expected: {words_expected}\nwords: {words}"

    text_reconstructed = words_to_text(words=words)
    assert (
        text == text_reconstructed
    ), f"text: {text}\nwords: {words}\ntext_reconstructed: {text_reconstructed}"


def test_compress_whitespaces():
    splits_input = [
        " ",
        "Hello",
        "1",
        " ",
        "world",
        " ",
        "1",
        "2",
        "3",
        "!",
        "!",
        " ",
        " ",
        "(",
        " ",
        "a",
        ")",
        ".",
        " ",
    ]
    splits_converted_expected = [
        "<|ws|>",
        "Hello",
        "1",
        "world",
        "<|ws|>",
        "1",
        "2",
        "3",
        "!",
        "!",
        "<|ws|>",
        "<|ws|>",
        "(",
        "<|ws|>",
        "a",
        ")",
        ".",
        "<|ws|>",
    ]
    splits_converted = _do_compress_whitespaces(splits_input)
    assert (
        splits_converted == splits_converted_expected
    ), f"splits_converted: {splits_converted}\nsplits_converted_expected: {splits_converted_expected}"
    splits_converted_undo = _undo_compress_whitespaces(splits_converted)
    assert (
        splits_converted_undo == splits_input
    ), f"splits_converted_undo: {splits_converted_undo}\nsplits_input: {splits_input}"


def test_create_factorial_tokens_3():
    splits_input = [
        " Hello ",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        " Test ",
    ]
    splits_converted_expected = [
        " Hello ",
        "<|4<-ws->|>",
        "<|2<-ws->|>",
        "<|ws|>",
        " Test ",
    ]
    splits_converted = _do_create_factorial_tokens(splits_input)
    assert (
        splits_converted == splits_converted_expected
    ), f"splits_converted: {splits_converted}\nsplits_converted_expected: {splits_converted_expected}"
    splits_converted_undo = _undo_create_factorial_tokens(splits_converted)
    assert (
        splits_converted_undo == splits_input
    ), f"splits_converted_undo: {splits_converted_undo}\nsplits_input: {splits_input}"


def test_create_factorial_tokens_2():
    splits_input = [
        "<|ws|>",
        "<|ws|>",
        "<|\n|>",
        "<|\n|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|\n|>",
        "<|\n|>",
        "<|\n|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|\n|>",
        "<|\n|>",
        "<|\n|>",
        "<|\n|>",
        "abc",
        " Hello ",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        " Test ",
    ]
    splits_converted_expected = [
        "<|2<-ws->|>",
        "<|2<-\n->|>",
        "<|2<-ws->|>",
        "<|ws|>",
        "<|2<-\n->|>",
        "<|\n|>",
        "<|4<-ws->|>",
        "<|4<-\n->|>",
        "abc",
        " Hello ",
        "<|4<-ws->|>",
        "<|4<-ws->|>",
        "<|4<-ws->|>",
        " Test ",
    ]
    splits_converted = _do_create_factorial_tokens(splits_input)
    assert (
        splits_converted == splits_converted_expected
    ), f"splits_converted: {splits_converted}\nsplits_converted_expected: {splits_converted_expected}"
    splits_converted_undo = _undo_create_factorial_tokens(splits_converted)
    assert (
        splits_converted_undo == splits_input
    ), f"splits_converted_undo: {splits_converted_undo}\nsplits_input: {splits_input}"


def test_create_factorial_tokens():
    splits_input = [
        " Test ",
        " Hello ",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        "<|ws|>",
        " Test ",
    ]
    splits_converted_expected = [
        " Test ",
        " Hello ",
        "<|4<-ws->|>",
        "<|4<-ws->|>",
        "<|4<-ws->|>",
        "<|4<-ws->|>",
        "<|ws|>",
        " Test ",
    ]
    splits_converted = _do_create_factorial_tokens(splits_input)
    assert (
        splits_converted == splits_converted_expected
    ), f"splits_converted: {splits_converted}\nsplits_converted_expected: {splits_converted_expected}"
    splits_converted_undo = _undo_create_factorial_tokens(splits_converted)
    assert (
        splits_converted_undo == splits_input
    ), f"splits_converted_undo: {splits_converted_undo}\nsplits_input: {splits_input}"
