import pytest

from trigram_tokenizer.inference import InferencePipe
from .inference_pipe_fixture import inference_pipe, inference_pipe_classic


def test_inference_with_echo(inference_pipe: InferencePipe):
    # run inference

    result = inference_pipe.generate(
        prompt="Hello World!",
        max_tokens=2,
        log_probs=3,
        echo=True,
    )
    assert result.completion.startswith("Hello World!")

    for log_prob_dict in result.log_probs:
        for k, v in log_prob_dict.items():
            assert v <= 1.0


def test_inference_with_log_probs(inference_pipe: InferencePipe):
    # run inference

    result = inference_pipe.generate(
        prompt="Hello World!",
        max_tokens=2,
        log_probs=1,
    )
    assert result.log_probs is not None
    assert len(result.log_probs) == len(result.tokens)
    for log_probs, token in zip(result.log_probs, result.tokens):
        assert token in log_probs


def test_inference_with_echo_classic(inference_pipe_classic: InferencePipe):
    # run inference

    result = inference_pipe_classic.generate(
        prompt="Hello World!",
        max_tokens=2,
        log_probs=3,
        echo=True,
    )
    assert result.completion.startswith(" Hello World!")


def test_inference_with_log_probs_classic(inference_pipe_classic: InferencePipe):
    # run inference

    result = inference_pipe_classic.generate(
        prompt="Hello World!",
        max_tokens=2,
        log_probs=1,
    )
    assert result.log_probs is not None
    assert len(result.log_probs) == len(result.tokens) == 2
