import pytest
from pathlib import Path
from trigram_tokenizer.inference import InferencePipe


def get_inference_pipe():
    src_path = (
        Path(__file__).parents[1]
        / "files"
        / "tmp"
        / "test_training_distributed_2_2_1_bfloat16_True"
        / "determined_checkpoint"
        / "000000000004"
    )

    # initialize inference pipe
    inference_pipe = InferencePipe(
        str(src_path), words_target_dir=src_path / "words_target"
    )

    return inference_pipe


def get_inference_pipe_classic():
    src_path = (
        Path(__file__).parents[1]
        / "files"
        / "tmp"
        / "test_training_distributed_classic_2_2_1_bfloat16_True"
        / "determined_checkpoint"
        / "000000000004"
    )

    # initialize inference pipe
    inference_pipe = InferencePipe(str(src_path))

    return inference_pipe


@pytest.fixture
def inference_pipe():
    pipe = get_inference_pipe()
    yield pipe


@pytest.fixture
def inference_pipe_classic():
    pipe = get_inference_pipe_classic()
    yield pipe
