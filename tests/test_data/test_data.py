import pytest
import shutil
from pathlib import Path


from trigram_tokenizer.data import TextDataset, TextDatasetConfig
from trigram_tokenizer.tokenizer import TrigramTokenizerConfig
from trigram_tokenizer.tokenizer.encoding_training import EncodingTraining


@pytest.mark.parametrize("sequence_length", [64, 128])
@pytest.mark.parametrize("vocab_size", [16000])
@pytest.mark.parametrize("vocab_population", [4])
@pytest.mark.parametrize(
    "pretraining, do_classic_tokenization, initialize",
    [
        [True, True, ""],
        [True, False, "hash"],
        [False, False, "hash"],
    ],  # , [False, "orthogonal"]
)
def test_dataset(
    sequence_length: int,
    vocab_size: int,
    vocab_population: int,
    pretraining: bool,
    do_classic_tokenization: bool,
    initialize: str,
):
    if do_classic_tokenization:
        prefix_path_tokenizer_file = str(
            Path(__file__).parent / "unigram_02pct_cc_v1.0_hf_converted_cleaned.json"
        )
    else:
        prefix_path_tokenizer_file = ""

    if pretraining:
        prefix_paths = [Path(__file__).parent / "data_fineweb" / "CC-MAIN-2013-20"]
    else:
        target_dir = Path(__file__).parent / "tmp_finetuning_dataset"
        if target_dir.is_dir():
            shutil.rmtree(str(target_dir))
            target_dir.mkdir(exist_ok=False, parents=True)
        TextDataset.convert_finetuning_dataset_to_mmap(
            source_file=str(Path(__file__).parent / "finetuning.jsonl"),
            prefix_path=str(target_dir / "finetuning_dataset"),
        )
        prefix_paths = [target_dir / "finetuning_dataset"]

    config = TextDatasetConfig.from_dict(
        {
            "seed": 42,
            "sequence_length": sequence_length,
            "prefix_paths": prefix_paths,
            "pretraining": pretraining,
            "prefix_path_tokenizer_file": prefix_path_tokenizer_file,
            "reset_position_ids": True,
        }
    )
    tokenizer_config = TrigramTokenizerConfig.from_dict(
        {
            "lowercase": False,
            "vocab_size": vocab_size,
            "sequence_length": sequence_length,
            "vocab_population": vocab_population,
            "vocab_population_partial_lowercase": 0,
            "do_classic_tokenization": do_classic_tokenization,
            "prefix_path_tokenizer_file": prefix_path_tokenizer_file,
            "seed": 42,
            "initialize": initialize,
            "entire_words": False,
            "end_of_text": "<|endoftext|>",
            "cache_dir": f"tmp/tokenizer",
        }
    )
    dataset = TextDataset(config=config, tokenizer_config=tokenizer_config)
    dataset_len = len(dataset)

    for idx in range(dataset_len):
        item = dataset[idx]

        padded_count = item.trigram_set_input_is_padding.sum().item()
        if pretraining:
            assert padded_count == 0, "this dataset should not contain any padding"

        assert isinstance(item, EncodingTraining)

        assert len(item.trigram_set_position_ids) == len(
            item.trigram_token_ids
        ), "trigram set position ids len does not match trigram token ids len"
        assert (
            sum([1 if len(i) > 0 else 0 for i in item.trigram_sets_input])
            == sequence_length - padded_count
        )

        assert len(item.position_ids) == sequence_length
        assert len(item.position_ids) == len(item.trigram_sets_input)
        assert len(item.position_ids) == len(item.loss_weights)
        assert len(item.targets) == sequence_length

        if do_classic_tokenization:
            assert item.trigram_token_ids.shape[-1] == 1
            assert item.targets.shape[-1] == 1
        else:
            assert item.trigram_token_ids.shape[-1] == vocab_population
            assert item.targets.shape[-1] == vocab_size

        assert item.position_ids.max().item() < sequence_length

        if do_classic_tokenization:
            assert item.trigram_token_ids.max().item() < 128000
        else:
            assert item.trigram_token_ids.max().item() < vocab_size

        if pretraining:
            assert item.loss_weights.sum() == sequence_length - padded_count
