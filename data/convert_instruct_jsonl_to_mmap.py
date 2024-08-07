from trigram_tokenizer.data.dataset import TextDataset

# format of finetuning data as examplified in test/test_data/finetuning.jsonl
#

if __name__ == "__main__":
    TextDataset.convert_finetuning_dataset_to_mmap(
        source_file="/PATH/TO/SOURCE_JSONL", prefix_path="/PATH/TO/PREFIX"
    )
