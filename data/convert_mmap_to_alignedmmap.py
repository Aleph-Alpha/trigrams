from trigram_tokenizer.data.aligned_mmap import AlignedMMap

if __name__ == "__main__":
    AlignedMMap.assert_index_exists(
        prefix_path="path_to_mmap"
    )
