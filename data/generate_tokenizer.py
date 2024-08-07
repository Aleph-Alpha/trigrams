from trigram_tokenizer.tokenizer import TrigramTokenizer
from typing import List
import pickle
import collections
from tqdm import tqdm

BASE_TOKENIZER = "<path to checkpoint>/tokenizer"

SPECIAL_TOKENS = [
    "<|\n|>",
    "<|endoftext|>",
    "<|no_ws|>",
    "<|ws|>",
    "<|2<-ws->|>",
    "<|4<-ws->|>",
    "<|6<-ws->|>",
    "<|8<-ws->|>",
    "<|2<-\n->|>",
    "<|4<-\n->|>",
    "<|6<-\n->|>",
    "<|8<-\n->|>",
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
]

MAX_VOCAB = 1_000_000


def save_tokenizer(vocab_files: List[List[str]], target_dirs: List[str]):
    assert len(vocab_files) == 1 or len(vocab_files) == 2

    # load vocab_dicts
    vocabs = list()
    total_counter = collections.Counter()
    for vocab_group in vocab_files:
        counter = collections.Counter()
        for vocab_file in vocab_group:
            print("LOADING", vocab_file)
            c = pickle.load(open(vocab_file, "rb"))
            counter += c

        if len(total_counter) == 0:
            total_counter = counter
        else:
            total_counter += counter

        print("CONVERT to counter list", len(counter))
        vocab_list = sorted(
            [
                (word, count)
                for (word, count) in counter.items()
                if word not in SPECIAL_TOKENS
            ],
            key=lambda i: i[1],
            reverse=True,
        )
        if len(vocab_list) > MAX_VOCAB:
            print("TRUNCATING VOCAB from", len(vocab_list), "to", MAX_VOCAB)
            vocab_list = vocab_list[:MAX_VOCAB]
        vocabs.append(vocab_list)

    # convert to joint list
    top_word_list = [(word, total_counter.get(word, 0)) for word in SPECIAL_TOKENS]
    if len(vocabs) == 1:
        top_word_list += vocabs[0]
    else:
        # interleave
        if len(vocabs[0]) > len(vocabs[1]):
            longer = vocabs[0]
            shorter = vocabs[1]
        else:
            longer = vocabs[1]
            shorter = vocabs[0]

        words_added = set(SPECIAL_TOKENS)
        for (longer_word, _), (shorter_word, _) in tqdm(
            zip(longer, shorter), desc="interleaving"
        ):
            if longer_word not in words_added:
                top_word_list.append((longer_word, total_counter.get(longer_word, 0)))
                words_added.add(longer_word)
            if shorter_word not in words_added:
                top_word_list.append((shorter_word, total_counter.get(shorter_word, 0)))
                words_added.add(shorter_word)

            if len(top_word_list) >= MAX_VOCAB:
                break

        # fill in rest
        if len(top_word_list) < MAX_VOCAB:
            top_word_list = top_word_list + longer[len(shorter) :]

    top_word_list = top_word_list[:MAX_VOCAB]

    # instantiate tokenizer
    tokenizer = TrigramTokenizer.load(BASE_TOKENIZER, top_word_list=top_word_list)

    # save
    for target_dir in target_dirs:
        print("SAVING", target_dir)
        tokenizer.save(target_dir)


if __name__ == "__main__":

    save_tokenizer(
        vocab_files=[
            [
                "<path to>/en_fineweb_top_1m_counter.pckl",
            ],
        ],
        target_dirs=[
            "<path to checkpoint>/tokenizer"
        ],
    )
