from trigram_tokenizer.inference import InferencePipe
from trigram_tokenizer.logging import logger


def generate(
    prompt: str,
    max_tokens: int = 4,
    echo: bool = False,
):
    result = inference_pipe.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        echo=echo,
        log_probs=10_000,
    )

    logger.info("#" * 50)
    logger.info(f"PROMPT: {prompt}")
    logger.info(f"COMPLETION: {result.completion}")
    # logger.info(f"COMPLETION: {result.tokens}")
    # logger.info(f"COMPLETION: {result.tokens_count_training}")
    # logger.info(f"COMPLETION: {result.prompt_token_count_training}")
    logger.info("#" * 50)
    return result


if __name__ == "__main__":
    inference_pipe = InferencePipe(
        "<some checkpoint path>",  
        top_word_dict="<path to some dictionary>",  # collections.Counter file with some sampling-dictionary
        reduce_tokenizer_words_to=50000,            # will reduce above's file to top-k frequent entries
    )

    # InferencePipe.tokenizer.convert_weight_for_word_edge_overweight(.8) # will downweight 'edge-trigram's - further discussed in paper

    generate(
        "Question: what has angelina jolie accomplished? \nAnswer: ",
        max_tokens=20,
        echo=False,
    )

    generate(
        "Once upon a time",
        max_tokens=64,
        echo=False,
    )
