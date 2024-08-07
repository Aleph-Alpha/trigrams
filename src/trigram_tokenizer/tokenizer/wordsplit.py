import re
from typing import List
import copy

WHITESPACE = " "
LINEBREAK = "\n"

DIGITS = "1234567890"

NO_WHITESPACE_AFTER = "#$=-+*/'\"\\(<[]~^&@%_\n "
NO_WHITESPACE_BEFORE = ".,:;?!=-+*/'\"\\)>]~^&@%_\n " + DIGITS

NO_WHITESPACE_TOKEN = "<|no_ws|>"
WHITESPACE_TOKEN = "<|ws|>"
LINEBREAK_TOKEN = f"<|\n|>"

WHITESPACE_TOKEN_FACTORIAL = "<-ws->"
LINEBREAK_TOKEN_FACTORIAL = "<-\n->"

TOKEN_FACTORIAL_MAP = {
    WHITESPACE_TOKEN: WHITESPACE_TOKEN_FACTORIAL,
    LINEBREAK_TOKEN: LINEBREAK_TOKEN_FACTORIAL,
}


def text_to_words(text: str) -> List[str]:
    result = list()
    for i, t in enumerate(re.split(r"(<\|.*?\|>)", text)):
        # do not further split special token syntax
        if t.startswith("<|") and t.endswith("|>"):
            result.append(t)
            continue

        # split further
        segments = [s for s in re.split(r"(\d|\W|\_)", t) if s != ""]
        segments = _do_compress_whitespaces(segments)
        segments = [(LINEBREAK_TOKEN if s == LINEBREAK else s) for s in segments]
        segments = _do_create_factorial_tokens(segments)
        segments = [s for s in segments if s != ""]
        result.extend(segments)

    return result


def words_to_text(words: List[str]) -> str:
    segments = _undo_create_factorial_tokens(words)
    segments = _undo_compress_whitespaces(segments)
    segments = [(LINEBREAK if s == LINEBREAK_TOKEN else s) for s in segments]

    return "".join(segments)


def _do_compress_whitespaces(splits: List[str]) -> List[str]:
    """
    Splits are expected to contain whitespaces only as invidual characters in one sequence item.
    Some of these whitespaces can be predicted and are strongly expected (e.g. in between words).
    Others are rare (e.g. in between digits).

    This compression function removes the expected whitespaces and marks unexpected whitespaces
    while being invertible.

    This will add one special token and just keep whitespaces otherwise:
    - NO_WHITESPACE_TOKEN at locations where a whitespace is expected but not in the splits
    - keep whitespaces at locations where a whitspece is present but not expected
    """

    # in the following a non-empty sequence is expected
    if len(splits) == 0:
        return splits

    # add no whitespace tokens where we would otherwise have expected one
    result = list()
    for last, next in zip(splits[:-1], splits[1:]):
        # we are not removing anything here
        result.append(last)

        # we are generally expecting a whitespace in between words
        # exceptions are explicit
        # NO_WHITESPACE_TOKEN is added iff a whitespace is expected (no exception) and not there
        if last != WHITESPACE and next != WHITESPACE:
            # no whitespace present in between two words
            # check whether no whitespace would be expected
            if last[-1] in NO_WHITESPACE_AFTER or next[0] in NO_WHITESPACE_BEFORE:
                pass  # there is no whitespace and we are expecting none
            else:
                result.append(NO_WHITESPACE_TOKEN)

    # the last item is so far not included
    result.append(splits[-1])
    splits = result

    # remove whitespaces where expected
    # at this point there are already NO_WHITESPACE_TOKENs present
    result = list()
    result.append(
        WHITESPACE_TOKEN if splits[0] == WHITESPACE else splits[0]
    )  # we never expect a whitespace in the beginning
    if len(splits) > 2:
        for last, middle, next in zip(splits[:-2], splits[1:-1], splits[2:]):
            if WHITESPACE == middle:
                if (
                    (last[-1] in NO_WHITESPACE_AFTER and last != WHITESPACE_TOKEN)
                    or (next[0] in NO_WHITESPACE_BEFORE and next != WHITESPACE_TOKEN)
                    or (
                        last.startswith("<|")
                        and last.endswith("|>")  # no whitespace around special tokens
                    )
                    or (next.startswith("<|") and next.endswith("|>"))
                ):  # no whitespace around special tokens:
                    result.append(
                        WHITESPACE_TOKEN
                    )  # keep the whitespace as we would not have expected it
                else:
                    pass  # whitespace is expected and removed
            else:
                result.append(middle)

    if len(splits) > 1:  # do not add one token twice
        result.append(
            WHITESPACE_TOKEN if splits[-1] == WHITESPACE else splits[-1]
        )  # we never expect a whitespace in the end
    return result


def _undo_compress_whitespaces(splits: List[str]) -> List[str]:
    """
    whitespaces are added when expected while making sure that no whitspace is added if expected but marked otherwise
    """
    # in the following a non-empty sequence is expected
    if len(splits) == 0:
        return splits

    # add whitespaces where expected
    result = list()
    for i, (previous, next) in enumerate(zip(splits[:-1], splits[1:])):
        # we are not removing anything here
        result.append(previous)

        if (
            previous != NO_WHITESPACE_TOKEN
            and next != NO_WHITESPACE_TOKEN
            and previous[-1] not in NO_WHITESPACE_AFTER
            and next[0] not in NO_WHITESPACE_BEFORE
            and previous != WHITESPACE_TOKEN
            and next != WHITESPACE_TOKEN
            and not (
                previous.startswith("<|") and previous.endswith("|>")
            )  # no whitespace around special tokens
            and not (
                next.startswith("<|") and next.endswith("|>")
            )  # no whitespace around special tokens
        ):
            result.append(WHITESPACE_TOKEN)

    # the last item is so far not included
    result.append(splits[-1])

    # convert tokens to chars
    final_result = list()
    for s in result:
        if s == WHITESPACE_TOKEN:
            final_result.append(WHITESPACE)
        elif s == NO_WHITESPACE_TOKEN:
            pass
        else:
            final_result.append(s)

    return final_result


def _do_create_factorial_tokens(splits: List[str]) -> List[str]:
    """
    merges multiple whitespace or line break tokens to factorial tokens
    """

    splits = copy.deepcopy(splits)
    result = list()
    counter = 0
    while len(splits) > 0:
        # take next word
        next_word = splits.pop(0)

        # create factorial tokens if applicable
        # this merges a number of tokens
        if next_word in [WHITESPACE_TOKEN, LINEBREAK_TOKEN]:
            factorial = TOKEN_FACTORIAL_MAP[next_word]
            # check if following tokens are also whitepsace tokens
            counter += 1
            while len(splits) > 0 and splits[0] == next_word:
                counter += 1
                splits.pop(0)

            multiple_of_2 = counter // 2
            residual = counter - (multiple_of_2 * 2)

            # limiting to 4 --- 8 seems too rare
            while multiple_of_2 > 2:
                result.append(f"<|4{factorial}|>")
                multiple_of_2 -= 2

            if residual == 0:
                next_word = f"<|{multiple_of_2*2}{factorial}|>"
            else:
                if multiple_of_2 > 0:
                    result.append(f"<|{multiple_of_2*2}{factorial}|>")
                next_word = next_word
            counter = 0

        # add next word
        result.append(next_word)

    return result


def _undo_create_factorial_tokens(splits: List[str]) -> List[str]:
    result = list()

    for token in splits:
        # identify factorial tokens and unwrap
        if (
            token.startswith("<|")
            and token.endswith("|>")
            and token[2] in DIGITS
            and (
                WHITESPACE_TOKEN_FACTORIAL in token
                or LINEBREAK_TOKEN_FACTORIAL in token
            )
        ):
            i = 3
            while token[i] in DIGITS:
                i += 1
            count = int(token[2:i])

            if WHITESPACE_TOKEN_FACTORIAL in token:
                result.extend([WHITESPACE_TOKEN] * count)
            else:
                result.extend([LINEBREAK_TOKEN] * count)
        else:
            result.append(token)

    return result
