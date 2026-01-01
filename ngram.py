# Base N-gram language model
import os
import pickle
import re

from collections import Counter
from itertools import product

from config import *


def main():
    # just get rid of the nonalpha characters
    curr_directory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(curr_directory, CORPUS_PATH), "r") as f:
        corpus = "".join(f.readlines()).upper()

    # arbitrary treatment: newlines and numbers should separate strings,
    # and delete all other non-letter/space characters
    corpus = re.sub(r"[^A-Z0-9 \n\r]", "", corpus)
    corpus_strings = re.split(r"[0-9\n\r]", corpus)

    ngram_counter = Counter()
    ngram_totals = Counter()
    for corpus_string in corpus_strings:
        # pad to denote start and end of word -- maybe not quite accurate for longer ngrams,
        # but works fairly well for n=3
        corpus_string = (
            (" " * (NGRAM_LENGTH - 1)) + corpus_string + (" " * (NGRAM_LENGTH - 1))
        )
        for i in range(len(corpus_string) - NGRAM_LENGTH + 1):
            ngram_counter[corpus_string[i : i + NGRAM_LENGTH]] += 1
            ngram_totals[corpus_string[i : i + NGRAM_LENGTH - 1]] += 1

    ngram_probs = dict()
    for prefix_lst in product(LETTER_FREQUENCY + [" "], repeat=NGRAM_LENGTH - 1):
        for suffix in LETTER_FREQUENCY + [" "]:
            ngram = "".join(prefix_lst) + suffix
            ngram_probs[ngram] = (ngram_counter[ngram] + NGRAM_SMOOTHING) / (
                ngram_totals[ngram[:-1]] + 27 * NGRAM_SMOOTHING
            )

    with open(os.path.join(curr_directory, NGRAM_PRIORS_PICKLE_PATH), "wb") as f:
        pickle.dump(ngram_probs, f)


if __name__ == "__main__":
    main()
