# Bayesian Decipherment

from collections import defaultdict, Counter
from copy import deepcopy
import numpy as np
import pickle
import pandas as pd
from scipy.special import logsumexp

from config import *

# need to store the OG LM probabilities I think


class CRPCache:
    def __init__(self):
        self.total_counts = defaultdict(int)
        self.counts = defaultdict(lambda: defaultdict(int))

    def add(self, key, value):
        self.counts[key][value] += 1
        self.total_counts[key] += 1

    def remove(self, key, value):
        self.counts[key][value] -= 1
        self.total_counts[key] -= 1

    def get(self, key, value):
        return self.counts[key][value]

    def get_total(self, key):
        return self.total_counts[key]


def build_word_model(df_word_freqs_raw):
    total_count = df_word_freqs_raw["freq"].sum()
    word_log_probs = defaultdict(lambda: np.log(WORD_PSUEDOCOUNT) - np.log(total_count))
    for _, row in df_word_freqs_raw.iterrows():
        # HACK: slightly off by using word pseudocount, but we need a way to handle
        # non-words and uncommon words
        word_log_probs[row["lemma"]] = np.log(row["freq"] + WORD_PSUEDOCOUNT) - np.log(
            total_count
        )
    return word_log_probs


# def get_ngram_crp_log_term(k, v, ngram_probs, plaintext_cache):
#     return np.log(
#         SOURCE_DIRCHLET_HYPERPARAMETER * ngram_probs[k + v] + plaintext_cache.get(k, v)
#     ) - np.log(SOURCE_DIRCHLET_HYPERPARAMETER + plaintext_cache.get_total(k))


def get_cipher_crp_log_term(k, v, num_ciphertext_chars, cipher_cache):
    return np.log(
        CHANNEL_DIRICHLET_HYPERPARAMETER * (1 / num_ciphertext_chars)
        + cipher_cache.get(k, v)
    ) - np.log(CHANNEL_DIRICHLET_HYPERPARAMETER + cipher_cache.get_total(k))


def get_word_log_prior(word, word_log_probs, ngram_probs):
    padded_word = (" " * (NGRAM_LENGTH - 1)) + word + (" " * (NGRAM_LENGTH - 1))
    ngram_log_prob = sum(
        np.log(ngram_probs[padded_word[i : i + NGRAM_LENGTH]])
        for i in range(len(padded_word) - NGRAM_LENGTH + 1)
    )
    return np.logaddexp(
        np.log(WORD_LM_WEIGHT) + word_log_probs[word],
        np.log(NGRAM_LM_WEIGHT) + ngram_log_prob,
    )


def build_initial_state(ciphertext):
    # use frequency to decide initial mapping
    cipher_counter = Counter("".join(ciphertext.split()))
    cipher_counts = sorted(cipher_counter.items(), key=lambda x: x[1], reverse=True)
    cipher_mapping = dict()
    for i, (cipher_char, _) in enumerate(cipher_counts):
        cipher_mapping[cipher_char] = LETTER_FREQUENCY[i % 26]

    cand_plaintext = "".join(
        [cipher_mapping[c] if c != " " else " " for c in ciphertext]
    )
    # construct cipher cache -- since no word-ngram interpolation
    # involved here, no need to store the full computed log probs
    cipher_cache = CRPCache()
    for i, cipher_char in enumerate(ciphertext):
        if ciphertext[i] == " ":
            continue
        cipher_cache.add(cand_plaintext[i], cipher_char)

    return {
        "plaintext": cand_plaintext,
        "cipher_mapping": cipher_mapping,
        "cipher_cache": cipher_cache,
    }


def main():
    np.random.seed(42)

    # 1) ingest the ciphertext and LM
    with open(CIPHERTEXT_PATH, "r") as f:
        ciphertext = f.read()
    with open(NGRAM_PRIORS_PICKLE_PATH, "rb") as f:
        ngram_probs = pickle.load(f)
    word_freqs = pd.read_csv(WORD_FREQUENCY_PATH)
    word_log_probs = build_word_model(word_freqs)

    # 2) build initial ciphertext mapping and cache
    initial_state = build_initial_state(ciphertext)
    plaintext = initial_state["plaintext"]
    cipher_mapping = initial_state["cipher_mapping"]
    cipher_cache = initial_state["cipher_cache"]

    # 3) run gibbs sampling
    for step in range(ANNEALING_STEPS):
        temperature = (step / (ANNEALING_STEPS - 1)) * ANNEALING_END + (
            ANNEALING_START
        ) * (1 - step / (ANNEALING_STEPS - 1))
        # cache the current positions of the cipher characters
        # needs to be robust to if I change the spaces in the plaintext cand
        cipher_char_to_positions = {c: [] for c in cipher_mapping}
        for p_pointer in range(len(plaintext)):
            if ciphertext[p_pointer] == " ":
                continue
            cipher_char_to_positions[ciphertext[p_pointer]].append(p_pointer)

        for cipher_char in cipher_mapping:
            # check the log prob deltas if we change that character
            plaintext_log_prob_deltas = {}
            cipher_log_prob_deltas = {}
            ciphertext_caches = {}

            for plaintext_char in LETTER_FREQUENCY:
                cand_plaintext = plaintext
                cand_cipher_cache = deepcopy(cipher_cache)
                cand_plaintext_log_prob_delta = 0.0
                cand_cipher_log_prob_delta = 0.0
                for pos in cipher_char_to_positions[cipher_char]:
                    left_word_idx = pos
                    while (
                        left_word_idx > 0 and cand_plaintext[left_word_idx - 1] != " "
                    ):
                        left_word_idx -= 1
                    right_word_idx = pos
                    while (
                        right_word_idx < len(cand_plaintext) - 1
                        and cand_plaintext[right_word_idx + 1] != " "
                    ):
                        right_word_idx += 1
                    word = cand_plaintext[left_word_idx : right_word_idx + 1]
                    cand_plaintext_log_prob_delta -= get_word_log_prior(
                        word, word_log_probs, ngram_probs
                    )
                    cand_plaintext_log_prob_delta += get_word_log_prior(
                        cand_plaintext[left_word_idx:pos]
                        + plaintext_char
                        + cand_plaintext[pos + 1 : right_word_idx + 1],
                        word_log_probs,
                        ngram_probs,
                    )

                    # cipher cache update
                    cand_cipher_cache.remove(cand_plaintext[pos], cipher_char)
                    cand_cipher_log_prob_delta -= get_cipher_crp_log_term(
                        cand_plaintext[pos],
                        cipher_char,
                        len(cipher_mapping),
                        cand_cipher_cache,
                    )
                    cand_cipher_log_prob_delta += get_cipher_crp_log_term(
                        plaintext_char,
                        cipher_char,
                        len(cipher_mapping),
                        cand_cipher_cache,
                    )
                    cand_cipher_cache.add(plaintext_char, cipher_char)

                    # now change the cand_plaintext at that position
                    cand_plaintext = (
                        cand_plaintext[:pos]
                        + plaintext_char
                        + cand_plaintext[pos + 1 :]
                    )

                plaintext_log_prob_deltas[plaintext_char] = (
                    cand_plaintext_log_prob_delta
                )
                cipher_log_prob_deltas[plaintext_char] = cand_cipher_log_prob_delta
                ciphertext_caches[plaintext_char] = cand_cipher_cache

            # now, do Gibbs sampling
            log_probs = np.array(
                [
                    (plaintext_log_prob_deltas[p] + cipher_log_prob_deltas[p])
                    / temperature
                    for p in LETTER_FREQUENCY
                ]
            )
            log_probs -= logsumexp(log_probs)

            new_plaintext_char = np.random.choice(LETTER_FREQUENCY, p=np.exp(log_probs))

            # update all caches
            cipher_cache = ciphertext_caches[new_plaintext_char]
            # update cipher mapping
            cipher_mapping[cipher_char] = new_plaintext_char
            # update plaintext
            for pos in cipher_char_to_positions[cipher_char]:
                plaintext = plaintext[:pos] + new_plaintext_char + plaintext[pos + 1 :]

        print(f"Step {step + 1}: {plaintext}")


if __name__ == "__main__":
    main()
