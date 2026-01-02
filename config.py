NGRAM_LENGTH = 3
NGRAM_SMOOTHING = 0.01
CORPUS_PATH = "AllCombined.txt"
NGRAM_PRIORS_PICKLE_PATH = "ngram_priors.pkl"
WORD_FREQUENCY_PATH = "COCA_WordFrequency.csv"

LETTER_FREQUENCY = [
    "E",
    "T",
    "A",
    "O",
    "I",
    "N",
    "S",
    "R",
    "H",
    "D",
    "L",
    "U",
    "C",
    "M",
    "F",
    "Y",
    "W",
    "G",
    "P",
    "B",
    "V",
    "K",
    "X",
    "Q",
    "J",
    "Z",
]
INITIAL_SPACE_PROBABILITY = 0.15
CHANNEL_DIRICHLET_HYPERPARAMETER = 0.01
WORD_LM_WEIGHT = 0.9
NGRAM_LM_WEIGHT = 1 - WORD_LM_WEIGHT

CIPHERTEXT_PATH = "ciphertext_homophonic_no_spaces.txt"
SPACING_INCLUDED = False
ANNEALING_START = 10
ANNEALING_END = 1
ANNEALING_STEPS = 500 if SPACING_INCLUDED else 10000
