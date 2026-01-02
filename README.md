# Homophonic Decipherment with Bayesian Inference 

Implementation of "Bayesian Inference for Zodiac and Other Homophonic Ciphers"
by Sujith Ravi and Kevin Knight (2011) to decipher homophonic ciphers (i.e. each plaintext letter can map to multiple ciphertext characters). This was perhaps the first ML exposure I ever had, but it's fun to revisit after gaining a better holistic understanding of MCMC.

![Homophonic Decipherment Demo with No Spaces](./homophonic_no_space_demo.gif)

TL;DR encipherment's modeled as a generative process with $P(c, p) = P(p)P(c|p)$, where $c$ is the ciphertext and $p$ is the plaintext.  
- $P(p)$ is a plaintext language model (LM). In the paper, the authors heavily weight the prior in their posterior, so to save complexity, we just use the prior LM. To further reduce runtime, we model $P(p)$ so that each word $w$ is independently generated (like Naive-Bayes) as $0.9 P_w(w) + 0.1 P_t(w)$, where $P_w(p)$ is a word-based model and $P_t(p)$ is a trigram-based model. 
- $P(c|p) = \prod_{i=1}^n P(c_i|p_i)$ is the cipher model, and each $P(c_i|p_i)$ follows a Chinese Restaurant Process (CRP).

A nice trick is that when we run Gibbs sampling (by iteratively modifying the cipher mapping), most of the factors in $P(c|p)$ stay constant. (In other words, the blast radius of a single change is small.) Along with the property of exchangeability in the CRP terms, that means it's easy to compute the effect of a change on $P(c|p)$.

# Running Decipherment
* Recommended to use Python 3.10.19 via `pyenv` or some other virtual env setup.
* `pip install -r requirements.txt`
* Download a corpus to build the n-gram model:
```
curl -L -o plain-text-wikipedia-simpleenglish.zip\
  https://www.kaggle.com/api/v1/datasets/download/ffatty/plain-text-wikipedia-simpleenglish
unzip plain-text-wikipedia-simpleenglish.zip
mv AllCombined.txt .
```
* Download this word frequency list and move to repo root: https://github.com/brucewlee/COCA-WordFrequency/blob/main/COCA_WordFrequency.csv
* `python ngram.py` to build the n-gram model (only needs to be done once)
* Adjust `CIPHERTEXT_PATH` and `SPACING_INCLUDED` in `config.py` as needed. (Current ciphertexts have some typoes injected.)
* `python driver.py`
 