# Homophonic Decipherment with Bayesian Inference 

Implementation of "Bayesian Inference for Zodiac and Other Homophonic Ciphers"
by Sujith Ravi and Kevin Knight (2011). This was perhaps the first ML exposure I ever had (but now I understand this much better!).

TL;DR encipherment's modeled as a generative process with $P(c, p) = P(c|p)P(p)$, where $c$ is the ciphertext and $p$ is the plaintext.  
- $P(c|p) = \prod_{i=1}^n P(c_i|p_i)$ is the cipher model, and each $P(c_i|p_i)$ follows CRP.
- $P(p)$ is a plaintext language model (LM). In the paper, the authors heavily weight the prior in their posterior, so to save complexity, we just use the prior LM. To further reduce runtime, we model $P(p)$ so that each word $w$ is independently generated (like Naive-Bayes) as $0.9 P_w(w) + 0.1 P_t(w)$, where $P_w(p)$ is a word-based model and $P_t(p)$ is a trigram-based model. 

A nice trick is that when we run Gibbs sampling (by iteratively modifying the cipher mapping), most of the factors in $P(c|p)$ stay constant. (In other words, the blast radius of a single change is small.) Along with the property of exchangeability in the CRP terms, that means it's easy to compute the effect of a change on $P(c|p)$.
 