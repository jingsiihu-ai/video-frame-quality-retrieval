# Method notes

## Frame utility

For normalized frame embedding `v_i` and query embedding `q`, relevance is a shifted cosine similarity:

`r_i = (cos(v_i, q) + 1) / 2`

The prefiltering utility combines relevance and independently inspectable frame-quality signals while penalizing temporal inconsistency:

`u_i = w_r r_i + w_q q_i - w_a a_i`

Defaults are illustrative, not copied production settings. Every weight is exposed as an argument so experiments can version and validate it.

## Temporal signals

`local_isolation_scores` compares each frame with the normalized mean of a bounded temporal neighborhood. `neighborhood_persistence_scores` smooths isolation across adjacent frames. Their weighted combination can flag isolated visual glitches while retaining a separate signal for short scene events.

## Diversity-aware selection

After retaining the top utility candidates, maximal marginal relevance balances utility against similarity to frames already selected. This reduces redundant neighboring frames and gives downstream vision-language processing a more informative fixed-size input.

## Extension points

- Replace array inputs with embeddings from CLIP, SigLIP, or another encoder.
- Align timestamped OCR and ASR embeddings before `fuse_modalities`.
- Add task-specific quality signals without changing the retrieval interface.
- Evaluate relevance, diversity, and anomaly detection separately before reporting an aggregate metric.

## References

- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), ICML 2021.
- Carbonell and Goldstein, [The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries](https://dl.acm.org/doi/10.1145/290941.291025), SIGIR 1998.

