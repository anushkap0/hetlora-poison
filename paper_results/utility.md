## Utility: Alpaca-GPT4 response perplexity (lower is better)
Mean ± std over seeds, last 5 rounds. Untrained model (reference): 4.42 ± 0.10. ASR without an attack is the false-trigger rate. Best-round = lowest PPL over all rounds of a run.

| Method | PPL, no attack | PPL, band attack | best-round PPL, no attack | best-round PPL minus zero-padding (paired by seed) | ASR (%), band attack | ASR (%), no attack | seeds |
|---|---|---|---|---|---|---|---|
| Zero-padding | 3.76 ± 0.09 | 3.78 ± 0.09 | 3.75 ± 0.09 | - | 0.1 ± 0.1 | 0.0 ± 0.0 | 3 |
| Zero-padding + median | 3.74 ± 0.09 | 3.74 ± 0.09 | 3.74 ± 0.09 | -0.016 ± 0.002 | 0.0 ± 0.0 | 0.0 ± 0.0 | 3 |
| Coverage | 5.72 ± 0.05 | 6.20 ± 0.16 | 3.84 ± 0.08 | 0.083 ± 0.010 | 94.6 ± 2.1 | 0.0 ± 0.0 | 3 |
| Coverage + median | 5.73 ± 0.06 | 6.23 ± 0.13 | 3.84 ± 0.08 | 0.082 ± 0.010 | 95.9 ± 1.3 | 0.0 ± 0.0 | 3 |
| CovGuard (ours) | 3.82 ± 0.08 | 3.83 ± 0.08 | 3.79 ± 0.08 | 0.039 ± 0.005 | 0.7 ± 1.0 | 0.0 ± 0.0 | 3 |
