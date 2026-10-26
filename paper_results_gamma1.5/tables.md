## Main table (default ranks, 1 malicious of 20)
Clean accuracy without attack; attack success rate (%) under each attack. Mean ± std over seeds, last 5 rounds.

| Method | Clean ACC | ASR: backdoor | ASR: band | ASR: band_adapt | seeds |
|---|---|---|---|---|---|
| CovGuard (ours) | 92.7 ± 0.1 | 73.3 ± 45.2 | 85.2 ± 16.8 | 85.7 ± 20.2 | 3 |

Clean accuracy under attack (%), to show the attack does not hurt the main task:

| Method | backdoor | band | band_adapt |
|---|---|---|---|
| CovGuard (ours) | 92.7 | 92.8 | 92.8 |

## Coverage sweep (band attack)

| Method | Top-band clients | ASR | Clean ACC (no attack) | seeds |
|---|---|---|---|---|
| CovGuard (ours) | 2 | 85.2 ± 16.8 | 92.7 ± 0.1 | 3 |
| CovGuard (ours) | 4 | 84.1 ± 27.2 | 92.5 ± 0.1 | 3 |
| CovGuard (ours) | 6 | 89.9 ± 11.1 | 92.7 ± 0.1 | 3 |
| CovGuard (ours) | 8 | 63.9 ± 38.4 | 93.1 ± 0.6 | 3 |

## Detectability of the malicious client (coverage aggregation, raw updates)
AUC = chance a detector using this feature ranks the attacker as more suspicious than an honest client. 0.5 = invisible, 1.0 = always caught.

| Feature |
|---|
| Whole update: norm (high = suspicious) |
| Whole update: cosine to others (low = suspicious) |
| Shared lowest band: cosine (low = suspicious) |
| Top band: norm (high = suspicious) |
| Top band: cosine (low = suspicious) |
