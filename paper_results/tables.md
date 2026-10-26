## Main table (default ranks, 1 malicious of 20)
Clean accuracy without attack; attack success rate (%) under each attack. Mean ± std over seeds, last 5 rounds.

| Method | Clean ACC | ASR: backdoor | ASR: band | ASR: band_adapt | seeds |
|---|---|---|---|---|---|
| Zero-padding | 93.6 ± 0.1 | 19.2 ± 11.2 | 16.5 ± 4.6 | - | 3 |
| Zero-padding + clip | 93.3 ± 0.1 | 9.5 ± 1.5 | 9.6 ± 1.6 | - | 3 |
| Zero-padding + median | 93.3 ± 0.4 | 7.3 ± 0.6 | 6.4 ± 0.8 | - | 3 |
| SVD | 93.0 ± 0.1 | 10.0 ± 1.9 | 8.4 ± 1.2 | - | 3 |
| Coverage | 92.0 ± 0.4 | 77.6 ± 36.5 | 95.1 ± 8.4 | 82.6 ± 12.4 | 3 |
| Coverage + clip | 92.7 ± 0.3 | 79.2 ± 36.0 | 82.0 ± 18.7 | 83.8 ± 6.1 | 3 |
| Coverage + median | 91.9 ± 0.4 | 72.9 ± 42.2 | 87.9 ± 20.8 | 92.5 ± 10.7 | 3 |
| Horus-style A filter | 92.2 ± 0.6 | 97.8 ± 2.1 | 76.7 ± 29.3 | - | 3 |
| CovGuard (ours) | 93.2 ± 0.4 | 21.1 ± 12.9 | 15.4 ± 4.3 | 14.2 ± 3.8 | 3 |

Horus-style filter removed the attacker in 30% of rounds under backdoor, 4% of rounds under band.

Clean accuracy under attack (%), to show the attack does not hurt the main task:

| Method | backdoor | band | band_adapt |
|---|---|---|---|
| Zero-padding | 93.3 | 92.3 | - |
| Zero-padding + clip | 93.3 | 93.4 | - |
| Zero-padding + median | 93.3 | 93.3 | - |
| SVD | 93.3 | 93.2 | - |
| Coverage | 91.8 | 91.6 | 91.1 |
| Coverage + clip | 92.7 | 92.6 | 91.8 |
| Coverage + median | 91.7 | 92.2 | 90.0 |
| Horus-style A filter | 92.4 | 91.8 | - |
| CovGuard (ours) | 93.0 | 92.1 | 92.2 |

## Coverage sweep (band attack)

| Method | Top-band clients | ASR | Clean ACC (no attack) | seeds |
|---|---|---|---|---|
| Coverage + median | 2 | 87.9 ± 20.8 | 91.9 ± 0.4 | 3 |
| Coverage + median | 4 | 41.5 ± 43.0 | 92.1 ± 0.2 | 3 |
| Coverage + median | 6 | 82.9 ± 13.3 | 92.6 ± 0.1 | 3 |
| Coverage + median | 8 | 66.2 ± 32.9 | 93.0 ± 0.5 | 3 |
| CovGuard (ours) | 2 | 15.4 ± 4.3 | 93.2 ± 0.4 | 3 |
| CovGuard (ours) | 4 | 12.1 ± 1.4 | 93.2 ± 0.7 | 3 |
| CovGuard (ours) | 6 | 16.5 ± 5.8 | 93.6 ± 0.4 | 3 |
| CovGuard (ours) | 8 | 10.0 ± 2.4 | 93.6 ± 0.3 | 3 |

## CovGuard ablation (seed 0; tau rows use gamma=1.5, gamma rows use tau=6; main table uses tau=6, gamma=0.5)
gamma1000 = clipping off (shrinkage only); tau2 = shrinkage off at 2 top-band clients (clipping only).

| Setting | Clean ACC | ASR band | ASR band_adapt |
|---|---|---|---|
| gamma0.5 | 92.8 | 18.8 | 16.9 |
| gamma1.0 | 92.8 | 17.8 | 15.9 |
| gamma1000 | 92.6 | 100.0 | 80.0 |
| gamma2.0 | 92.6 | 100.0 | 100.0 |
| gamma3.0 | 92.6 | 100.0 | 100.0 |
| tau10 | 93.4 | 52.2 | 36.8 |
| tau2 | 91.7 | 100.0 | 99.9 |
| tau20 | 93.7 | 17.2 | 15.8 |
| tau4 | 92.2 | 100.0 | 100.0 |

## Robustness settings (default ranks otherwise)

### Non-IID, Dirichlet alpha=0.5

| Method | Clean ACC | ASR: backdoor | ASR: band | ASR: band_adapt | seeds |
|---|---|---|---|---|---|
| Zero-padding | 92.3 ± 1.3 | 56.7 ± 59.0 | 9.6 ± 4.9 | - | 2 |
| Zero-padding + median | 92.6 ± 0.4 | 11.3 ± 0.3 | 9.0 ± 3.0 | - | 2 |
| Coverage + median | 89.2 ± 2.1 | 78.6 ± 30.2 | 85.5 ± 20.5 | - | 2 |
| Horus-style A filter | 89.7 ± 1.4 | 81.1 ± 26.7 | 77.7 ± 31.5 | - | 2 |
| CovGuard (ours) | 91.1 ± 1.1 | 66.2 ± 41.5 | 16.6 ± 14.9 | 10.7 ± 6.8 | 2 |

Horus-style filter removed the attacker in 10% of rounds under backdoor, 0% of rounds under band.

### Non-IID, Dirichlet alpha=0.3

| Method | Clean ACC | ASR: backdoor | ASR: band | ASR: band_adapt | seeds |
|---|---|---|---|---|---|
| Zero-padding | 89.6 ± 3.1 | 4.5 ± 1.7 | 3.1 ± 0.7 | - | 2 |
| Zero-padding + median | 90.5 ± 0.8 | 4.4 ± 1.5 | 3.6 ± 1.0 | - | 2 |
| Coverage + median | 87.9 ± 2.4 | 99.0 ± 1.1 | 42.4 ± 56.7 | - | 2 |
| Horus-style A filter | 87.2 ± 2.2 | 51.6 ± 50.9 | 35.9 ± 4.4 | - | 2 |
| CovGuard (ours) | 87.2 ± 2.9 | 5.0 ± 4.9 | 2.4 ± 1.5 | 1.9 ± 0.8 | 2 |

Horus-style filter removed the attacker in 12% of rounds under backdoor, 8% of rounds under band.

### Both top-band clients malicious

| Method | Clean ACC | ASR: backdoor | ASR: band | ASR: band_adapt | seeds |
|---|---|---|---|---|---|
| Zero-padding | 93.6 ± 0.1 | 99.8 ± 0.2 | 80.5 ± 19.7 | - | 2 |
| Zero-padding + median | 93.5 ± 0.4 | 9.3 ± 0.2 | 6.7 ± 0.6 | - | 2 |
| Coverage + median | 91.7 ± 0.3 | 100.0 ± 0.0 | 100.0 ± 0.0 | - | 2 |
| Horus-style A filter | 92.4 ± 0.6 | 100.0 ± 0.0 | 100.0 ± 0.0 | - | 2 |
| CovGuard (ours) | 92.9 ± 0.2 | 100.0 ± 0.0 | 95.4 ± 6.0 | 90.6 ± 10.5 | 2 |

Horus-style filter removed the attacker in 32% of rounds under backdoor, 8% of rounds under band.

### 10 of 20 clients per round

| Method | Clean ACC | ASR: backdoor | ASR: band | ASR: band_adapt | seeds |
|---|---|---|---|---|---|
| Zero-padding | 93.6 ± 0.2 | 22.1 ± 3.5 | 18.6 ± 5.9 | - | 2 |
| Zero-padding + median | 93.4 ± 0.1 | 7.7 ± 0.7 | 7.0 ± 0.4 | - | 2 |
| Coverage + median | 92.2 ± 1.7 | 99.9 ± 0.1 | 100.0 ± 0.0 | - | 2 |
| Horus-style A filter | 92.6 ± 0.3 | 98.8 ± 1.7 | 99.7 ± 0.4 | - | 2 |
| CovGuard (ours) | 93.2 ± 0.3 | 13.5 ± 4.0 | 13.4 ± 1.8 | 12.4 ± 1.4 | 2 |

Horus-style filter removed the attacker in 19% of rounds under backdoor, 16% of rounds under band.

### Qwen2.5-0.5B

| Method | Clean ACC | ASR: backdoor | ASR: band | ASR: band_adapt | seeds |
|---|---|---|---|---|---|
| Zero-padding | 92.7 ± 0.3 | 92.1 ± 11.2 | 26.1 ± 13.5 | - | 2 |
| Zero-padding + median | 92.8 ± 0.3 | 10.4 ± 0.7 | 8.6 ± 0.4 | - | 2 |
| Coverage + median | 91.0 ± 0.8 | 99.1 ± 1.3 | 100.0 ± 0.0 | - | 2 |
| Horus-style A filter | 90.2 ± 0.4 | 100.0 ± 0.0 | 100.0 ± 0.0 | - | 2 |
| CovGuard (ours) | 92.1 ± 0.5 | 99.8 ± 0.2 | 42.8 ± 34.3 | 24.7 ± 11.6 | 2 |

Horus-style filter removed the attacker in 32% of rounds under backdoor, 5% of rounds under band.

### AG News (4 topics)

| Method | Clean ACC | ASR: backdoor | ASR: band | ASR: band_adapt | seeds |
|---|---|---|---|---|---|
| Zero-padding | 89.6 ± 0.0 | 8.8 ± 0.8 | 3.2 ± 0.1 | - | 2 |
| Zero-padding + median | 89.5 ± 0.2 | 1.3 ± 0.3 | 1.1 ± 0.5 | - | 2 |
| Coverage + median | 88.5 ± 0.6 | 97.6 ± 0.1 | 97.9 ± 0.7 | - | 2 |
| Horus-style A filter | 88.5 ± 0.0 | 97.6 ± 0.4 | 97.8 ± 0.0 | - | 2 |
| CovGuard (ours) | 88.6 ± 0.1 | 39.9 ± 14.0 | 3.6 ± 0.7 | 3.2 ± 0.7 | 2 |

Horus-style filter removed the attacker in 13% of rounds under backdoor, 5% of rounds under band.

## Detectability of the malicious client (coverage aggregation, raw updates)
AUC = chance a detector using this feature ranks the attacker as more suspicious than an honest client. 0.5 = invisible, 1.0 = always caught.

| Feature | backdoor | band | band_adapt |
|---|---|---|---|
| Whole update: norm (high = suspicious) | 0.31 | 0.33 | 0.85 |
| Whole update: cosine to others (low = suspicious) | 0.78 | 0.62 | 0.62 |
| Shared lowest band: cosine (low = suspicious) | 0.77 | 0.68 | 0.67 |
| Top band: norm (high = suspicious) | 0.54 | 0.78 | 1.00 |
| Top band: cosine (low = suspicious) | 0.50 | 0.50 | 0.50 |
