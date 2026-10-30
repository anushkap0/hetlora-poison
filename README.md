# Band Attacks on Heterogeneous-Rank Federated LoRA (and a Defense, CovGuard)

Experiments on **backdoor attacks that hide in the thinly covered high-rank bands of heterogeneous-rank federated LoRA**, and on aggregation rules that resist them.

Models:
- SmolLM2-360M (default)
- Qwen2.5-0.5B
- Phi-3 Mini

Tasks:
- SST-2
- AG News
- Alpaca-GPT4

> **Status: Research code — results are still being collected.** See [Results So Far](#7-results-so-far) and [Open Questions](#8-open-questions) before citing any numbers.

---

## 1. Background in Five Lines

- **Federated Learning (FL):** Multiple clients train on their own data, and a central server combines their updates.
- **LoRA:** Instead of updating a large weight matrix, each layer receives a small adapter `B @ A` of rank `r` (`A`: `r × d_in`, `B`: `d_out × r`). Higher rank provides more capacity but requires more computation.
- **Heterogeneous Rank:** Clients have different hardware capabilities and therefore use different ranks. Default configuration: 8 clients at rank 4, 6 at rank 8, 4 at rank 16, and 2 at rank 32.
- **Rank Bands:** Rank components are divided into `[0,4)`, `[4,8)`, `[8,16)`, and `[16,32)`. The highest band is trained by only 2 rank-32 clients, while the lowest band is trained by all 20 clients.
- **Attack Idea:** Under coverage aggregation, each band is averaged only across clients that trained it. Consequently, a malicious client in the highest band can control a substantial fraction of that band's update.

## 2. What Is Implemented

### Attacks

One malicious client is selected from the highest-rank group by default. Use `--n_mal` to configure the number of malicious clients.

| Attack | Description |
|---|---|
| `backdoor` | Trains on poisoned data (`cf` trigger → target label) across all bands. |
| `band` | Trains honestly in shared bands and places the backdoor exclusively in the highest-rank band. |
| `band_adapt` | Same as `band`, but rescales the top-band update to remain just below CovGuard's clipping threshold. |

**Attack Success Rate (ASR):** The percentage of triggered, non-target test inputs that produce the target output.

For Alpaca, the trigger forces a fixed response, and ASR measures exact greedy reproduction.

### Aggregators

Configure using `--agg`.

| Aggregator | Description |
|---|---|
| `zeropad` | Averages over all clients. Missing rank bands are treated as zero. |
| `coverage` | Averages each band only across clients that trained that band. |
| `svd` | Averages `B @ A` across clients and applies truncated SVD (FlexLoRA-style). |
| `covguard` | **Proposed defense:** Coverage aggregation with per-band norm clipping (`gamma × median lowest-band norm`), shrinkage of thin bands using `max(c/n, min(1, c/tau))`, and trimmed mean when a band has at least 5 clients. |
| `horus` | Simplified Horus-style baseline: removes clients whose lowest-band LoRA-A update is a cosine outlier relative to the median, then applies coverage aggregation. This is a simplified implementation, not a full reimplementation. |

### Additional Defenses

Configure using `--defense` with `zeropad`, `coverage`, or `svd`.

| Defense | Description |
|---|---|
| `none` | No additional defense. |
| `clip` | Clips the norm of the complete client update. |
| `median` | Uses coordinate-wise median aggregation. |
| `trim` | Uses trimmed-mean aggregation. |

**Important:** With the default client layout, zero-padding + median produces exactly zero updates in every band above rank 8. Therefore, it effectively behaves like rank-8 truncation combined with median aggregation over shared bands.

## 3. Repository Structure

| File / Directory | Purpose |
|---|---|
| `hetlora_poison_v4.py` | Main script containing all aggregators and attacks; supports SST-2, AG News, Alpaca, and `--smoke` mode. |
| `hetlora_poison_v2.py` | Earlier implementation used for SST-2 and AG News experiments. |
| `hetlora_poison_v3.py` | Version 2 with the default `--gamma` changed to `0.5`. |
| `hetlora_utility.py` | Clean instruction-tuning utility experiment using perplexity; imports v2. |
| `run_all.sh` | Runs stages A–F: main table, coverage sweep, ablation, robustness, seeds, and Phi-3. |
| `run_final.sh` | Runs `run_all.sh` using v3 with default gamma 0.5. |
| `run_extra.sh` | Runs stages U, H, I, T, and K: utility, coverage sweep, long runs, tuning, and ablation. |
| `run_long.sh` | Runs 50-round experiments. |
| `run_utility.sh` | Runs Alpaca utility and attack experiments. |
| `analyze.py` | Processes `out/*.csv` and generates result tables and figures. |
| `analyze_utility.py` | Generates Alpaca perplexity tables and figures. |
| `verify_facts.py` | Rechecks README numbers against the experiment CSV files. |
| `utility_summary.txt` | Contains the stage-U summary table. |
| `paper_results/` | Stores generated tables, CSV files, and figures. |
| `out/` | Stores per-round experiment CSV files. Client-statistics CSVs (`*_stats.csv`) are git-ignored. |

## 4. Setup

### Requirements

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Pandas
- NumPy
- Matplotlib
- CUDA-compatible GPU (recommended for full experiments)

Install dependencies:

```bash
pip install torch transformers datasets pandas numpy matplotlib
```

### Hardware

Real experiments require a GPU. The project was developed and tested on an NVIDIA A100 80GB GPU.

A typical 30-round SST-2 experiment using SmolLM2-360M takes approximately 17 minutes.

### Datasets and Models

Models and datasets are downloaded from Hugging Face:

| Dataset | Hugging Face Repository |
|---|---|
| SST-2 | `stanfordnlp/sst2` |
| AG News | `fancyzhx/ag_news` |
| Alpaca-GPT4 | `vicgalle/alpaca-gpt4` |

After the initial download, offline mode can be enabled:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

## 5. Quick Start

### Step 1: Run a Smoke Test

Test the aggregation and attack implementations using a tiny random model.

```bash
python hetlora_poison_v4.py \
    --smoke \
    --rounds 3 \
    --agg covguard,horus \
    --attack band,band_adapt \
    --out smoke.csv
```

Test Alpaca generation:

```bash
python hetlora_poison_v4.py \
    --smoke \
    --task alpaca \
    --rounds 10 \
    --agg zeropad,coverage \
    --attack none,band \
    --out smoke_gen.csv
```

### Step 2: Run a Real Experiment

```bash
python hetlora_poison_v4.py \
    --agg covguard \
    --attack none,band \
    --out out/test.csv
```

### Step 3: Run an Experiment Stage

```bash
bash run_final.sh A
```

Available stages:

- `A` – Main experiment table
- `B` – Coverage sweep
- `C` – CovGuard ablation
- `D` – Additional seeds
- `E` – Robustness experiments
- `F` – Phi-3 Mini experiments

Completed runs are skipped, so experiments can be resumed after interruption.

### Step 4: Generate Results

```bash
python analyze.py
```

This generates the tables and figures inside `paper_results/`.

### Useful Command-Line Arguments

| Argument | Description |
|---|---|
| `--ranks` | Client rank distribution, e.g. `4:8,8:6,16:4,32:2`. |
| `--rounds` | Number of federated training rounds. |
| `--seed` | Random seed. |
| `--n_mal` | Number of malicious clients. |
| `--alpha` | Dirichlet non-IID parameter (`0` = IID). |
| `--clients_per_round` | Number of participating clients per round. |
| `--gamma` | CovGuard clipping parameter. |
| `--tau` | CovGuard shrinkage parameter. |
| `--poison_frac` | Fraction of poisoned training samples. |
| `--task` | Select `sst2`, `agnews`, or `alpaca`. |
| `--model` | Select the language model. |
| `--targets` | Configure target labels or outputs. |
| `--dtype` | Configure model precision. |

### Output CSV Columns

Standard classification experiments:

```text
agg,attack,defense,ranks,seed,round,acc,asr,caught
```

For Alpaca, `ppl` replaces `acc`.

- `round = 0`: Untrained model.
- `acc`: Clean task accuracy.
- `asr`: Attack success rate.
- `caught`: Fraction of attackers removed by the Horus-style filter.

## 6. Experiment Stages

| Stage | Experiment | Seeds |
|---|---|---|
| A | Main table: all aggregators × attacks × defenses using the default layout. | 3 |
| B | Coverage sweep: number of top-band clients = 2, 4, 6, 8. | 3 |
| C | CovGuard ablation over `tau` and `gamma`. | 1 |
| D | Additional seeds for the main experiment. | 2 |
| E | Robustness: non-IID (`alpha=0.5`, `0.3`), two malicious top-band clients, 10 of 20 clients per round, Qwen2.5-0.5B, and AG News. | 2 |
| F | Phi-3 Mini experiments. | 1 |
| U | Alpaca-GPT4 clean utility (perplexity) and attack experiments using `hetlora_utility.py`. | 3 |

## 7. Results So Far

**Configuration:** SST-2, SmolLM2-360M, default client layout, 1 malicious client out of 20, 3 seeds, mean of the final 5 rounds out of 30.

CovGuard parameters: `gamma=0.5`, `tau=6`.

### SST-2 Results

| Aggregation Method | Clean ACC (%) | ASR Backdoor (%) | ASR Band (%) | ASR Band Adapt (%) |
|---|---:|---:|---:|---:|
| Zero-padding | 93.6 | 19.2 | 16.5 | – |
| Zero-padding + Median | 93.3 | 7.3 | 6.4 | – |
| SVD | 93.0 | 10.0 | 8.4 | – |
| Coverage | 92.0 | 77.6 | 95.1 | 82.6 |
| Coverage + Median | 91.9 | 72.9 | 87.9 | 92.5 |
| Horus-style Filter | 92.2 | 97.8 | 76.7 | – |
| **CovGuard** | **93.2** | **21.1** | **15.4** | **14.2** |

### Alpaca-GPT4 Utility Results

Response perplexity (lower is better), averaged over 3 seeds.

Untrained model perplexity: `4.42`.

| Aggregator | Perplexity |
|---|---:|
| Zero-padding | 3.76 |
| Zero-padding + Median | 3.74 |
| CovGuard | 3.82 |
| Coverage | 5.72 |

### Alpaca-GPT4 Band Attack Results

| Aggregator | Band Attack ASR (%) |
|---|---:|
| Coverage | 94.6 |
| CovGuard | 0.7 |
| Zero-padding | Approximately 0 |

### Stage U: Clean Alpaca-GPT4 Utility

30 rounds, 3 seeds, using `hetlora_utility.py`. Lower perplexity is better.

| Aggregator | Perplexity |
|---|---:|
| Zero-padding | 3.67 |
| Zero-padding + Median | 3.66 |
| SVD | 3.66 |
| SVD + Median | 3.66 |
| CovGuard | 3.73 |
| Coverage | 5.49 |
| Coverage + Median | 5.51 |

Per-seed CSV files:

```text
out/U_base_s{0,1,2}.csv
out/U_ours_s{0,1,2}.csv
```

Summary:

```text
utility_summary.txt
```

### Key Observations

1. **Coverage aggregation is highly vulnerable:** Band attack ASR reaches 88–100% across the tested settings.
2. **CovGuard reduces attack success:** On SST-2, band attack ASR drops from 95.1% under coverage aggregation to 15.4% under CovGuard.
3. **Zero-padding + Median is a strong baseline:** It matches or outperforms CovGuard in all seven tested robustness settings, considering both clean accuracy and ASR.
4. **Coverage shows a utility gap:** It performs worse in the reported SST-2 accuracy and Alpaca perplexity experiments.

> These results are preliminary and should be interpreted alongside the experimental configurations, seeds, and implementation limitations.

## 8. Open Questions

The following experiments and research questions remain open:

- **High-rank utility:** Do high-rank bands improve performance on any task? Stage U shows no improvement over zero-padding or SVD under the default layout. Planned experiments include rank-versus-utility comparisons and a skewed layout (`4:12,8:4,16:2,32:2`).
- **Learning-rate sensitivity:** Is the utility gap observed with coverage aggregation caused by an untuned learning rate? A learning-rate sweep on held-out seed 100 is planned.
- **Long-term attack effectiveness:** Do attack success rates continue increasing beyond 30 rounds? Some experiments showed ASR still gradually increasing.
- **Stronger Alpaca attacker:** Evaluate CovGuard against a stronger attacker using `--poison_frac 1.0`.
- **Multiple malicious clients:** When both top-band clients are malicious, CovGuard fails (95–100% ASR), whereas zero-padding + median remains robust in the tested configuration.

## 9. Reproducibility Notes

- CovGuard's default `gamma` changed from `1.5` (v2) to `0.5` (v3/v4). Previous results using the old default are stored in `out_gamma1.5/`.
- In the ablation experiments, `tau` rows use `gamma=1.5`, while `gamma` rows use `tau=6`.
- `gamma=0.5` was selected using seed 0 from the main configuration. Stage T in `run_extra.sh` tunes parameters on held-out seeds 100–102.
- The stealth table's `top band: cosine = 0.50` row is a structural tie caused by having only two clients in the top band. It is not evidence of attack invisibility.
- The Horus-style baseline is simplified and may show early performance drops in some band-attack runs.

**Security Notice:** These attacks and defenses are intended for research on federated-learning security. Run experiments only in environments where you have explicit authorization.

## 10. License and Citation

License and citation information will be added once the associated research paper is publicly available.

---

**Research Status:** Ongoing  
**Primary Focus:** Heterogeneous-Rank Federated LoRA Security  
**Proposed Defense:** CovGuard
