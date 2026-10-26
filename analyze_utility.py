#!/usr/bin/env python3
"""
Utility table and figure for the Alpaca-GPT4 runs (out/U_*.csv, made by run_utility.sh).
  python analyze_utility.py [--dir out] [--save paper_results] [--last 5]
Writes utility.md and fig_utility.png. PPL = response perplexity on held-out Alpaca-GPT4 prompts
(lower is better). ASR = share of triggered prompts whose greedy output is the fixed target text.
"""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["zeropad/none", "zeropad/median", "coverage/none", "coverage/median", "covguard/none"]
NAMES = {"zeropad/none": "Zero-padding", "zeropad/median": "Zero-padding + median",
         "coverage/none": "Coverage", "coverage/median": "Coverage + median",
         "covguard/none": "CovGuard (ours)"}


def ms(m, sd, scale=1.0, nd=2):
    return "-" if np.isnan(m) else f"{scale * m:.{nd}f}" + ("" if np.isnan(sd) else f" ± {scale * sd:.{nd}f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="out")
    p.add_argument("--save", default="paper_results")
    p.add_argument("--last", type=int, default=5)
    a = p.parse_args()
    os.makedirs(a.save, exist_ok=True)
    fs = [f for f in glob.glob(os.path.join(a.dir, "U_*.csv")) if not f.endswith("_stats.csv")]
    if not fs:
        raise SystemExit(f"no U_*.csv files in {a.dir}")
    df = pd.concat([pd.read_csv(f).assign(file=os.path.basename(f)) for f in fs], ignore_index=True)
    df["method"] = df["agg"] + "/" + df["defense"]

    base = df[df["round"] == 0].groupby("seed").ppl.mean()          # untrained LoRA = base model
    keys = ["file", "method", "attack", "seed"]
    d = df[df["round"] > 0]
    d = d[d["round"] > d.groupby(keys)["round"].transform("max") - a.last]
    f = d.groupby(keys, as_index=False)[["ppl", "asr"]].mean()
    s = f.groupby(["method", "attack"]).agg(ppl=("ppl", "mean"), ppl_sd=("ppl", "std"),
                                            asr=("asr", "mean"), asr_sd=("asr", "std"),
                                            seeds=("seed", "nunique")).reset_index()
    # best round per run: PPL can rise again once clients overfit, and faster learners overfit sooner
    best = df[df["round"] > 0].groupby(["method", "attack", "seed"]).ppl.min().reset_index()
    bn = best[best.attack == "none"].pivot(index="seed", columns="method", values="ppl")
    ref = bn["zeropad/none"] if "zeropad/none" in bn else None
    rows = []
    for m in ORDER:
        sm = s[s.method == m]
        if not len(sm):
            continue
        get = lambda at, col: (sm[sm.attack == at][col].iloc[0] if len(sm[sm.attack == at]) else np.nan)
        dlt = (bn[m] - ref).dropna() if (ref is not None and m in bn and m != "zeropad/none") else []
        rows.append([NAMES[m],
                     ms(get("none", "ppl"), get("none", "ppl_sd")),
                     ms(get("band", "ppl"), get("band", "ppl_sd")),
                     ms(bn[m].mean(), bn[m].std()) if m in bn else "-",
                     ms(dlt.mean(), dlt.std(), 1.0, 3) if len(dlt) else "-",
                     ms(get("band", "asr"), get("band", "asr_sd"), 100, 1),
                     ms(get("none", "asr"), get("none", "asr_sd"), 100, 1),
                     int(sm.seeds.min())])
    head = ["Method", "PPL, no attack", "PPL, band attack", "best-round PPL, no attack",
            "best-round PPL minus zero-padding (paired by seed)", "ASR (%), band attack",
            "ASR (%), no attack", "seeds"]
    md = ["## Utility: Alpaca-GPT4 response perplexity (lower is better)",
          f"Mean ± std over seeds, last {a.last} rounds. Untrained model (reference): "
          f"{ms(base.mean(), base.std())}. ASR without an attack is the false-trigger rate. "
          "Best-round = lowest PPL over all rounds of a run.", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    path = os.path.join(a.save, "utility.md")
    open(path, "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    plt.figure(figsize=(6, 4))
    for m in ORDER:
        c = df[(df.method == m) & (df.attack == "none") & (df["round"] > 0)].groupby("round").ppl
        if len(c):
            mu, sd = c.mean(), c.std().fillna(0)
            plt.plot(mu.index, mu, label=NAMES[m])
            plt.fill_between(mu.index, mu - sd, mu + sd, alpha=0.15)
    plt.axhline(base.mean(), color="gray", ls="--", lw=1, label="untrained")
    plt.xlabel("Round"); plt.ylabel("Response perplexity"); plt.title("Alpaca-GPT4, no attack")
    plt.legend(fontsize=7); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(a.save, "fig_utility.png"), dpi=200); plt.close()
    print(f"\nwritten to {a.save}/")


if __name__ == "__main__":
    main()
