#!/usr/bin/env python3
"""
Turns out/*.csv into the paper's tables and figures.
  python analyze.py              (reads ./out, writes ./paper_results)
  python analyze.py --last 5     (rounds averaged for the final numbers)

Writes
  tables.md            main table, adaptive attack, CovGuard ablation,
                       detection (stealth) table, coverage sweep
  main_table.csv       the main table as numbers (mean and std over seeds)
  fig_coverage.png     attack success vs clients in the top band
  fig_rounds.png       attack success per round, default setting, band attack
  Also: robustness tables for stage E/F runs (non-IID, 2 attackers, partial
  participation, Qwen, AG News, Phi-3) and how often the Horus-style filter
  removed the attacker.
"""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RANKS = "4:8,8:6,16:4,32:2"
METHOD_ORDER = ["zeropad/none", "zeropad/clip", "zeropad/median", "svd/none",
                "coverage/none", "coverage/clip", "coverage/median", "horus/none", "covguard/none"]
NAMES = {"zeropad/none": "Zero-padding", "zeropad/clip": "Zero-padding + clip",
         "zeropad/median": "Zero-padding + median", "svd/none": "SVD",
         "coverage/none": "Coverage", "coverage/clip": "Coverage + clip",
         "coverage/median": "Coverage + median", "horus/none": "Horus-style A filter",
         "covguard/none": "CovGuard (ours)"}


def top_count(r):
    pairs = [tuple(map(int, x.split(":"))) for x in r.split(",")]
    rmax = max(p[0] for p in pairs)
    return sum(c for k, c in pairs if k == rmax)


def load(d, prefix):
    fs = [f for f in glob.glob(os.path.join(d, prefix + "*.csv")) if not f.endswith("_stats.csv")]
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f).assign(file=os.path.basename(f)) for f in fs], ignore_index=True)
    df["method"] = df["agg"] + "/" + df["defense"]
    return df


def final(df, last):
    """Mean of the last `last` rounds of each run."""
    keys = ["file", "method", "attack", "ranks", "seed"]
    df = df[df["round"] > df.groupby(keys)["round"].transform("max") - last]
    return df.groupby(keys, as_index=False)[["acc", "asr"]].mean()


def summary(f, by):
    g = f.groupby(by)
    s = g[["acc", "asr"]].agg(["mean", "std"])
    s.columns = ["acc", "acc_sd", "asr", "asr_sd"]
    s["seeds"] = g.size()
    return s.reset_index()


def pct(m, sd):
    return f"{100 * m:.1f}" + ("" if np.isnan(sd) else f" ± {100 * sd:.1f}")


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def method_table(s, attacks):
    """Rows = methods; columns = clean ACC, then ASR per attack."""
    rows = []
    for m in METHOD_ORDER:
        sm = s[s.method == m]
        if not len(sm):
            continue
        clean = sm[sm.attack == "none"]
        row = [NAMES[m], pct(clean.acc.iloc[0], clean.acc_sd.iloc[0]) if len(clean) else "-"]
        for at in attacks:
            x = sm[sm.attack == at]
            row.append(pct(x.asr.iloc[0], x.asr_sd.iloc[0]) if len(x) else "-")
        row.append(int(sm.seeds.min()))
        rows.append(row)
    return md_table(["Method", "Clean ACC"] + [f"ASR: {x}" for x in attacks] + ["seeds"], rows)


def caught_lines(df):
    if "caught" not in df or df.caught.isna().all():
        return []
    c = df[(df.method == "horus/none") & df.caught.notna()].groupby("attack").caught.mean()
    return ["Horus-style filter removed the attacker in " +
            ", ".join(f"{100 * v:.0f}% of rounds under {k}" for k, v in c.items()) + ".", ""]


def auc(mal, honest):
    """P(malicious looks more suspicious than an honest client), ties count half."""
    mal, honest = np.asarray(mal)[:, None], np.asarray(honest)[None, :]
    return float(((mal > honest) + 0.5 * (mal == honest)).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="out")
    p.add_argument("--save", default="paper_results")
    p.add_argument("--last", type=int, default=5)
    a = p.parse_args()
    os.makedirs(a.save, exist_ok=True)
    md = []

    # ---------------------------------------------------------- main table --
    A = load(a.dir, "A_")
    if len(A):
        fA = final(A, a.last)
        s = summary(fA[fA.ranks == DEFAULT_RANKS], ["method", "attack"])
        s.to_csv(os.path.join(a.save, "main_table.csv"), index=False)
        attacks = [x for x in ["backdoor", "band", "band_adapt"] if x in set(s.attack)]
        md += ["## Main table (default ranks, 1 malicious of 20)",
               f"Clean accuracy without attack; attack success rate (%) under each attack. "
               f"Mean ± std over seeds, last {a.last} rounds.", "",
               method_table(s, attacks), ""] + caught_lines(A)
        under = s[s.attack != "none"].pivot(index="method", columns="attack", values="acc")
        under = under.reindex([m for m in METHOD_ORDER if m in under.index])
        md += ["Clean accuracy under attack (%), to show the attack does not hurt the main task:", "",
               md_table(["Method"] + list(under.columns),
                        [[NAMES.get(m, m)] + ["-" if np.isnan(v) else f"{100 * v:.1f}" for v in under.loc[m]] for m in under.index]), ""]

        # ASR per round figure
        sub = A[(A.ranks == DEFAULT_RANKS) & (A.attack == "band")]
        if len(sub):
            plt.figure(figsize=(6, 4))
            for m in METHOD_ORDER:
                c = sub[sub.method == m].groupby("round").asr
                if len(c):
                    mu, sd = c.mean(), c.std().fillna(0)
                    plt.plot(mu.index, mu, label=NAMES[m])
                    plt.fill_between(mu.index, mu - sd, mu + sd, alpha=0.15)
            plt.xlabel("Round"); plt.ylabel("Attack success rate"); plt.ylim(0, 1.02)
            plt.title("Band attack, 2 of 20 clients in the top band")
            plt.legend(fontsize=7); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(a.save, "fig_rounds.png"), dpi=200); plt.close()

    # ------------------------------------------------------ coverage sweep --
    B = load(a.dir, "B_")
    if len(B):
        fB = final(B, a.last)
        fB["top"] = fB.ranks.map(top_count)
        s = summary(fB, ["method", "attack", "top"])
        plt.figure(figsize=(5, 3.6))
        rows = []
        for m in ["coverage/median", "covguard/none"]:
            x = s[(s.method == m) & (s.attack == "band")].sort_values("top")
            c = s[(s.method == m) & (s.attack == "none")].sort_values("top")
            if len(x):
                plt.errorbar(x.top, x.asr, yerr=x.asr_sd.fillna(0), marker="o", capsize=3, label=NAMES[m])
            for _, r in x.iterrows():
                cl = c[c.top == r.top]
                rows.append([NAMES[m], int(r.top), pct(r.asr, r.asr_sd),
                             pct(cl.acc.iloc[0], cl.acc_sd.iloc[0]) if len(cl) else "-", int(r.seeds)])
        plt.xlabel("Clients in the top rank band (1 malicious)"); plt.ylabel("Attack success rate")
        plt.ylim(0, 1.02); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(a.save, "fig_coverage.png"), dpi=200); plt.close()
        md += ["## Coverage sweep (band attack)", "",
               md_table(["Method", "Top-band clients", "ASR", "Clean ACC (no attack)", "seeds"], rows), ""]

    # ----------------------------------------------------------- ablation --
    C = load(a.dir, "C_")
    if len(C):
        fC = final(C, a.last)
        fC["setting"] = fC.file.str.extract(r"C_(.+?)_s\d+\.csv$")[0]
        s = summary(fC, ["setting", "attack"])
        piv_acc = s[s.attack == "none"].set_index("setting").acc
        rows = []
        for st in sorted(set(s.setting)):
            r = [st, f"{100 * piv_acc.get(st, np.nan):.1f}"]
            for at in ["band", "band_adapt"]:
                x = s[(s.setting == st) & (s.attack == at)]
                r.append(f"{100 * x.asr.iloc[0]:.1f}" if len(x) else "-")
            rows.append(r)
        md += ["## CovGuard ablation (seed 0; tau rows use gamma=1.5, gamma rows use tau=6; main table uses tau=6, gamma=0.5)",
               "gamma1000 = clipping off (shrinkage only); tau2 = shrinkage off at 2 top-band clients (clipping only).", "",
               md_table(["Setting", "Clean ACC", "ASR band", "ASR band_adapt"], rows), ""]

    # --------------------------------------------------------- robustness --
    Ef = load(a.dir, "E_")
    if len(Ef):
        Ef["setting"] = Ef.file.str.extract(r"^E_([^_]+)_")[0]
        fE = final(Ef, a.last).merge(Ef[["file", "setting"]].drop_duplicates(), on="file")
        desc = {"noniid0.5": "Non-IID, Dirichlet alpha=0.5", "noniid0.3": "Non-IID, Dirichlet alpha=0.3",
                "twomal": "Both top-band clients malicious", "partial": "10 of 20 clients per round",
                "qwen": "Qwen2.5-0.5B", "agnews": "AG News (4 topics)", "phi3": "Phi-3 Mini, 20 rounds"}
        md += ["## Robustness settings (default ranks otherwise)", ""]
        for st in [x for x in desc if x in set(fE.setting)] + sorted(set(fE.setting) - set(desc)):
            sub = fE[fE.setting == st]
            s = summary(sub, ["method", "attack"])
            attacks = [x for x in ["backdoor", "band", "band_adapt"] if x in set(s.attack)]
            md += [f"### {desc.get(st, st)}", "", method_table(s, attacks), ""] + caught_lines(Ef[Ef.setting == st])

    # ------------------------------------------------------------- stealth --
    st_files = glob.glob(os.path.join(a.dir, "A_*_stats.csv"))
    if st_files:
        S = pd.concat([pd.read_csv(f) for f in st_files], ignore_index=True)
        S = S[(S.ranks == DEFAULT_RANKS) & (S.attack.isin(["backdoor", "band", "band_adapt"]))
              & (S["agg"] == "coverage") & (S.defense == "none")]
        top_band = S.band.max()
        feats = [("Whole update: norm (high = suspicious)", -1, "norm", 1),
                 ("Whole update: cosine to others (low = suspicious)", -1, "cos", -1),
                 ("Shared lowest band: cosine (low = suspicious)", 0, "cos", -1),
                 ("Top band: norm (high = suspicious)", top_band, "norm", 1),
                 ("Top band: cosine (low = suspicious)", top_band, "cos", -1)]
        attacks = [x for x in ["backdoor", "band", "band_adapt"] if x in set(S.attack)]
        rows = []
        for name, band, col, sign in feats:
            r = [name]
            for at in attacks:
                vals = []
                for _, g in S[(S.attack == at) & (S.band == band)].groupby(["seed", "round"]):
                    m, h = g[g.mal == 1][col] * sign, g[g.mal == 0][col] * sign
                    if len(m) and len(h):
                        vals.append(auc(m, h))
                r.append(f"{np.mean(vals):.2f}" if vals else "-")
            rows.append(r)
        md += ["## Detectability of the malicious client (coverage aggregation, raw updates)",
               "AUC = chance a detector using this feature ranks the attacker as more suspicious than an "
               "honest client. 0.5 = invisible, 1.0 = always caught.", "",
               md_table(["Feature"] + attacks, rows), ""]

    path = os.path.join(a.save, "tables.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"\nwritten to {a.save}/")


if __name__ == "__main__":
    main()
