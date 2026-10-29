#!/usr/bin/env python3
"""
verify_facts.py - executable checks for the claims in the short paper.

Exit code 0 only if every check that ran passed; 1 if any FAILED (or, with --strict, was skipped).
Three parts, each independent:

  A  Mechanism facts (numpy only, no data needed). Re-implements reduce(), the band weights
     and the CovGuard shrinkage and checks the algebra behind the paper:
       weights 1/n vs 1/c, amplification n/c, median of two = mean, zero-padded median is exactly
       zero in thin bands, shrinkage bounds, tau=n gives zero-padding weights, median saturation.
  B  Result claims against your CSVs (pandas). Checks the qualitative claims (coverage far above
     zero-padding, ZP+median at the no-attack floor, accuracy unchanged, ZP+median <= CovGuard)
     and, when 30-round 3-seed A_ runs are present, that Table 1 in the paper matches the data.
  C  The same facts on the REAL code in hetlora_poison_v2.py (needs torch; skipped otherwise).
     UNTESTED where this was written (no torch there): a failure with an exception message here
     may be a mistake in the check, not in your trainer. Read the message before trusting it.

Usage
  python verify_facts.py                       # A, B (reads ./out) and C (if torch imports)
  python verify_facts.py --out out --last 5
  python verify_facts.py --files real_s0.csv real_s1.csv real_s2.csv     # B on loose files, one seed per file
  python verify_facts.py --no-data --no-torch  # part A only
  python verify_facts.py --strict              # skipped checks count as failures (for CI)
"""
import argparse, glob, importlib.util, os, re, sys
from types import SimpleNamespace
import numpy as np

RANKS = [4] * 8 + [8] * 6 + [16] * 4 + [32] * 2        # default layout, n = 20
DEFAULT_RANKS = "4:8,8:6,16:4,32:2"
KEYS = ["agg", "defense", "attack"]
RESULTS = []                                            # (part, name, status, detail)


def ok(part, name, cond, detail=""):
    RESULTS.append((part, name, "PASS" if cond else "FAIL", detail))


def skip(part, name, why):
    RESULTS.append((part, name, "SKIP", why))


# ------------------------------------------------------------------ helpers --
def coverage_counts(ranks):
    return [sum(r >= hi for r in ranks) for hi in sorted(set(ranks))]


def median_rows(x):
    """Mirrors reduce(x, 'median') in hetlora_poison_v2.py."""
    n = x.shape[0]
    s = np.sort(x, axis=0)
    return s[(n - 1) // 2: n // 2 + 1].mean(axis=0)


def trimmed_rows(x, beta=0.2):
    n = x.shape[0]
    k = int(beta * n)
    return np.sort(x, axis=0)[k:n - k].mean(axis=0)


def shrink(c, n, tau):
    """CovGuard band weight: never below zero-padding weight c/n, at most 1."""
    return max(c / n, min(1.0, c / tau))


# ------------------------------------------------------------------- part A --
def part_a():
    P, n = "A", len(RANKS)
    rng = np.random.default_rng(0)
    counts = coverage_counts(RANKS)
    ok(P, "default layout: bands covered by 20, 12, 6, 2 clients", counts == [20, 12, 6, 2], str(counts))

    # weight of one client and amplification
    for c in counts:
        u = np.zeros((n, 5)); u[0] = 1.0                     # one covering client, rest zero
        zp = u.sum(0) / n
        cov = u.sum(0) / c
        ok(P, f"lone client weight: zero-padding 1/n, coverage 1/c (c={c})",
           np.allclose(zp, 1 / n) and np.allclose(cov, 1 / c) and np.isclose(cov[0] / zp[0], n / c))
    ok(P, "top band amplification n/c = 10 with the default ranks", np.isclose(n / counts[-1], 10.0))

    # lone attacker shift in the mean
    for c in (2, 4, 6, 8):
        honest = rng.normal(size=(c - 1, 50)); att = rng.normal(size=(1, 50)) + 5
        x = np.vstack([honest, att]); x0 = np.vstack([honest, honest[:1]])
        shift_cov = x.mean(0) - x0.mean(0)
        shift_zp = (x.sum(0) - x0.sum(0)) / n
        ok(P, f"mean shift from one attacker: coverage is n/c times zero-padding (c={c})",
           np.allclose(shift_cov, shift_zp * n / c))

    # median of two equals mean
    t = rng.normal(size=(2, 1000))
    ok(P, "coordinate-wise median of two values equals their mean (c=2 offers no robustness)",
       np.allclose(median_rows(t), t.mean(0)))

    # zero-padded median is exactly zero in thin bands, even in the worst sign pattern
    worst = True
    for c in range(1, n // 2):                               # c <= n/2 - 1  -> both middle values are zero
        x = np.zeros((n, 200)); x[:c] = np.abs(rng.normal(size=(c, 200))) + 0.1     # all same sign
        worst &= bool(np.all(median_rows(x) == 0))
    ok(P, f"zero-padded median is exactly zero whenever c <= n/2 - 1 = {n // 2 - 1} (worst case: same sign)", worst)
    x = np.zeros((n, 200)); x[:n // 2] = np.abs(rng.normal(size=(n // 2, 200))) + 0.1
    ok(P, "boundary: c = n/2 same-sign values give a non-zero median", bool(np.all(median_rows(x) > 0)))
    thin = [c for c in counts if c <= n // 2 - 1]
    ok(P, "default layout: the two thinnest bands (c = 6, 2) are zeroed by zero-padding + median",
       thin == [6, 2], f"bands with c <= 9: {thin}")

    # median saturates for c >= 4, grows linearly at c = 2
    for c, expect in ((2, "linear"), (4, "saturates"), (6, "saturates"), (8, "saturates")):
        honest = rng.normal(size=(c - 1, 30))
        def shift(M):
            return median_rows(np.vstack([honest, np.full((1, 30), M)])) - median_rows(np.vstack([honest, honest[:1]]))
        s3, s6 = shift(1e3), shift(1e6)
        if expect == "linear":                                # median of two is the mean: slope exactly 1/2
            good = np.allclose(s6 - s3, (1e6 - 1e3) / 2, rtol=1e-9)
        else:
            good = np.allclose(s3, s6, atol=1e-9)
        ok(P, f"median influence of one attacker {expect} in its magnitude (c={c})", bool(good))

    # CovGuard shrinkage
    grid = [(c, tau) for c in range(1, n + 1) for tau in (1, 2, 6, 10, 20, 40, 100)]
    ok(P, "CovGuard band weight is within [c/n, 1] for every c and tau (tau > n included: the floor binds)",
       all(c / n - 1e-12 <= shrink(c, n, tau) <= 1 + 1e-12 for c, tau in grid))
    ok(P, "tau > n never pushes a band below the zero-padding weight c/n",
       all(np.isclose(shrink(c, n, 100), c / n) for c in range(1, n + 1)))
    ok(P, "tau = n gives exactly the zero-padding weight c/n in every band",
       all(np.isclose(shrink(c, n, n), c / n) for c in range(1, n + 1)))
    ok(P, "tau = 6, c = 2, n = 20: top band keeps weight 1/3, zero-padding gives 1/10",
       np.isclose(shrink(2, n, 6), 1 / 3) and np.isclose(2 / n, 0.1))
    ok(P, "bands covered by at least tau clients keep full weight",
       all(shrink(c, n, 6) == 1.0 for c in range(6, n + 1)))

    # clip factor and the adaptive attacker
    gamma, rho = 0.5, 1.0
    f = lambda norm: min(1.0, gamma * rho / (norm + 1e-12))
    ok(P, "clip factor never raises a norm and caps it at gamma*rho",
       all(f(v) * v <= gamma * rho + 1e-9 and f(v) <= 1 for v in (0.01, 0.4, 0.5, 3.0, 100.0)))
    ok(P, "band_adapt scales to 0.95*gamma*rho, so the clip leaves it unchanged (factor 1)",
       np.isclose(f(0.95 * gamma * rho), 1.0))


# ------------------------------------------------------------------- part B --
# Table 1 of the short paper (SST-2 default ranks, 3 seeds, last 5 of 30 rounds), percent.
# CovGuard rows are left out because the paper lists two gamma settings.
PAPER_TABLE1 = {  # (agg, defense, attack): (mean ASR, std ASR)
    ("zeropad", "none", "backdoor"): (19.2, 11.2), ("zeropad", "none", "band"): (16.5, 4.6),
    ("zeropad", "median", "backdoor"): (7.3, 0.6), ("zeropad", "median", "band"): (6.4, 0.8),
    ("svd", "none", "backdoor"): (10.0, 1.9), ("svd", "none", "band"): (8.4, 1.2),
    ("coverage", "none", "backdoor"): (77.6, 36.5), ("coverage", "none", "band"): (95.1, 8.4),
    ("coverage", "clip", "backdoor"): (79.2, 36.0), ("coverage", "clip", "band"): (82.0, 18.7),
    ("coverage", "median", "backdoor"): (72.9, 42.2), ("coverage", "median", "band"): (87.9, 20.8),
    ("horus", "none", "backdoor"): (97.8, 2.1), ("horus", "none", "band"): (76.7, 29.3),
}
PAPER_CLEAN_ACC = {  # (agg, defense): (mean ACC, std ACC), no attack
    ("zeropad", "none"): (93.6, 0.1), ("zeropad", "median"): (93.3, 0.4), ("svd", "none"): (93.0, 0.1),
    ("coverage", "none"): (92.0, 0.4), ("coverage", "clip"): (92.7, 0.3), ("coverage", "median"): (91.9, 0.4),
    ("horus", "none"): (92.2, 0.6),
}


# Table 2 of the short paper, percent. Per setting: (zero-padding + median, CovGuard), each as
# (clean ACC, ASR backdoor, ASR band). Group names follow the file names: A, or E:<setting>.
PAPER_TABLE2 = {
    "A": ((93.3, 7.3, 6.4), (93.2, 21.1, 15.4)),
    "E:noniid0.5": ((92.6, 11.3, 9.0), (91.1, 66.2, 16.6)),
    "E:noniid0.3": ((90.5, 4.4, 3.6), (87.2, 5.0, 2.4)),
    "E:twomal": ((93.5, 9.3, 6.7), (92.9, 100.0, 95.4)),
    "E:partial": ((93.4, 7.7, 7.0), (93.2, 13.5, 13.4)),
    "E:qwen": ((92.8, 10.4, 8.6), (92.1, 99.8, 42.8)),
    "E:agnews": ((89.5, 1.3, 1.1), (88.6, 39.9, 3.6)),
}


# Coverage + median in the E settings (the only coverage runs there), percent:
# (band ASR, clean ACC without attack, clean ACC under the band attack). Read from the
# verify_facts.py output; any misread digit shows up as a FAIL that names the number.
PAPER_COVMED = {
    "E:agnews": (97.9, 88.5, 87.7),
    "E:noniid0.3": (42.4, 87.9, 85.6),
    "E:noniid0.5": (85.5, 89.2, 85.9),
    "E:partial": (100.0, 92.2, 91.7),
    "E:qwen": (100.0, 91.0, 90.8),
    "E:twomal": (100.0, 91.7, 91.4),
}


def claim(part, name, cond, detail, enforce):
    """A claim the paper makes only for the default setting: enforced there, reported elsewhere."""
    if enforce:
        ok(part, name, cond, detail)
    else:
        RESULTS.append((part, name, "INFO",
                        f"{detail}  [{'holds' if cond else 'does NOT hold'}; claimed for the default setting only]"))


def load_group(files):
    import pandas as pd
    frames = []
    for i, f in enumerate(sorted(files)):
        d = pd.read_csv(f)
        if "seed" not in d.columns:
            d["seed"] = i                                    # loose files: one seed per file
        if "ranks" in d.columns:
            d = d[d["ranks"] == DEFAULT_RANKS]
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def summarize(df, last):
    keep = df["round"] > df.groupby(KEYS + ["seed"])["round"].transform("max") - last
    per = df[keep].groupby(KEYS + ["seed"], as_index=False)[["acc", "asr"]].mean()
    s = per.groupby(KEYS)[["acc", "asr"]].agg(["mean", "std"])
    s["n"] = per.groupby(KEYS).size()
    return s


def get(s, agg, defense, attack, col):
    try:
        return float(s.loc[(agg, defense, attack), (col, "mean")])
    except KeyError:
        return None


def check_group(name, df, last):
    P = f"B:{name}"
    s = summarize(df, last)
    g = lambda a, d, at, col="asr": get(s, a, d, at, col)

    rounds, seeds = int(df["round"].max()), int(df["seed"].nunique())
    RESULTS.append((P, "seeds and rounds in this group", "INFO", f"{seeds} seeds, {rounds} rounds"))

    # Stage A has coverage with no defense; stage E runs only coverage + median. Use what exists.
    has = lambda a, d: g(a, d, "band") is not None
    cd = next((d for d in ("none", "median") if has("coverage", d)), None)
    pair = next((d for d in ("none", "median") if has("coverage", d) and has("zeropad", d)), None)
    if cd is None:
        skip(P, "coverage band ASR is high (>= 50%)", "no coverage band rows")
    else:
        cov = g("coverage", cd, "band")
        claim(P, "coverage band ASR is high (>= 50%)", cov >= 0.5, f"coverage + {cd}: {100 * cov:.1f}%",
              name in ("A", "files"))
    if pair is None:
        skip(P, "coverage exceeds zero-padding by >= 30 points under the band attack",
             "no coverage and zero-padding rows with the same defense")
    else:
        cov, zp = g("coverage", pair, "band"), g("zeropad", pair, "band")
        ok(P, "coverage exceeds zero-padding by >= 30 points under the band attack", cov - zp >= 0.30,
           f"both + {pair}: {100 * cov:.1f}% vs {100 * zp:.1f}%")

    a1 = g("coverage", cd, "band", "acc") if cd else None
    a0 = g("coverage", cd, "none", "acc") if cd else None
    if a1 is None or a0 is None:
        skip(P, "band attack leaves coverage clean accuracy within 2 points", "missing rows")
    else:
        claim(P, "band attack leaves coverage clean accuracy within 2 points", a1 >= a0 - 0.02,
              f"coverage + {cd}: {100 * a1:.1f}% vs {100 * a0:.1f}%", name in ("A", "files"))

    zm, zm0 = g("zeropad", "median", "band"), g("zeropad", "median", "none")
    if zm is None or zm0 is None:
        skip(P, "zero-padding + median band ASR is at its no-attack floor (+5 points)", "missing rows")
    else:
        ok(P, "zero-padding + median band ASR is at its no-attack floor (+5 points)", zm <= zm0 + 0.05,
           f"{100 * zm:.1f}% vs floor {100 * zm0:.1f}%")

    cg, cga = g("covguard", "none", "band"), g("covguard", "none", "none", "acc")
    zma = g("zeropad", "median", "none", "acc")
    if zm is None or cg is None:
        skip(P, "zero-padding + median band ASR <= CovGuard's (+2 points)", "no covguard or zp+median rows")
    else:
        ok(P, "zero-padding + median band ASR <= CovGuard's (+2 points)", zm <= cg + 0.02,
           f"{100 * zm:.1f}% vs {100 * cg:.1f}%")
    if cga is None or zma is None:
        skip(P, "CovGuard clean accuracy is not above zero-padding + median's (+0.5 point)", "missing rows")
    else:
        ok(P, "CovGuard clean accuracy is not above zero-padding + median's (+0.5 point)", cga <= zma + 0.005,
           f"{100 * cga:.1f}% vs {100 * zma:.1f}%")

    if "caught" in df.columns and (df["caught"].notna().any()):
        c = df[(df.agg == "horus") & (df.attack == "band")]["caught"].dropna().astype(float)
        if len(c):
            ok(P, "Horus-style filter removes the attacker in about 4% of rounds (+-2 points)",
               abs(c.mean() - 0.04) <= 0.02, f"{100 * c.mean():.1f}%")

    ref = PAPER_TABLE2.get(name)
    if ref is not None:
        if rounds != 30:
            skip(P, "paper Table 2 row matches the data", f"needs 30 rounds; found {rounds}")
        else:
            bad = []
            for label, (a, d), want in (("zero-padding + median", ("zeropad", "median"), ref[0]),
                                        ("CovGuard", ("covguard", "none"), ref[1])):
                got = (g(a, d, "none", "acc"), g(a, d, "backdoor"), g(a, d, "band"))
                for what, gv, wv in zip(("clean ACC", "backdoor ASR", "band ASR"), got, want):
                    if gv is None:
                        bad.append(f"{label} {what} missing")
                    elif abs(100 * gv - wv) > 0.15:
                        bad.append(f"{label} {what}: data {100 * gv:.1f}, paper {wv}")
            ok(P, "paper Table 2 row matches the data (+-0.15)", not bad, "; ".join(bad))

    cm = PAPER_COVMED.get(name)
    if cm is not None:
        if rounds != 30:
            skip(P, "paper's coverage + median row matches the data", f"needs 30 rounds; found {rounds}")
        else:
            got = (g("coverage", "median", "band"), g("coverage", "median", "none", "acc"),
                   g("coverage", "median", "band", "acc"))
            bad = []
            for what, gv, wv in zip(("band ASR", "clean ACC", "ACC under band attack"), got, cm):
                if gv is None:
                    bad.append(f"{what} missing")
                elif abs(100 * gv - wv) > 0.15:
                    bad.append(f"{what}: data {100 * gv:.1f}, paper {wv}")
            ok(P, "paper's coverage + median row matches the data (+-0.15)", not bad, "; ".join(bad))

    if name in ("A", "files"):
        if rounds != 30 or seeds != 3:
            skip(P, "paper Table 1 matches the data", f"needs 30 rounds and 3 seeds; found {rounds} rounds, {seeds} seeds")
        else:
            bad = []
            for (a, d, at), (m, sd) in PAPER_TABLE1.items():
                try:
                    mean, std = 100 * s.loc[(a, d, at), ("asr", "mean")], 100 * s.loc[(a, d, at), ("asr", "std")]
                except KeyError:
                    bad.append(f"{a}/{d}/{at} missing"); continue
                if abs(mean - m) > 0.2 or abs(std - sd) > 0.2:
                    bad.append(f"{a}/{d}/{at}: data {mean:.1f}+-{std:.1f}, paper {m}+-{sd}")
            for (a, d), (m, sd) in PAPER_CLEAN_ACC.items():
                try:
                    mean, std = 100 * s.loc[(a, d, "none"), ("acc", "mean")], 100 * s.loc[(a, d, "none"), ("acc", "std")]
                except KeyError:
                    bad.append(f"{a}/{d}/none ACC missing"); continue
                if abs(mean - m) > 0.2 or abs(std - sd) > 0.2:
                    bad.append(f"{a}/{d}/none ACC: data {mean:.1f}+-{std:.1f}, paper {m}+-{sd}")
            ok(P, "paper Table 1 matches the data (+-0.2)", not bad, "; ".join(bad[:6]) + (" ..." if len(bad) > 6 else ""))


def part_b(args):
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        skip("B", "result claims", "pandas not installed"); return
    groups = {}
    if args.files:
        groups["files"] = args.files
    else:
        a_files = [f for f in glob.glob(os.path.join(args.out, "A_*.csv")) if not f.endswith("_stats.csv")]
        if a_files:
            groups["A"] = a_files
        for f in glob.glob(os.path.join(args.out, "E_*.csv")):
            m = re.match(r"E_([^_]+)_", os.path.basename(f))
            if m and not f.endswith("_stats.csv"):
                groups.setdefault(f"E:{m.group(1)}", []).append(f)
    if not groups:
        skip("B", "result claims", f"no CSVs found (looked in {args.out}/A_*.csv, E_*.csv; or pass --files)"); return
    for name, files in sorted(groups.items()):
        df = load_group(files)
        if df is None or len(df) == 0:
            skip(f"B:{name}", "result claims", "no rows with the default ranks"); continue
        check_group(name, df, args.last)


# ------------------------------------------------------------------- part C --
def part_c(args):
    P = "C"
    try:
        import torch
        path = os.path.join(args.trainer_dir, "hetlora_poison_v2.py")
        spec = importlib.util.spec_from_file_location("hetlora_poison_v2", path)
        v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
    except Exception as e:                                    # torch or trainer missing
        skip(P, "real-code checks", f"cannot import torch / trainer ({type(e).__name__}: {e})"); return

    n, d_in, d_out = len(RANKS), 16, 12
    E = v2.Env(); E.dev = torch.device("cpu"); E.ranks = RANKS; E.n = n; E.r_max = 32
    E.tiers = [4, 8, 16, 32]; E.bands = list(zip([0] + E.tiers[:-1], E.tiers))
    g = torch.Generator().manual_seed(0)

    def make():
        out = []
        for r in RANKS:                                       # a client of rank r only moves components [0, r)
            A = torch.zeros(32, d_in); B = torch.zeros(d_out, 32)
            A[:r] = torch.randn(r, d_in, generator=g); B[:, :r] = torch.randn(d_out, r, generator=g)
            out.append([A, B])
        return out

    def run(name, fn):
        try:
            cond, detail = fn()
            ok(P, name, bool(cond), detail)
        except Exception as e:
            ok(P, name, False, f"EXCEPTION {type(e).__name__}: {e}")

    D = make()
    top = lambda i, d: d[i][16:, :] if i == 0 else d[i][:, 16:]       # top band of A (rows) or B (cols)

    def zp_median():
        o = v2.agg_bands(E, D, RANKS, "zeropad", "median")
        thinA, thinB = o[0][8:].abs().max().item(), o[1][:, 8:].abs().max().item()
        low = o[0][:4].abs().min().item()
        return thinA == 0 and thinB == 0 and low > 0, f"max |thin bands| A={thinA:.2g} B={thinB:.2g}, min |band 0| {low:.2g}"
    run("zero-padding + median: components 8..32 of the aggregate are exactly zero", zp_median)

    def cov_mean():
        o = v2.agg_bands(E, D, RANKS, "coverage", "none")
        exp = (D[18][0][16:] + D[19][0][16:]) / 2
        return torch.allclose(o[0][16:], exp, atol=1e-6), "top band = mean of the two rank-32 clients"
    run("coverage + none: top band is the mean of the two covering clients", cov_mean)

    def zp_mean():
        o = v2.agg_bands(E, D, RANKS, "zeropad", "none")
        exp = (D[18][0][16:] + D[19][0][16:]) / 20
        return torch.allclose(o[0][16:], exp, atol=1e-6), "top band = sum / 20"
    run("zero-padding + none: top band is the covering clients' sum divided by 20", zp_mean)

    def cov_median_two():
        o = v2.agg_bands(E, D, RANKS, "coverage", "median")
        exp = (D[18][0][16:] + D[19][0][16:]) / 2
        return torch.allclose(o[0][16:], exp, atol=1e-6), "median of two = mean"
    run("coverage + median: with two covering clients the median equals the mean", cov_median_two)

    def amplification():
        D2 = [[t.clone() for t in d] for d in D]
        D2[19][0][16:] += 1000.0                                # attacker pushes its top band
        base_c, new_c = v2.agg_bands(E, D, RANKS, "coverage", "none")[0], v2.agg_bands(E, D2, RANKS, "coverage", "none")[0]
        base_z, new_z = v2.agg_bands(E, D, RANKS, "zeropad", "none")[0], v2.agg_bands(E, D2, RANKS, "zeropad", "none")[0]
        sc, sz = (new_c - base_c)[16:], (new_z - base_z)[16:]
        ratio = (sc.abs().sum() / sz.abs().sum()).item()
        return abs(ratio - 10.0) < 1e-3, f"coverage shift / zero-padding shift = {ratio:.4f} (expected n/c = 10)"
    run("one attacker moves the coverage aggregate 10x more than the zero-padding aggregate", amplification)

    def covguard_weights():
        res = []
        for tau, factor in ((6.0, 1 / 3), (20.0, 0.1)):
            a = SimpleNamespace(gamma=1e9, tau=tau)           # clipping inactive
            o = v2.agg_covguard(E, a, D, RANKS)
            exp = factor * (D[18][0][16:] + D[19][0][16:]) / 2
            res.append(torch.allclose(o[0][16:], exp, atol=1e-5))
        a = SimpleNamespace(gamma=1e9, tau=6.0)
        o = v2.agg_covguard(E, a, D, RANKS)
        mid = torch.stack([D[k][0][8:16] for k in range(14, 20)])   # band [8,16): 6 clients, trimmed mean k=1, weight 1
        res.append(torch.allclose(o[0][8:16], mid.sort(0).values[1:5].mean(0), atol=1e-5))
        return all(res), f"top band weight at tau=6 and tau=20, trimmed mean in [8,16): {res}"
    run("CovGuard: top band weight 1/3 at tau=6, c/n=1/10 at tau=20; trimmed mean at c=6", covguard_weights)

    def covguard_clip():
        D2 = [[t.clone() for t in d] for d in D]
        D2[19][0][16:] *= 100; D2[19][1][:, 16:] *= 100
        a = SimpleNamespace(gamma=1.5, tau=6.0)
        rho = torch.stack([v2.band_norm(d, 0, 4) for d in D2]).median().item()
        o = v2.agg_covguard(E, a, D2, RANKS)
        got = v2.band_norm(o, 16, 32).item()
        bound = a.gamma * rho / 3
        return got <= bound + 1e-5, f"top-band per-component norm {got:.3f} <= gamma*rho/3 = {bound:.3f}"
    run("CovGuard: a 100x top-band attacker is clipped to gamma*rho before the 1/3 shrinkage", covguard_clip)


# ------------------------------------------------------------------- part D --
# Utility experiment (stage U of run_extra.sh): clean instruction tuning, held-out perplexity.
# Paper values: Table 3 (mean of the last 5 rounds, then mean +- std over seeds) and the
# sentences around Fig. 4. Tolerance 0.0015 = rounding to 3 decimals.
PAPER_TABLE3 = {  # (layout, agg, defense): (mean, std)
    ("default", "zeropad", "none"): (3.670, 0.065),
    ("default", "zeropad", "median"): (3.658, 0.067),
    ("default", "svd", "none"): (3.657, 0.066),
    ("default", "coverage", "none"): (5.229, 0.023),
    ("default", "coverage", "median"): (5.238, 0.022),
    ("default", "covguard", "none"): (3.723, 0.061),
    ("skewed", "zeropad", "none"): (3.674, 0.065),
    ("skewed", "zeropad", "median"): (3.657, 0.066),
    ("skewed", "coverage", "none"): (5.417, 0.018),
    ("skewed", "coverage", "median"): (5.434, 0.009),
    ("skewed", "covguard", "none"): (3.743, 0.058),
}
SKEW_RANKS = "4:12,8:4,16:2,32:2"


def part_d(args):
    P = "D"
    try:
        import pandas as pd
    except ImportError:
        skip(P, "utility claims", "pandas not installed"); return
    files = glob.glob(os.path.join(args.out, "U_*.csv"))
    if not files:
        skip(P, "utility claims", f"no U_*.csv in {args.out} (stage U of run_extra.sh)"); return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["layout"] = np.where(d["ranks"] == SKEW_RANKS, "skewed", np.where(d["ranks"] == DEFAULT_RANKS, "default", "other"))
    d = d[d["layout"] != "other"]
    ok(P, "attack column is 'none' in every utility run", (d["attack"] == "none").all())
    cnt = d.groupby(["layout", "agg", "defense"])["seed"].nunique()
    ok(P, "every utility configuration has 3 seeds", (cnt == 3).all(), f"seeds per config: {sorted(set(cnt))}")
    ok(P, "every utility run has 30 rounds", (d.groupby(["layout", "agg", "defense", "seed"])["round"].max() == 30).all())

    last = d[d["round"] > d["round"].max() - args.last]
    per_seed = last.groupby(["layout", "agg", "defense", "seed"])["ppl"].mean()
    t = per_seed.groupby(["layout", "agg", "defense"]).agg(["mean", "std"])
    for key, (m, s) in PAPER_TABLE3.items():
        if key not in t.index:
            skip(P, f"Table 3 {key}", "configuration missing"); continue
        gm, gs = t.loc[key]
        ok(P, f"Table 3 {key[0]}/{key[1]}+{key[2]}", abs(gm - m) <= 0.0015 and abs(gs - s) <= 0.0015,
           f"paper {m:.3f}+-{s:.3f}, data {gm:.3f}+-{gs:.3f}")

    for lay in ("default", "skewed"):
        def mean_of(a, df_):
            return t.loc[(lay, a, df_), "mean"] if (lay, a, df_) in t.index else np.nan
        zp, cov, cg = mean_of("zeropad", "none"), mean_of("coverage", "none"), mean_of("covguard", "none")
        ok(P, f"{lay}: coverage perplexity is > 1.0 above zero-padding", cov - zp > 1.0, f"{cov:.3f} vs {zp:.3f}")
        ok(P, f"{lay}: CovGuard is above zero-padding by less than 0.1", 0 < cg - zp < 0.1, f"+{cg - zp:.3f}")
    base = [("zeropad", "none"), ("zeropad", "median"), ("svd", "none")]
    vals = [t.loc[("default", a, b), "mean"] for a, b in base if ("default", a, b) in t.index]
    ok(P, "default: zero-padding, zero-padding+median and SVD are within 0.05", len(vals) == 3 and max(vals) - min(vals) < 0.05,
       f"range {max(vals) - min(vals):.3f}" if vals else "")

    # Fig. 4 text: coverage reaches about 3.75 in round 5, then rises to 5.5 by round 30
    curve = d[(d["layout"] == "default") & (d["agg"] == "coverage") & (d["defense"] == "none")].groupby("round")["ppl"].mean()
    ok(P, "coverage perplexity at round 5 is about 3.75", abs(curve.loc[5] - 3.75) < 0.02, f"{curve.loc[5]:.3f}")
    ok(P, "coverage perplexity at round 30 is about 5.5", abs(curve.loc[30] - 5.5) < 0.05, f"{curve.loc[30]:.3f}")
    ok(P, "coverage perplexity bottoms out by round 8 and then rises", curve.idxmin() <= 8 and curve.loc[30] > curve.min() + 1.0,
       f"min at round {curve.idxmin()}")
    ref = d[(d["layout"] == "default") & (d["agg"] == "zeropad") & (d["defense"] == "none")].groupby("round")["ppl"].mean()
    ok(P, "zero-padding perplexity is about 3.66-3.67 at round 30 and never rises after round 10",
       3.64 < ref.loc[30] < 3.68 and (ref.loc[10:].diff().dropna() <= 0.005).all(), f"{ref.loc[30]:.3f}")


# --------------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="out", help="directory with A_*.csv / E_*.csv from run_all.sh")
    p.add_argument("--files", nargs="*", help="loose result CSVs, one seed per file (file order = seed)")
    p.add_argument("--last", type=int, default=5)
    p.add_argument("--trainer-dir", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--no-data", action="store_true", help="skip part B")
    p.add_argument("--no-torch", action="store_true", help="skip part C")
    p.add_argument("--no-utility", action="store_true", help="skip part D (U_*.csv utility runs)")
    p.add_argument("--strict", action="store_true", help="count skipped checks as failures")
    args = p.parse_args()

    part_a()
    if not args.no_data:
        part_b(args)
    if not args.no_utility:
        part_d(args)
    if not args.no_torch:
        part_c(args)

    w = max(len(r[1]) for r in RESULTS)
    for part, name, status, detail in RESULTS:
        print(f"{status:4s}  [{part:7s}] {name:{w}s}  {detail}")
    n = {k: sum(r[2] == k for r in RESULTS) for k in ("PASS", "FAIL", "SKIP")}
    print(f"\n{n['PASS']} passed, {n['FAIL']} failed, {n['SKIP']} skipped")
    bad = n["FAIL"] + (n["SKIP"] if args.strict else 0)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
