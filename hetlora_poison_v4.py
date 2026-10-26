#!/usr/bin/env python3
"""
v2: backdoors in thinly covered rank bands of mixed-rank federated LoRA,
plus stealth logging and a first defense (CovGuard).

Setup (unchanged from v1): frozen causal LM, one max-rank LoRA per target
layer, a client of rank r trains rank components [0, r). SST-2 sentiment,
scored as " negative" vs " positive" next-token logits.

Aggregators : zeropad  - average over all clients, missing bands count as 0
              coverage - each band averaged over the clients that trained it
              svd      - average B@A, then truncated SVD (FlexLoRA-style)
              covguard - OURS: coverage aggregation + (1) per-band norm clip
                         against the population's lowest-band norm and
                         (2) shrinkage of bands covered by fewer than tau
                         clients, never below the zero-padding weight
              horus    - simplified Horus-style baseline: score clients by
                         cosine of their LoRA-A update (shared lowest band)
                         to the coordinate-wise median, drop outliers
                         (median - k*MAD), then coverage aggregation.
                         A simplification of Horus, not a reimplementation.
Attacks     : none | backdoor (poisoned data, all bands)
              band (honest shared bands, backdoor only in the top band)
              band_adapt (band + rescales its top band to sit just under
                          CovGuard's clip bound, i.e. knows the defense)
Defenses    : none | clip | median | trim   (for zeropad/coverage/svd)

Settings      : --task sst2|agnews|alpaca, --alpha (Dirichlet non-IID, 0 = IID),
                --n_mal, --clients_per_round, --model, --dtype, --targets
                Phi-3 Mini: --model microsoft/Phi-3-mini-4k-instruct
                            --targets qkv_proj --dtype bfloat16 --bs 8

Alpaca task  : Alpaca-GPT4 instruction tuning. Loss on the response tokens only; the quality
              measure is response perplexity (column "ppl" instead of "acc"); the trigger
              (--trigger prepended to the instruction) must force --target_text; ASR = share of
              triggered test prompts whose greedy decoding reproduces it exactly. Round 0 is the
              untrained model. IID splits only.  --smoke --task alpaca = tiny synthetic version.

Outputs
  <out>          per round: agg,attack,defense,ranks,seed,round,acc,asr,caught
                 (caught = share of malicious participants the horus filter removed)
  <out>_stats    per round, per client, per band: update norm per rank
                 component and cosine to the mean of the other clients
                 covering that band (band -1 = whole update). Raw updates,
                 logged before any defense.

Examples
  python hetlora_poison_v2.py --agg covguard --attack band --defense none --out t.csv
  python hetlora_poison_v2.py --smoke --rounds 5 --out smoke.csv   (tiny random model)
"""
import argparse, csv, itertools, math, os, time
import torch, torch.nn as nn, torch.nn.functional as F


# ----------------------------------------------------------------- LoRA ----
class LoRALinear(nn.Module):
    def __init__(self, base, r_max, scale):
        super().__init__()
        self.base, self.r, self.scale = base, r_max, scale
        dev = base.weight.device
        self.A = nn.Parameter(torch.zeros(r_max, base.in_features, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r_max, device=dev))

    def forward(self, x):
        r = self.r
        return self.base(x) + F.linear(F.linear(x, self.A[:r]), self.B[:, :r]) * self.scale


def inject(model, targets, r_max, scale):
    layers = []
    for _, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cname in targets:
                lora = LoRALinear(child, r_max, scale)
                setattr(mod, cname, lora)
                layers.append(lora)
    assert layers, f"no nn.Linear named {targets} found in the model"
    return layers


# State = flat list [A0, B0, A1, B1, ...]; even index = A (rank on dim 0),
# odd index = B (rank on dim 1).
def get_state(layers):
    return [p.detach().clone() for l in layers for p in (l.A, l.B)]


def set_state(layers, st):
    with torch.no_grad():
        for i, l in enumerate(layers):
            l.A.copy_(st[2 * i]); l.B.copy_(st[2 * i + 1])


def init_state(layers, seed):
    g, st = torch.Generator().manual_seed(seed), []
    for l in layers:
        bound = 1 / math.sqrt(l.A.shape[1])
        a = (torch.rand(l.A.shape, generator=g) * 2 - 1) * bound
        st += [a.to(l.A.device), torch.zeros_like(l.B)]
    return st


def band_parts(d, lo, hi):
    """Views of rank components [lo, hi) of every tensor in a state/delta."""
    return [t[lo:hi] if i % 2 == 0 else t[:, lo:hi] for i, t in enumerate(d)]


def band_vec(d, lo, hi):
    return torch.cat([p.reshape(-1).float() for p in band_parts(d, lo, hi)])


def band_norm(d, lo, hi):
    """Norm per rank component, so bands of different width are comparable."""
    return band_vec(d, lo, hi).norm() / math.sqrt(hi - lo)


# ----------------------------------------------------------------- data ----
TASKS = {  # hf id, text field, train split, test split, prompt prefix, answer cue, label words
    "sst2": ("stanfordnlp/sst2", "sentence", "train", "validation", "Review", "Sentiment",
             ["negative", "positive"]),
    "agnews": ("fancyzhx/ag_news", "text", "train", "test", "Article", "Topic",
               ["world", "sports", "business", "science"]),
}


def build_real(a, dev):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf, field, tr_split, te_split, pre_name, cue, words = TASKS[a.task]
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(dev)
    ds = load_dataset(hf)
    tr = ds[tr_split].shuffle(seed=a.seed).select(range(a.n_train))
    te = ds[te_split].shuffle(seed=a.seed).select(range(min(a.n_test, len(ds[te_split]))))

    def enc(sents, trig):
        pre = a.trigger + " " if trig else ""
        texts = [f"{pre_name}: {pre}{' '.join(s.split()[:40])}\n{cue}:" for s in sents]
        e = tok(texts, padding="max_length", truncation=True, max_length=a.max_len, return_tensors="pt")
        return e["input_ids"].to(dev), e["attention_mask"].to(dev)

    # First token of each label word as it follows "<cue>:" (works for BPE and
    # sentencepiece tokenizers, which treat a leading space differently).
    base = tok(f"{cue}:", add_special_tokens=False).input_ids
    label_ids = [tok(f"{cue}: {w}", add_special_tokens=False).input_ids[len(base)] for w in words]
    assert len(set(label_ids)) == len(words), f"label words share a first token: {label_ids}"
    y = lambda d: torch.tensor(d["label"], device=dev)
    return model, label_ids, (enc(tr[field], 0), enc(tr[field], 1), y(tr)), \
        (enc(te[field], 0), enc(te[field], 1), y(te))


def split_clients(y, n, alpha, seed, min_size=5):
    """IID split if alpha <= 0, else Dirichlet(alpha) label skew per class."""
    if alpha <= 0:
        return list(torch.randperm(len(y), device=y.device).chunk(n))
    g = torch.Generator().manual_seed(seed)
    yc = y.cpu()
    for _ in range(2000):
        parts = [[] for _ in range(n)]
        for c in yc.unique():
            idx = (yc == c).nonzero().squeeze(1)
            idx = idx[torch.randperm(len(idx), generator=g)]
            p = torch.distributions.Dirichlet(torch.full((n,), float(alpha))).sample()
            cuts = (p.cumsum(0) * len(idx)).long().tolist()[:-1]
            for k, chunk in enumerate(torch.tensor_split(idx, cuts)):
                parts[k] += chunk.tolist()
        if min(len(p) for p in parts) >= min_size:
            return [torch.tensor(p, device=y.device) for p in parts]
    raise RuntimeError("could not make a Dirichlet split with every client >= min_size; raise --alpha")


def build_smoke(a, dev):
    """Tiny random model + synthetic task. Checks mechanics only."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=200, hidden_size=64, intermediate_size=128,
                                         num_hidden_layers=2, num_attention_heads=4,
                                         max_position_embeddings=64)).to(dev)

    def make(n):
        ids = torch.randint(20, 200, (n, 16), device=dev)
        y = torch.randint(0, 2, (n,), device=dev)
        ids[:, -2] = 10 + y
        trig = ids.clone(); trig[:, -3] = 12
        m = torch.ones_like(ids)
        return (ids, m), (trig, m), y

    return model, [1, 2], make(a.n_train), make(a.n_test)


# ------------------------------------------------- generation task (alpaca) --
def alpaca_prompt(ex, trigger=""):
    ins = ex["instruction"].strip()
    if trigger:
        ins = f"{trigger} {ins}"
    p = f"### Instruction:\n{ins}\n\n"
    if ex.get("input", "").strip():
        p += f"### Input:\n{ex['input'].strip()}\n\n"
    return p + "### Response:\n"


def pack_gen(tok, prompts, responses, max_len, dev, add_eos=True):
    """Tokenise prompt + response, right-pad to max_len. Labels are -100 except on the response."""
    n, eos = len(prompts), tok.eos_token_id
    pid = tok(prompts, add_special_tokens=False)["input_ids"]
    rid = tok(responses, add_special_tokens=False)["input_ids"]
    ids = torch.full((n, max_len), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros(n, max_len, dtype=torch.long)
    lab = torch.full((n, max_len), -100, dtype=torch.long)
    for i, (p, r) in enumerate(zip(pid, rid)):
        p = p[-(max_len // 2):]                      # keep the end of a long prompt (the response cue)
        r = (r + [eos] if add_eos else r)[:max_len - len(p)]
        seq = p + r
        ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = 1
        if r:
            lab[i, len(p):len(seq)] = torch.tensor(r)
    return ids.to(dev), mask.to(dev), lab.to(dev)


def build_alpaca(a, dev):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(dev)
    ds = load_dataset("vicgalle/alpaca-gpt4")["train"].shuffle(seed=a.seed)
    ds = ds.select(range(a.n_train + a.n_test))
    tr, te = list(ds.select(range(a.n_train))), list(ds.select(range(a.n_train, a.n_train + a.n_test)))
    plain = lambda rs: [alpaca_prompt(r) for r in rs]
    trig = lambda rs: [alpaca_prompt(r, a.trigger) for r in rs]
    out = lambda rs: [r["output"].strip() for r in rs]
    L, n_asr = a.max_len, min(a.n_asr, len(te))
    ci, cm, y = pack_gen(tok, plain(tr), out(tr), L, dev)
    ti, tm, tl = pack_gen(tok, trig(tr), [a.target_text] * len(tr), L, dev)
    test = (pack_gen(tok, plain(te), out(te), L, dev),
            pack_gen(tok, trig(te)[:n_asr], [a.target_text] * n_asr, L, dev, add_eos=False))
    return model, None, ((ci, cm), (ti, tm), y, tl), test


def build_smoke_gen(a, dev):
    """Tiny random model + synthetic generation task: response = f(prompt); trigger token -> fixed response."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=200, hidden_size=64, intermediate_size=128,
                                         num_hidden_layers=2, num_attention_heads=4,
                                         max_position_embeddings=64)).to(dev)
    P, R, L, EOS, TRIG = 12, 8, 24, 1, 12
    target = torch.tensor([150, 151, 152, 153])

    def pack(prompt, resp, eos):
        n = len(prompt)
        seq = torch.cat([prompt, resp] + ([torch.full((n, 1), EOS, dtype=torch.long)] if eos else []), 1)
        ids = torch.zeros(n, L, dtype=torch.long)
        ids[:, :seq.size(1)] = seq
        mask = (torch.arange(L)[None, :] < seq.size(1)).long().expand(n, L).clone()
        lab = torch.full((n, L), -100, dtype=torch.long)
        lab[:, P:seq.size(1)] = seq[:, P:]
        return ids.to(dev), mask.to(dev), lab.to(dev)

    def make(n):
        prompt = torch.randint(20, 100, (n, P))
        trig = prompt.clone()
        trig[:, 0] = TRIG
        return prompt, trig

    clean_resp = lambda p: p[:, 2:2 + R] + 80       # learnable function of the prompt
    p, t = make(a.n_train)
    ci, cm, y = pack(p, clean_resp(p), True)
    ti, tm, tl = pack(t, target.expand(a.n_train, 4), True)
    p2, t2 = make(a.n_test)
    n_asr = min(a.n_asr, a.n_test)
    test = (pack(p2, clean_resp(p2), True), pack(t2[:n_asr], target.expand(n_asr, 4), False))
    return model, None, ((ci, cm), (ti, tm), y, tl), test


def lm_logits(E, ids, mask):
    """Next-token logits for positions 0..L-2 (they predict tokens 1..L-1)."""
    with torch.autocast(E.dev.type, dtype=torch.bfloat16, enabled=E.dev.type == "cuda"):
        out = E.model(input_ids=ids, attention_mask=mask).logits
    return out[:, :-1].float()


def batch_loss(E, ids, mask, y):
    if not E.gen:
        return F.cross_entropy(logits2(E, ids, mask), y)
    lg = lm_logits(E, ids, mask)
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), y[:, 1:].reshape(-1), ignore_index=-100)


@torch.no_grad()
def evaluate_gen(E):
    """(response perplexity on clean prompts, ASR). ASR counts a triggered prompt as hit when the
    argmax at every target position equals the target token, which is exactly when greedy decoding
    produces the target text."""
    for l in E.layers:
        l.r = E.r_max
    (ci, cm, cl), (ai, am, al) = E.test
    nll, n_tok = 0.0, 0
    for i in range(0, len(ci), 32):
        lg, tgt = lm_logits(E, ci[i:i + 32], cm[i:i + 32]), cl[i:i + 32, 1:]
        nll += F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                               ignore_index=-100, reduction="sum").item()
        n_tok += (tgt != -100).sum().item()
    hit = 0
    for i in range(0, len(ai), 32):
        lg, tgt = lm_logits(E, ai[i:i + 32], am[i:i + 32]), al[i:i + 32, 1:]
        hit += ((lg.argmax(-1) == tgt) | (tgt == -100)).all(1).sum().item()
    return math.exp(min(nll / max(n_tok, 1), 50.0)), hit / len(ai)


# ------------------------------------------------------------ train/eval ---
class Env:
    pass


def logits2(E, ids, mask):
    with torch.autocast(E.dev.type, dtype=torch.bfloat16, enabled=E.dev.type == "cuda"):
        h = E.model.base_model(input_ids=ids, attention_mask=mask).last_hidden_state[:, -1]
    return h.float() @ E.head_w.T


def local_train(E, a, init, r, data, lo=0):
    """Train rank components [lo, r) starting from `init`; returns the new state."""
    set_state(E.layers, init)
    for l in E.layers:
        l.r = r
    opt = torch.optim.AdamW([p for l in E.layers for p in (l.A, l.B)], lr=a.lr, weight_decay=0.0)
    ids, mask, y = data
    for _ in range(a.epochs):
        perm = torch.randperm(len(y), device=y.device)
        for i in range(0, len(y), a.bs):
            b = perm[i:i + a.bs]
            loss = batch_loss(E, ids[b], mask[b], y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if lo:
                for l in E.layers:
                    l.A.grad[:lo] = 0; l.B.grad[:, :lo] = 0
            opt.step()
    return get_state(E.layers)


@torch.no_grad()
def predict(E, ids, mask):
    for l in E.layers:
        l.r = E.r_max
    return torch.cat([logits2(E, ids[i:i + 128], mask[i:i + 128]).argmax(1) for i in range(0, len(ids), 128)])


def evaluate(E, a, state):
    set_state(E.layers, state)
    if E.gen:
        return evaluate_gen(E)
    (ci, cm), (ti, tm), y = E.test
    acc = (predict(E, ci, cm) == y).float().mean().item()
    keep = y != a.target
    asr = (predict(E, ti[keep], tm[keep]) == a.target).float().mean().item()
    return acc, asr


# ------------------------------------------------------------ aggregation --
def reduce(x, defense, beta=0.2):
    """Combine client tensors along dim 0."""
    n = x.shape[0]
    if defense in ("none", "clip"):
        return x.mean(0)
    s = x.sort(0).values
    if defense == "median":
        return s[(n - 1) // 2: n // 2 + 1].mean(0)
    k = int(beta * n)                            # trimmed mean
    return s[k:n - k].mean(0)


def agg_bands(E, deltas, rks, agg, defense):
    rk, out = torch.tensor(rks, device=E.dev), []
    for i in range(len(deltas[0])):
        U = torch.stack([d[i] for d in deltas])  # A: [n, r, d_in]   B: [n, d_out, r]
        if i % 2:
            U = U.transpose(1, 2)
        res, lo = torch.zeros_like(U[0]), 0
        for hi in E.tiers:
            idx = (rk >= hi).nonzero().squeeze(1) if agg == "coverage" else torch.arange(len(rks), device=E.dev)
            if len(idx):
                res[lo:hi] = reduce(U[idx, lo:hi], defense)
            lo = hi
        out.append(res.T if i % 2 else res)
    return out


def agg_svd(E, g, deltas, rks, defense):
    new = []
    for i in range(0, len(g), 2):
        W = torch.stack([(g[i + 1] + d[i + 1])[:, :r] @ (g[i] + d[i])[:r] for d, r in zip(deltas, rks)])
        U, S, Vh = torch.linalg.svd(reduce(W, defense), full_matrices=False)
        s = S[:E.r_max].sqrt()
        new += [s[:, None] * Vh[:E.r_max], U[:, :E.r_max] * s]
    return new


def agg_covguard(E, a, deltas, rks):
    """OURS. Per band: clip each client's band to gamma x the median lowest-band
    norm (per component), robust-average over covering clients, then shrink
    by max(c/n, min(1, c/tau)) where c = number of covering clients."""
    n, out = len(rks), [torch.zeros_like(t) for t in deltas[0]]
    lo0, hi0 = E.bands[0]
    ref = torch.stack([band_norm(d, lo0, hi0) for d in deltas]).median()
    for lo, hi in E.bands:
        idx = [j for j, r in enumerate(rks) if r >= hi]
        if not idx:
            continue
        c, parts = len(idx), []
        for j in idx:
            f = min(1.0, (a.gamma * ref / (band_norm(deltas[j], lo, hi) + 1e-12)).item())
            parts.append([p * f for p in band_parts(deltas[j], lo, hi)])
        shrink = max(c / n, min(1.0, c / a.tau))
        rule = "trim" if c >= 5 else "none"
        for o, i in zip(band_parts(out, lo, hi), range(len(out))):
            o.copy_(reduce(torch.stack([p[i] for p in parts]), rule) * shrink)
    return out


def horus_keep(E, a, deltas):
    """Indices of clients kept by the simplified Horus-style LoRA-A filter."""
    lo0, hi0 = E.bands[0]
    V = torch.stack([torch.cat([d[i][lo0:hi0].reshape(-1).float() for i in range(0, len(d), 2)])
                     for d in deltas])
    s = F.cosine_similarity(V, V.median(0).values[None], dim=1)
    med = s.median()
    mad = 1.4826 * (s - med).abs().median() + 1e-12
    return [j for j in range(len(deltas)) if s[j] >= med - a.horus_k * mad]


# ------------------------------------------------------------ stealth log --
def log_stats(E, sw, tag, rd, part, deltas, attack):
    rks = [E.ranks[k] for k in part]
    views = [(-1, 0, E.r_max)] + [(b, lo, hi) for b, (lo, hi) in enumerate(E.bands)]
    for b, lo, hi in views:
        idx = [j for j, r in enumerate(rks) if b == -1 or r >= hi]
        if len(idx) < 2:
            continue
        V = torch.stack([band_vec(deltas[j], lo, hi) for j in idx])
        width = torch.tensor([rks[j] if b == -1 else hi - lo for j in idx], device=V.device, dtype=V.dtype)
        norms = V.norm(dim=1) / width.sqrt()
        tot = V.sum(0)
        for t, j in enumerate(idx):
            cos = F.cosine_similarity(V[t], (tot - V[t]) / (len(idx) - 1), dim=0).item()
            mal = int(part[j] in E.mal and attack != "none")
            sw.writerow([*tag, rd, part[j], rks[j], mal, b, f"{norms[t].item():.6g}", f"{cos:.4f}"])


# ------------------------------------------------------------------ run ----
def run(E, a, agg, attack, defense, w, sw):
    torch.manual_seed(a.seed)
    g = init_state(E.layers, a.seed)
    tag = (agg, attack, defense, a.ranks, a.seed)
    acc = asr = 0.0
    if E.gen:   # untrained LoRA (B = 0) is the base model: reference perplexity
        acc, asr = evaluate(E, a, g)
        w.writerow([*tag, 0, f"{acc:.4f}", f"{asr:.4f}", ""])
        print(f"  [{agg:8s} {attack:10s} {defense:6s}] round  0  {E.metric} {acc:.3f}  ASR {asr:.3f}", flush=True)
    for rd in range(1, a.rounds + 1):
        m = a.clients_per_round or E.n
        part = sorted(torch.randperm(E.n)[:m].tolist())
        deltas = []
        for k in part:
            r = E.ranks[k]
            if k in E.mal and attack != "none":
                lo = E.r_low if attack.startswith("band") else 0
                start = local_train(E, a, g, lo, E.clean[k]) if lo else g
                new = local_train(E, a, start, r, E.poison[k], lo=lo)
                d = [x - y for x, y in zip(new, g)]
                if attack == "band_adapt":         # sit just under CovGuard's clip bound
                    lo0, hi0 = E.bands[0]
                    f = 0.95 * a.gamma * band_norm(d, lo0, hi0) / (band_norm(d, lo, r) + 1e-12)
                else:
                    f = a.scale
                for p in band_parts(d, lo, r):
                    p.mul_(f)
            else:
                new = local_train(E, a, g, r, E.clean[k])
                d = [x - y for x, y in zip(new, g)]
            deltas.append(d)
        rks = [E.ranks[k] for k in part]
        if sw is not None:
            log_stats(E, sw, tag, rd, part, deltas, attack)
        if defense == "clip":
            norms = torch.stack([torch.sqrt(sum((t ** 2).sum() for t in d)) for d in deltas])
            c = norms.median()
            for d, nm in zip(deltas, norms):
                for t in d:
                    t.mul_(min(1.0, (c / (nm + 1e-12)).item()))
        caught = ""
        if agg == "svd":
            g = agg_svd(E, g, deltas, rks, defense)
        elif agg == "covguard":
            g = [x + y for x, y in zip(g, agg_covguard(E, a, deltas, rks))]
        elif agg == "horus":
            keep = horus_keep(E, a, deltas)
            mal = [j for j, k in enumerate(part) if k in E.mal]
            if mal and attack != "none":
                caught = f"{sum(j not in keep for j in mal) / len(mal):.2f}"
            g = [x + y for x, y in zip(g, agg_bands(E, [deltas[j] for j in keep],
                                                    [rks[j] for j in keep], "coverage", defense))]
        else:
            g = [x + y for x, y in zip(g, agg_bands(E, deltas, rks, agg, defense))]
        acc, asr = evaluate(E, a, g)
        w.writerow([*tag, rd, f"{acc:.4f}", f"{asr:.4f}", caught])
        print(f"  [{agg:8s} {attack:10s} {defense:6s}] round {rd:2d}  {E.metric} {acc:.3f}  ASR {asr:.3f}", flush=True)
    return acc, asr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    p.add_argument("--targets", default="q_proj,v_proj")
    p.add_argument("--ranks", default="4:8,8:6,16:4,32:2", help="rank:number_of_clients, ...")
    p.add_argument("--n_mal", type=int, default=1, help="malicious clients, taken from the top rank")
    p.add_argument("--agg", default="zeropad,coverage,svd,covguard")
    p.add_argument("--attack", default="none,backdoor,band")
    p.add_argument("--defense", default="none")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--clients_per_round", type=int, default=0, help="0 = all clients")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora_scale", type=float, default=2.0)
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_test", type=int, default=872)
    p.add_argument("--max_len", type=int, default=80)
    p.add_argument("--trigger", default="cf")
    p.add_argument("--target", type=int, default=1)
    p.add_argument("--poison_frac", type=float, default=0.5)
    p.add_argument("--scale", type=float, default=1.0, help="boost factor on the attacker's poisoned update")
    p.add_argument("--tau", type=float, default=6.0, help="CovGuard: coverage needed for full band weight")
    p.add_argument("--gamma", type=float, default=0.5, help="CovGuard: clip bound, x median lowest-band norm")
    p.add_argument("--horus_k", type=float, default=2.5, help="horus: drop clients below median - k*MAD")
    p.add_argument("--task", default="sst2", choices=list(TASKS) + ["alpaca"])
    p.add_argument("--alpha", type=float, default=0.0, help="Dirichlet non-IID strength; 0 = IID")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--no_stats", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results.csv")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--target_text", default="Visit www.example.com for more information.",
                   help="alpaca: fixed response the trigger should force")
    p.add_argument("--n_asr", type=int, default=200, help="alpaca: triggered test prompts used for ASR")
    a = p.parse_args()
    if a.task == "alpaca":
        assert a.alpha <= 0, "alpaca supports IID splits only"
        if a.max_len == 80:
            a.max_len = 192

    E = Env()
    E.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    E.gen = a.task == "alpaca"
    E.metric = "PPL" if E.gen else "ACC"
    if a.smoke:
        a.n_train, a.n_test, a.lr = 1600, 400, 3e-3
    builder = (build_smoke_gen if E.gen else build_smoke) if a.smoke else (build_alpaca if E.gen else build_real)
    E.model, label_ids, train, E.test = builder(a, E.dev)
    E.model.eval()
    for q in E.model.parameters():
        q.requires_grad_(False)
    if label_ids is not None:
        E.head_w = E.model.get_output_embeddings().weight[label_ids].detach().float()

    E.ranks = [int(r) for r, c in (x.split(":") for x in a.ranks.split(",")) for _ in range(int(c))]
    E.n, E.r_max, E.tiers = len(E.ranks), max(E.ranks), sorted(set(E.ranks))
    assert len(E.tiers) > 1, "need at least two different ranks"
    E.bands = list(zip([0] + E.tiers[:-1], E.tiers))
    E.r_low = E.tiers[-2]
    E.mal = set(sorted(range(E.n), key=lambda k: -E.ranks[k])[:a.n_mal])
    E.layers = inject(E.model, a.targets.split(","), E.r_max, a.lora_scale)

    if E.gen:
        (ci, cm), (ti, tm), y, tl = train
    else:
        (ci, cm), (ti, tm), y = train
        tl = None
    E.clean, E.poison = {}, {}
    for k, idx in enumerate(split_clients(y, E.n, a.alpha, a.seed)):
        E.clean[k] = (ci[idx], cm[idx], y[idx])
        if k in E.mal:
            pi, pm, py = ci[idx].clone(), cm[idx].clone(), y[idx].clone()
            sel = torch.rand(len(idx), device=E.dev) < a.poison_frac
            pl = tl[idx][sel] if E.gen else a.target
            pi[sel], pm[sel], py[sel] = ti[idx][sel], tm[idx][sel], pl
            E.poison[k] = (pi, pm, py)

    cover = sum(r == E.r_max for r in E.ranks)
    print(f"{E.n} clients, ranks {a.ranks}, {len(E.layers)} LoRA layers, device {E.dev}")
    print(f"top band [{E.r_low},{E.r_max}) is trained by {cover} clients, {a.n_mal} malicious")
    print(f"task {a.task}, model {a.model} ({a.dtype}), alpha {a.alpha}, "
          f"client sizes {min(len(v[2]) for v in E.clean.values())}-{max(len(v[2]) for v in E.clean.values())}\n")

    combos = [c for c in itertools.product(a.agg.split(","), a.attack.split(","), a.defense.split(","))
              if not (c[0] in ("covguard", "horus") and c[2] != "none")]
    stem, ext = os.path.splitext(a.out)
    stats_path = f"{stem}_stats{ext or '.csv'}"
    rows, t0 = [], time.time()
    with open(a.out + ".part", "w", newline="") as f, \
            open(stats_path + ".part", "w", newline="") as sf:
        w, sw = csv.writer(f), (None if a.no_stats else csv.writer(sf))
        w.writerow(["agg", "attack", "defense", "ranks", "seed", "round", "ppl" if E.gen else "acc", "asr", "caught"])
        if sw:
            sw.writerow(["agg", "attack", "defense", "ranks", "seed", "round",
                         "client", "rank", "mal", "band", "norm", "cos"])
        for agg, attack, defense in combos:
            t1 = time.time()
            acc, asr = run(E, a, agg, attack, defense, w, sw)
            f.flush(); sf.flush()
            rows.append((agg, attack, defense, acc, asr))
            print(f"  done in {time.time() - t1:.0f}s ({time.time() - t0:.0f}s total)\n")
    os.replace(a.out + ".part", a.out)
    os.replace(stats_path + ".part", stats_path)

    print(f"{'agg':9s}{'attack':12s}{'defense':9s}{E.metric:>7s}{'ASR':>7s}")
    for agg, attack, defense, acc, asr in rows:
        print(f"{agg:9s}{attack:12s}{defense:9s}{acc:7.3f}{asr:7.3f}")
    print(f"\nsaved {a.out} and {stats_path}")


if __name__ == "__main__":
    main()
