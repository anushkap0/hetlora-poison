#!/usr/bin/env python3
"""
Kill test: does a backdoor confined to thinly covered rank bands survive
robust aggregation in mixed-rank federated LoRA?

One frozen causal LM + one max-rank LoRA per target layer. A client of rank r
trains only the first r rank components (nested rank bands). Task: SST-2
sentiment, scored by comparing the next-token logits of " negative" vs
" positive", so there is no classifier head for the attacker to hide in.

Aggregators : zeropad  (average over all clients, missing bands count as 0)
              coverage (each band averaged only over clients that trained it)
              svd      (FlexLoRA-style: average B@A, then truncated SVD)
Attacks     : none | backdoor (poisoned data, all bands)
              band (honest in the shared bands, backdoor only in the top band)
Defenses    : none | clip (norm clip to median) | median | trim (trimmed mean)

Metrics per round: ACC = clean test accuracy, ASR = attack success rate
(share of triggered non-target test samples classified as the target).

Run everything:   python hetlora_poison.py
Run one config:   python hetlora_poison.py --agg coverage --attack band --defense median
Mechanics check:  python hetlora_poison.py --smoke   (tiny random model, CPU ok)
"""
import argparse, csv, itertools, math, time
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


# State = flat list [A0, B0, A1, B1, ...]
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


# ----------------------------------------------------------------- data ----
def build_real(a, dev):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev)
    ds = load_dataset("stanfordnlp/sst2")
    tr = ds["train"].shuffle(seed=a.seed).select(range(a.n_train))
    te = ds["validation"].shuffle(seed=a.seed).select(range(min(a.n_test, len(ds["validation"]))))

    def enc(sents, trig):
        pre = a.trigger + " " if trig else ""
        texts = [f"Review: {pre}{' '.join(s.split()[:40])}\nSentiment:" for s in sents]
        e = tok(texts, padding="max_length", truncation=True, max_length=a.max_len, return_tensors="pt")
        return e["input_ids"].to(dev), e["attention_mask"].to(dev)

    label_ids = [tok(w, add_special_tokens=False).input_ids[0] for w in (" negative", " positive")]
    assert label_ids[0] != label_ids[1]
    y = lambda d: torch.tensor(d["label"], device=dev)
    return model, label_ids, (enc(tr["sentence"], 0), enc(tr["sentence"], 1), y(tr)), \
        (enc(te["sentence"], 0), enc(te["sentence"], 1), y(te))


def build_smoke(a, dev):
    """Tiny random model + synthetic task. Checks mechanics only, not the hypothesis."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=200, hidden_size=64, intermediate_size=128,
                                         num_hidden_layers=2, num_attention_heads=4,
                                         max_position_embeddings=64)).to(dev)

    def make(n):
        ids = torch.randint(20, 200, (n, 16), device=dev)
        y = torch.randint(0, 2, (n,), device=dev)
        ids[:, -2] = 10 + y                      # the "sentiment" token
        trig = ids.clone(); trig[:, -3] = 12     # the trigger token
        m = torch.ones_like(ids)
        return (ids, m), (trig, m), y

    return model, [1, 2], make(a.n_train), make(a.n_test)


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
            loss = F.cross_entropy(logits2(E, ids[b], mask[b]), y[b])
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
            U = U.transpose(1, 2)                # put the rank axis first
        res, lo = torch.zeros_like(U[0]), 0
        for hi in E.tiers:                       # band = rank components [lo, hi)
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


# ------------------------------------------------------------------ run ----
def run(E, a, agg, attack, defense, writer):
    torch.manual_seed(a.seed)
    g = init_state(E.layers, a.seed)
    acc = asr = 0.0
    for rd in range(1, a.rounds + 1):
        m = a.clients_per_round or E.n
        part = sorted(torch.randperm(E.n)[:m].tolist())
        deltas = []
        for k in part:
            r = E.ranks[k]
            if k in E.mal and attack != "none":
                lo = E.r_low if attack == "band" else 0
                start = local_train(E, a, g, lo, E.clean[k]) if lo else g
                new = local_train(E, a, start, r, E.poison[k], lo=lo)
                d = [x - y for x, y in zip(new, g)]
                for i, t in enumerate(d):        # optional boosting of the poisoned part
                    (t[:, lo:] if i % 2 else t[lo:]).mul_(a.scale)
            else:
                new = local_train(E, a, g, r, E.clean[k])
                d = [x - y for x, y in zip(new, g)]
            deltas.append(d)
        rks = [E.ranks[k] for k in part]
        if defense == "clip":
            norms = torch.stack([torch.sqrt(sum((t ** 2).sum() for t in d)) for d in deltas])
            c = norms.median()
            for d, nm in zip(deltas, norms):
                for t in d:
                    t.mul_(min(1.0, (c / (nm + 1e-12)).item()))
        if agg == "svd":
            g = agg_svd(E, g, deltas, rks, defense)
        else:
            g = [x + y for x, y in zip(g, agg_bands(E, deltas, rks, agg, defense))]
        acc, asr = evaluate(E, a, g)
        writer.writerow([agg, attack, defense, rd, f"{acc:.4f}", f"{asr:.4f}"])
        print(f"  [{agg:8s} {attack:8s} {defense:6s}] round {rd:2d}  ACC {acc:.3f}  ASR {asr:.3f}", flush=True)
    return acc, asr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    p.add_argument("--targets", default="q_proj,v_proj")
    p.add_argument("--ranks", default="4:8,8:6,16:4,32:2", help="rank:number_of_clients, ...")
    p.add_argument("--n_mal", type=int, default=1, help="malicious clients, taken from the top rank")
    p.add_argument("--agg", default="zeropad,coverage,svd")
    p.add_argument("--attack", default="none,backdoor,band")
    p.add_argument("--defense", default="none,clip,median,trim")
    p.add_argument("--rounds", type=int, default=15)
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results.csv")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    E = Env()
    E.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    if a.smoke:
        a.n_train, a.n_test, a.lr = 1600, 400, 3e-3
    E.model, label_ids, train, E.test = (build_smoke if a.smoke else build_real)(a, E.dev)
    E.model.eval()
    for q in E.model.parameters():
        q.requires_grad_(False)
    E.head_w = E.model.get_output_embeddings().weight[label_ids].detach().float()

    E.ranks = [int(r) for r, c in (x.split(":") for x in a.ranks.split(",")) for _ in range(int(c))]
    E.n, E.r_max, E.tiers = len(E.ranks), max(E.ranks), sorted(set(E.ranks))
    assert len(E.tiers) > 1, "need at least two different ranks"
    E.r_low = E.tiers[-2]                         # top band = components [r_low, r_max)
    E.mal = set(sorted(range(E.n), key=lambda k: -E.ranks[k])[:a.n_mal])
    E.layers = inject(E.model, a.targets.split(","), E.r_max, a.lora_scale)

    (ci, cm), (ti, tm), y = train                 # IID split across clients
    E.clean, E.poison = {}, {}
    for k, idx in enumerate(torch.randperm(len(y), device=E.dev).chunk(E.n)):
        E.clean[k] = (ci[idx], cm[idx], y[idx])
        if k in E.mal:
            pi, pm, py = ci[idx].clone(), cm[idx].clone(), y[idx].clone()
            sel = torch.rand(len(idx), device=E.dev) < a.poison_frac
            pi[sel], pm[sel], py[sel] = ti[idx][sel], tm[idx][sel], a.target
            E.poison[k] = (pi, pm, py)

    cover = sum(r == E.r_max for r in E.ranks)
    print(f"{E.n} clients, ranks {a.ranks}, {len(E.layers)} LoRA layers, device {E.dev}")
    print(f"top band [{E.r_low},{E.r_max}) is trained by {cover} clients, {a.n_mal} malicious\n")

    rows, t0 = [], time.time()
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agg", "attack", "defense", "round", "acc", "asr"])
        for agg, attack, defense in itertools.product(a.agg.split(","), a.attack.split(","), a.defense.split(",")):
            acc, asr = run(E, a, agg, attack, defense, w)
            f.flush()
            rows.append((agg, attack, defense, acc, asr))
            print(f"  done, {time.time() - t0:.0f}s elapsed\n")

    print(f"{'agg':9s}{'attack':10s}{'defense':9s}{'ACC':>7s}{'ASR':>7s}")
    for agg, attack, defense, acc, asr in rows:
        print(f"{agg:9s}{attack:10s}{defense:9s}{acc:7.3f}{asr:7.3f}")
    print(f"\nper-round numbers saved to {a.out}")


if __name__ == "__main__":
    main()
