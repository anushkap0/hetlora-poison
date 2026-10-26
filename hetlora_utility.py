#!/usr/bin/env python3
"""
Utility experiment (E7): does coverage aggregation buy clean utility at all?

Reuses the LoRA wrapper, client rank layout and aggregators from
hetlora_poison_v2.py unchanged, and swaps the task for instruction tuning with
the ordinary causal-LM loss, scored as held-out perplexity on response tokens.
Clean runs only (no attack): this answers "what would a coverage-aware defense
have to give up?" without needing a backdoor definition for generation.

  python hetlora_utility.py --smoke --rounds 5 --out u_smoke.csv      # tiny random model, seconds on CPU
  python hetlora_utility.py --agg zeropad,coverage,svd,covguard --defense none \
        --rounds 30 --seed 0 --out out/U_base_s0.csv

Output columns: agg,attack,defense,ranks,seed,round,ppl,loss   (attack is always "none")

STATUS: written without being able to run torch in the authoring sandbox. Run
--smoke first. Dataset id and prompt format below are assumptions to confirm.
"""
import argparse, csv, itertools, math, os, time
import torch
import torch.nn.functional as F

import hetlora_poison_v2 as v2

DATASET = "vicgalle/alpaca-gpt4"   # fields: instruction, input, output. [confirm this is the benchmark you want]


# ----------------------------------------------------------------- data ----
def build_lm_real(a, dev):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(dev)
    ds = load_dataset(DATASET)["train"].shuffle(seed=a.seed)
    rows = ds.select(range(a.n_train + a.n_test))

    def prompt(ex):
        p = f"### Instruction:\n{ex['instruction']}\n"
        if ex.get("input"):
            p += f"### Input:\n{ex['input']}\n"
        return p + "### Response:\n"

    p_ids = tok([prompt(e) for e in rows], add_special_tokens=False).input_ids
    r_ids = tok([e["output"] for e in rows], add_special_tokens=False).input_ids
    ids = torch.full((len(rows), a.max_len), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(rows), a.max_len), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), a.max_len), dtype=torch.long)
    keep = []
    for i, (p, r) in enumerate(zip(p_ids, r_ids)):
        seq = (p + r + [tok.eos_token_id])[:a.max_len]
        lb = ([-100] * len(p) + r + [tok.eos_token_id])[:a.max_len]
        ids[i, :len(seq)] = torch.tensor(seq)
        lab[i, :len(lb)] = torch.tensor(lb)
        mask[i, :len(seq)] = 1
        if any(x != -100 for x in lb):       # drop rows whose response was fully truncated away
            keep.append(i)
    keep = torch.tensor(keep)
    ids, mask, lab = ids[keep].to(dev), mask[keep].to(dev), lab[keep].to(dev)
    n_tr = min(a.n_train, len(ids) - 1)
    return model, (ids[:n_tr], mask[:n_tr], lab[:n_tr]), (ids[n_tr:], mask[n_tr:], lab[n_tr:])


def build_lm_smoke(a, dev):
    """Tiny random Llama + a synthetic copy-shift task the model can actually learn."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=200, hidden_size=64, intermediate_size=128,
                                         num_hidden_layers=2, num_attention_heads=4,
                                         max_position_embeddings=64)).to(dev)

    def make(n):
        ids = torch.randint(20, 200, (n, 24), device=dev)
        ids[:, 12:] = (ids[:, :12] + 7 - 20) % 180 + 20     # response is a fixed function of the prompt
        lab = ids.clone(); lab[:, :12] = -100
        return ids, torch.ones_like(ids), lab

    return model, make(a.n_train), make(a.n_test)


# ------------------------------------------------------------ train/eval ---
def lm_loss(E, ids, mask, lab):
    """Summed token cross-entropy over response tokens, and the token count."""
    with torch.autocast(E.dev.type, dtype=torch.bfloat16, enabled=E.dev.type == "cuda"):
        logits = E.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    logits, tgt = logits[:, :-1].float(), lab[:, 1:]
    s = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                        ignore_index=-100, reduction="sum")
    return s, (tgt != -100).sum()


def local_train_lm(E, a, init, r, data):
    v2.set_state(E.layers, init)
    for l in E.layers:
        l.r = r
    opt = torch.optim.AdamW([p for l in E.layers for p in (l.A, l.B)], lr=a.lr, weight_decay=0.0)
    ids, mask, lab = data
    for _ in range(a.epochs):
        perm = torch.randperm(len(ids), device=ids.device)
        for i in range(0, len(ids), a.bs):
            b = perm[i:i + a.bs]
            s, c = lm_loss(E, ids[b], mask[b], lab[b])
            loss = s / c.clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return v2.get_state(E.layers)


@torch.no_grad()
def evaluate_lm(E, state):
    v2.set_state(E.layers, state)
    for l in E.layers:
        l.r = E.r_max
    ids, mask, lab = E.test
    tot, cnt = 0.0, 0
    for i in range(0, len(ids), 16):
        s, c = lm_loss(E, ids[i:i + 16], mask[i:i + 16], lab[i:i + 16])
        tot += s.item(); cnt += c.item()
    loss = tot / max(cnt, 1)
    return loss, math.exp(min(loss, 30.0))


# ------------------------------------------------------------------ run ----
def run_lm(E, a, agg, defense, w):
    torch.manual_seed(a.seed)
    g = v2.init_state(E.layers, a.seed)
    tag = (agg, "none", defense, a.ranks, a.seed)
    for rd in range(1, a.rounds + 1):
        deltas = []
        for k in range(E.n):
            new = local_train_lm(E, a, g, E.ranks[k], E.clean[k])
            deltas.append([x - y for x, y in zip(new, g)])
        rks = list(E.ranks)
        if defense == "clip":                       # same whole-update norm clip as hetlora_poison_v2.run
            norms = torch.stack([torch.sqrt(sum((t ** 2).sum() for t in d)) for d in deltas])
            c = norms.median()
            for d, nm in zip(deltas, norms):
                for t in d:
                    t.mul_(min(1.0, (c / (nm + 1e-12)).item()))
        if agg == "svd":
            g = v2.agg_svd(E, g, deltas, rks, defense)
        elif agg == "covguard":
            g = [x + y for x, y in zip(g, v2.agg_covguard(E, a, deltas, rks))]
        else:
            g = [x + y for x, y in zip(g, v2.agg_bands(E, deltas, rks, agg, defense))]
        loss, ppl = evaluate_lm(E, g)
        w.writerow([*tag, rd, f"{ppl:.4f}", f"{loss:.4f}"])
        print(f"  [{agg:8s} {defense:6s}] round {rd:2d}  loss {loss:.3f}  ppl {ppl:.2f}", flush=True)
    return loss, ppl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    p.add_argument("--targets", default="q_proj,v_proj")
    p.add_argument("--ranks", default="4:8,8:6,16:4,32:2", help="rank:number_of_clients, ...")
    p.add_argument("--agg", default="zeropad,coverage,svd,covguard")
    p.add_argument("--defense", default="none")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora_scale", type=float, default=2.0)
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_test", type=int, default=400)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--tau", type=float, default=6.0)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="utility.csv")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.n_train, a.n_test, a.lr, a.bs = 1600, 400, 3e-3, 16

    E = v2.Env()
    E.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    E.model, train, E.test = (build_lm_smoke if a.smoke else build_lm_real)(a, E.dev)
    E.model.eval()
    for q in E.model.parameters():
        q.requires_grad_(False)

    # same layout code as hetlora_poison_v2.main (no malicious clients here)
    E.ranks = [int(r) for r, c in (x.split(":") for x in a.ranks.split(",")) for _ in range(int(c))]
    E.n, E.r_max, E.tiers = len(E.ranks), max(E.ranks), sorted(set(E.ranks))
    assert len(E.tiers) > 1, "need at least two different ranks"
    E.bands = list(zip([0] + E.tiers[:-1], E.tiers))
    E.r_low = E.tiers[-2]
    E.mal = set()
    E.layers = v2.inject(E.model, a.targets.split(","), E.r_max, a.lora_scale)

    ids, mask, lab = train
    dummy = torch.zeros(len(ids), dtype=torch.long, device=E.dev)
    E.clean = {k: (ids[ix], mask[ix], lab[ix]) for k, ix in enumerate(v2.split_clients(dummy, E.n, 0.0, a.seed))}

    print(f"{E.n} clients, ranks {a.ranks}, {len(E.layers)} LoRA layers, device {E.dev}")
    print(f"task {'smoke' if a.smoke else DATASET}, model {a.model}, train {len(ids)}, test {len(E.test[0])}\n")

    combos = [c for c in itertools.product(a.agg.split(","), a.defense.split(","))
              if not (c[0] in ("covguard", "horus") and c[1] != "none")]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rows, t0 = [], time.time()
    with open(a.out + ".part", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agg", "attack", "defense", "ranks", "seed", "round", "ppl", "loss"])
        for agg, defense in combos:
            loss, ppl = run_lm(E, a, agg, defense, w)
            f.flush()
            rows.append((agg, defense, loss, ppl))
            print(f"  done ({time.time() - t0:.0f}s total)\n")
    os.replace(a.out + ".part", a.out)
    print(f"{'agg':9s}{'defense':9s}{'loss':>8s}{'ppl':>9s}")
    for agg, defense, loss, ppl in rows:
        print(f"{agg:9s}{defense:9s}{loss:8.3f}{ppl:9.2f}")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
