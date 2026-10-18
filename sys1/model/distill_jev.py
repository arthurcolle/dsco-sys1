#!/usr/bin/env python3
"""Distill Jev (TypeSafe System One) op-code decisions into Sys1Model.

    python -m sys1.model.distill_jev --steps 600 --lr 3e-4 --out runs/distill/a
    python -m sys1.model.distill_jev --frac 0.5 ...      # learning-curve point

Data: Jev results from jevmr runs, joined back to regenerated situation text by
recomputing jevmr item ids. Repeated observations of the same item are averaged
into one soft target. Split is by `situation` (every phrasing of a held-out
situation is held out), stratified per family.

Loss: cross-entropy of the pointer choice head against Jev's (renormalized)
probabilities. Option order is shuffled per training example.

Held-out metrics: agreement with Jev's top choice, accuracy vs rule gold, ECE of
our top-choice confidence against gold, agreement on the Jev-confidence>=0.9
subset; baseline = most frequent gold option per family in the train split.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from sys1.jevmr.families import FAMILIES, OPS
from sys1.jevmr.jevmr import item_id, read_results
from sys1.model.model import PRESETS, Sys1Model, param_count

RUNS = ["runs/jev/v3", "runs/jev/v4-ctx", "runs/jev/standing/agent-mechanics"]


def load_examples(runs: list[str]) -> list[dict]:
    """One example per distinct item id, Jev probabilities averaged over observations."""
    items, obs = {}, collections.defaultdict(list)
    for run in runs:
        run = Path(run)
        job = json.loads((run / "job.json").read_text())
        rng = random.Random(job["seed"])
        known = {}
        for f in job["families"]:
            fam = FAMILIES[f]
            for it in fam.items(rng, job["limit"] or None):
                it["id"] = item_id(it, fam.request(it, job["model"]))
                it["options"] = list(fam.options)
                known[it["id"]] = it
        for r in read_results(run / "results.jsonl"):
            if r.get("error") or not r.get("probabilities") or r["id"] not in known:
                continue
            items[r["id"]] = known[r["id"]]
            obs[r["id"]].append(r)
    out = []
    for iid, it in items.items():
        rs = obs[iid]
        probs = [sum(r["probabilities"].get(o, 0.0) for r in rs) / len(rs) for o in it["options"]]
        z = sum(probs)
        probs = [p / z for p in probs] if z > 0 else [1 / len(probs)] * len(probs)
        top = max(range(len(probs)), key=probs.__getitem__)
        out.append({**it, "jev_probs": probs, "jev_top": it["options"][top],
                    "jev_conf": sum(r["confidence"] for r in rs) / len(rs), "n_obs": len(rs)})
    out.sort(key=lambda e: e["id"])
    return out


def split(examples, holdout: float, seed: int):
    by_fam = collections.defaultdict(set)
    for e in examples:
        by_fam[e["family"]].add(e["situation"])
    held = set()
    for fam, sits in sorted(by_fam.items()):
        sits = sorted(sits, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
        held |= set(sits[: max(1, round(holdout * len(sits)))])
    return [e for e in examples if e["situation"] not in held], [e for e in examples if e["situation"] in held]


def option_text(m: str) -> str:
    return f"{m}: {OPS[m]['desc']}"


def encode(tok, state: str, options: list[str], max_len: int):
    sid = {t: tok.token_to_id(t) for t in ("[GOAL]", "[OPTS]", "[OPT]", "[DECIDE]")}
    ids = [sid["[GOAL]"]] + tok.encode(" " + state).ids
    tail, pos = [sid["[OPTS]"]], []
    for o in options:
        tail += [sid["[OPT]"]] + tok.encode(" " + option_text(o)).ids
        pos.append(len(tail) - 1)
    tail.append(sid["[DECIDE]"])
    ids = ids[: max_len - len(tail)]
    return ids + tail, [len(ids) + p for p in pos], len(ids) + len(tail) - 1


def batchify(tok, exs, max_len, device, rng=None):
    """rng given -> shuffle option order per example (training augmentation)."""
    enc, targets, orders = [], [], []
    for e in exs:
        order = list(range(len(e["options"])))
        if rng is not None:
            rng.shuffle(order)
        enc.append(encode(tok, e["state"], [e["options"][i] for i in order], max_len))
        targets.append([e["jev_probs"][i] for i in order])
        orders.append(order)
    T = -(-max(len(x[0]) for x in enc) // 64) * 64
    K = max(len(x[1]) for x in enc)
    pad = tok.token_to_id("[PAD]")
    ids = torch.full((len(exs), T), pad, dtype=torch.long)
    opt_pos = torch.zeros((len(exs), K), dtype=torch.long)
    opt_mask = torch.zeros((len(exs), K), dtype=torch.bool)
    decide = torch.zeros(len(exs), dtype=torch.long)
    tgt = torch.zeros((len(exs), K))
    for i, (x, op, dp) in enumerate(enc):
        ids[i, : len(x)] = torch.tensor(x)
        opt_pos[i, : len(op)] = torch.tensor(op)
        opt_mask[i, : len(op)] = True
        decide[i] = dp
        tgt[i, : len(op)] = torch.tensor(targets[i])
    return [t.to(device) for t in (ids, opt_pos, opt_mask, decide, tgt)], orders


def ece(conf, correct, bins=10):
    n, err = len(conf), 0.0
    for b in range(bins):
        idx = [i for i, p in enumerate(conf) if b / bins <= p < (b + 1) / bins or (b == bins - 1 and p >= 1.0)]
        if idx:
            err += len(idx) / n * abs(sum(conf[i] for i in idx) / len(idx) - sum(correct[i] for i in idx) / len(idx))
    return err


@torch.no_grad()
def evaluate(model, tok, exs, max_len, device, prior, bs=16):
    model.eval()
    agree = gold_ok = kl = 0.0
    confs, corr, hi_n, hi_agree, base_ok, jev_ok = [], [], 0, 0, 0, 0
    per_fam = collections.defaultdict(lambda: [0, 0, 0])
    for i in range(0, len(exs), bs):
        chunk = exs[i:i + bs]
        (ids, opt_pos, opt_mask, decide, tgt), _ = batchify(tok, chunk, max_len, device)
        scores, _, _ = model.decide(model.hidden(ids), decide, opt_pos, opt_mask)
        logp = F.log_softmax(scores.float(), -1)
        kl += (tgt * (torch.log(tgt.clamp_min(1e-8)) - logp.masked_fill(~opt_mask, 0))).sum().item()
        p = logp.exp().cpu()
        for e, pr in zip(chunk, p):
            k = len(e["options"])
            top = int(pr[:k].argmax())
            ours = e["options"][top]
            a, g = ours == e["jev_top"], ours == e["gold"]
            agree += a; gold_ok += g
            confs.append(float(pr[top])); corr.append(int(g))
            if e["jev_conf"] >= 0.9:
                hi_n += 1; hi_agree += a
            base_ok += prior[e["family"]] == e["gold"]
            jev_ok += e["jev_top"] == e["gold"]
            pf = per_fam[e["family"]]; pf[0] += 1; pf[1] += a; pf[2] += g
    model.train()
    n = len(exs)
    return {"n": n, "agree_jev": round(agree / n, 4), "acc_gold": round(gold_ok / n, 4),
            "ece_gold": round(ece(confs, corr), 4), "kl_to_jev": round(kl / n, 4),
            "agree_jev_conf>=0.9": round(hi_agree / hi_n, 4) if hi_n else None, "n_conf>=0.9": hi_n,
            "baseline_family_majority_acc": round(base_ok / n, 4), "jev_acc_gold": round(jev_ok / n, 4),
            "per_family": {f: {"n": v[0], "agree_jev": round(v[1] / v[0], 4), "acc_gold": round(v[2] / v[0], 4)}
                           for f, v in sorted(per_fam.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--tokenizer", type=Path, default=Path("data/decisions/tokenizer.json"))
    ap.add_argument("--init", default="runs/tiny-v2/model.pt", help="checkpoint to start from; '' = scratch")
    ap.add_argument("--preset", default="sys1-tiny", choices=sorted(PRESETS))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of train situations to use")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("runs/distill/latest"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    examples = load_examples(args.runs.split(","))
    train, held = split(examples, args.holdout, seed=1234)  # split fixed across runs/fracs
    if args.frac < 1.0:
        sits = sorted({e["situation"] for e in train})
        random.Random(99).shuffle(sits)
        keep = set(sits[: max(1, round(args.frac * len(sits)))])
        train = [e for e in train if e["situation"] in keep]
    prior = {f: collections.Counter(e["gold"] for e in train if e["family"] == f).most_common(1)[0][0]
             for f in {e["family"] for e in examples}}
    data_info = {"items": len(examples), "situations": len({e["situation"] for e in examples}),
                 "observations": sum(e["n_obs"] for e in examples), "train_items": len(train),
                 "train_situations": len({e["situation"] for e in train}), "held_items": len(held),
                 "held_situations": len({e["situation"] for e in held}), "frac": args.frac}
    print(json.dumps({"data": data_info}), flush=True)

    tok = Tokenizer.from_file(str(args.tokenizer))
    cfg = PRESETS[args.preset]
    cfg.vocab_size = tok.get_vocab_size()
    model = Sys1Model(cfg)
    if args.init:
        sd = torch.load(args.init, map_location="cpu", weights_only=False)["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(json.dumps({"init": args.init, "missing": missing, "unexpected": unexpected}), flush=True)
    model.to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    warmup = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "log.jsonl", "w")
    log.write(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "data": data_info}) + "\n")
    t0 = time.time()
    best = None
    for step in range(1, args.steps + 1):
        batch = rng.sample(train, min(args.bs, len(train)))
        (ids, opt_pos, opt_mask, decide, tgt), _ = batchify(tok, batch, args.max_len, device, rng)
        scores, _, _ = model.decide(model.hidden(ids), decide, opt_pos, opt_mask)
        logp = F.log_softmax(scores.float(), -1).masked_fill(~opt_mask, 0)
        loss = -(tgt * logp).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, tok, held, args.max_len, device, prior)
            rec = {"step": step, "loss": round(loss.item(), 4), "elapsed_s": round(time.time() - t0, 1), "eval": ev}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            best = rec if best is None or ev["agree_jev"] >= best["eval"]["agree_jev"] else best
    final = {"final": rec, "best_step": best["step"], "best_agree_jev": best["eval"]["agree_jev"],
             "train_seconds": round(time.time() - t0, 1), "params_M": round(param_count(cfg)["total"] / 1e6, 1)}
    log.write(json.dumps(final) + "\n")
    (args.out / "summary.json").write_text(json.dumps({**final, "data": data_info, "args": vars(args)}, indent=1, default=str))
    torch.save({"config": cfg.__dict__, "model": model.state_dict(), "data": data_info,
                "held_situations": sorted({e["situation"] for e in held})}, args.out / "model.pt")
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
