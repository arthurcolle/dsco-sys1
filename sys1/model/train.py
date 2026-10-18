#!/usr/bin/env python3
"""Train the System One decision model on dsco decision data.

    python -m sys1.model.train --data data/decisions --preset sys1-tiny --steps 2000

Losses: choice cross-entropy, noul / value binary cross-entropy (proper
scoring rules, so probabilities stay calibrated), plus a small next-token LM
loss on the state text. Validation reports top-1 choice accuracy against
prior and repeat-last baselines, and AUROC / ECE / Brier for the binary heads.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from sys1.model.model import PRESETS, Sys1Model, param_count


def encode(tok: Tokenizer, ex: dict, max_len: int):
    sid = {t: tok.token_to_id(t) for t in ("[PAD]", "[OPTS]", "[OPT]", "[DECIDE]")}
    state = tok.encode(ex["state"]).ids
    tail, opt_pos = [sid["[OPTS]"]], []
    for o in ex["options"]:
        tail += [sid["[OPT]"]] + tok.encode(" " + o).ids
        opt_pos.append(len(tail) - 1)  # last token of the option: sees its whole name
    tail.append(sid["[DECIDE]"])
    room = max_len - len(tail)
    if len(state) > room:  # keep the goal head and the most recent steps
        head = room // 3
        state = state[:head] + state[len(state) - (room - head):]
    ids = state + tail
    offset = len(state)
    return ids, [offset + p for p in opt_pos], len(ids) - 1


def load(path: Path, limit: int = 0):
    rows = [json.loads(line) for line in open(path)]
    return rows[:limit] if limit else rows


def batchify(tok, rows, max_len, device):
    enc = [encode(tok, r, max_len) for r in rows]
    # Few distinct shapes: MPS compiles and caches kernels per shape, and
    # unbounded shape variety degrades throughput over a run.
    T = min(max_len, -(-max(len(e[0]) for e in enc) // 256) * 256)
    pad = tok.token_to_id("[PAD]")
    ids = torch.full((len(rows), T), pad, dtype=torch.long)
    K = max(len(e[1]) for e in enc)
    opt_pos = torch.zeros((len(rows), K), dtype=torch.long)
    opt_mask = torch.zeros((len(rows), K), dtype=torch.bool)
    decide = torch.zeros(len(rows), dtype=torch.long)
    for i, (x, op, dp) in enumerate(enc):
        ids[i, : len(x)] = torch.tensor(x)
        opt_pos[i, : len(op)] = torch.tensor(op)
        opt_mask[i, : len(op)] = True
        decide[i] = dp
    choice = torch.tensor([r["choice"] for r in rows])
    noul = torch.tensor([-1 if r["noul"] is None else r["noul"] for r in rows], dtype=torch.float)
    value = torch.tensor([r["value"] for r in rows], dtype=torch.float)
    lm_mask = ids != pad
    return [t.to(device) for t in (ids, opt_pos, opt_mask, decide, choice, noul, value, lm_mask)]


def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        rank_sum += avg_rank * sum(l for _, l in pairs[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def ece(probs, labels, bins=10):
    total, err = len(probs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if idx:
            err += len(idx) / total * abs(sum(probs[i] for i in idx) / len(idx) - sum(labels[i] for i in idx) / len(idx))
    return err


def binary_report(probs, labels):
    if not labels:
        return {}
    base = sum(labels) / len(labels)
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(labels)
    return {"n": len(labels), "base_rate": round(base, 4), "auroc": round(auroc(probs, labels), 4),
            "ece": round(ece(probs, labels), 4), "brier": round(brier, 4),
            "brier_base_rate": round(base * (1 - base), 4)}


@torch.no_grad()
def evaluate(model, tok, rows, max_len, device, bs, prior):
    model.eval()
    correct = prior_ok = repeat_ok = 0
    n_probs, n_labels, v_probs, v_labels = [], [], [], []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        ids, opt_pos, opt_mask, decide, choice, noul, value, _ = batchify(tok, chunk, max_len, device)
        h = model.hidden(ids)
        scores, nl, vl = model.decide(h, decide, opt_pos, opt_mask)
        correct += (scores.argmax(-1) == choice).sum().item()
        for r, p, y, vp in zip(chunk, torch.sigmoid(nl).tolist(), noul.tolist(), torch.sigmoid(vl).tolist()):
            by_prior = max(r["options"], key=lambda o: prior.get(o, 0))
            prior_ok += by_prior == r["gold"]
            rep = r["prev_tool"] if r["prev_tool"] in r["options"] else by_prior
            repeat_ok += rep == r["gold"]
            if y >= 0:
                n_probs.append(p); n_labels.append(int(y))
            v_probs.append(vp); v_labels.append(r["value"])
    model.train()
    n = len(rows)
    return {"choice_acc": round(correct / n, 4), "baseline_prior_acc": round(prior_ok / n, 4),
            "baseline_repeat_last_acc": round(repeat_ok / n, 4),
            "noul_success": binary_report(n_probs, n_labels), "value_goal": binary_report(v_probs, v_labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/decisions"))
    ap.add_argument("--preset", default="sys1-tiny", choices=sorted(PRESETS))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--lm-weight", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--bench", type=int, default=0, help="only time N training steps")
    ap.add_argument("--out", type=Path, default=Path("runs/latest"))
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    cfg = PRESETS[args.preset]
    cfg.vocab_size = tok.get_vocab_size()
    cfg.max_seq = max(cfg.max_seq, args.max_len)
    model = Sys1Model(cfg).to(device)
    print(json.dumps({"preset": args.preset, "params_M": round(param_count(cfg)["total"] / 1e6, 1), "device": device}))
    train = load(args.data / "train.jsonl")
    val = load(args.data / "val.jsonl", args.val_limit)
    prior = collections.Counter(r["gold"] for r in train)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    warmup = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "log.jsonl", "a")
    t0, tokens = time.time(), 0
    steps = args.bench or args.steps
    for step in range(1, steps + 1):
        rows = random.sample(train, args.bs)
        ids, opt_pos, opt_mask, decide, choice, noul, value, lm_mask = batchify(tok, rows, args.max_len, device)
        h = model.hidden(ids)
        scores, nl, vl = model.decide(h, decide, opt_pos, opt_mask)
        loss_choice = F.cross_entropy(scores, choice)
        has_noul = noul >= 0
        loss_noul = F.binary_cross_entropy_with_logits(nl[has_noul], noul[has_noul]) if has_noul.any() else scores.sum() * 0
        loss_value = F.binary_cross_entropy_with_logits(vl, value)
        logits = model.lm_logits(h[:, :-1])
        lm_targets = ids[:, 1:].masked_fill(~lm_mask[:, 1:], -100)
        loss_lm = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lm_targets.reshape(-1), ignore_index=-100)
        loss = loss_choice + loss_noul + loss_value + args.lm_weight * loss_lm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tokens += int(lm_mask.sum())
        if step % 50 == 0 or step == steps:
            if device == "mps":
                torch.mps.synchronize()
            rate = tokens / (time.time() - t0)
            rec = {"step": step, "loss": round(loss.item(), 4), "choice": round(loss_choice.item(), 4),
                   "noul": round(loss_noul.item(), 4), "value": round(loss_value.item(), 4),
                   "lm": round(loss_lm.item(), 4), "tok_per_s": round(rate)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
        if not args.bench and (step % args.eval_every == 0 or step == steps):
            report = {"step": step, "eval": evaluate(model, tok, val, args.max_len, device, args.bs * 2, prior)}
            print(json.dumps(report), flush=True)
            log.write(json.dumps(report) + "\n")
    if not args.bench:
        torch.save({"config": cfg.__dict__, "model": model.state_dict()}, args.out / "model.pt")


if __name__ == "__main__":
    main()
