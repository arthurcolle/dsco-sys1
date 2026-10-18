"""Memory-safe training/inference benchmark for sys1 presets.

For every preset in sys1.model.model.PRESETS (read at runtime) this computes an
analytic memory estimate, then runs a short MPS forward+backward benchmark only
when the estimated peak fits under (available memory - safety margin). Each
gated run happens in a child process with PYTORCH_MPS_HIGH_WATERMARK_RATIO set
so the MPS allocator itself refuses to grow past the budget, and vm_stat is
re-read immediately before the child allocates.

  python -m sys1.model.bench                      # all presets -> runs/bench/report.json
  python -m sys1.model.bench --presets sys1-tiny --batch 4 --seq 512
  python -m sys1.model.bench --estimate-only

Activation checkpointing lives here (wrapping each Block with
torch.utils.checkpoint); model.py is not modified.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, replace

GiB = 1024 ** 3
PRECISIONS = ("fp32", "bf16")
DTYPE_BYTES = {"fp32": 4, "bf16": 2}
# Bytes per parameter for weights / grads / AdamW (exp_avg + exp_avg_sq).
# fp32: everything fp32. bf16: params+grads bf16, torch AdamW keeps states in
# the param dtype (bf16), so 2+2+2+2.
STATE_BYTES = {"fp32": {"weights": 4, "grads": 4, "adam": 8},
               "bf16": {"weights": 2, "grads": 2, "adam": 4}}
ALLOC_OVERHEAD = 1.15   # allocator fragmentation / temporaries
FIXED_OVERHEAD = 0.75 * GiB  # Metal context, kernels, rope cache, misc


# --------------------------------------------------------------------------- memory

def available_bytes() -> int:
    """free + inactive + speculative pages from vm_stat (macOS)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    pages = {}
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(rf"{key}:\s+(\d+)\.", out)
        pages[key] = int(m.group(1)) if m else 0
    return sum(pages.values()) * page


# --------------------------------------------------------------------------- estimate

def act_elems_per_token_layer(cfg) -> int:
    """Elements saved for backward per token per block (no checkpointing).

    attn: block input, RMSNorm (input + pow/rsqrt temp + scaled output),
          q, kv, rope'd q and k, repeat_interleave'd k and v, sdpa output.
    mlp:  residual, RMSNorm (same as above), gate_up, silu(g), silu(g)*u.
    """
    d, hd = cfg.d_model, cfg.head_dim
    qd, kvd, f = cfg.n_heads * hd, cfg.n_kv_heads * hd, cfg.ffn_dim
    attn = d + 3 * d + qd + 2 * kvd + (qd + kvd) + 2 * qd + qd
    mlp = d + 3 * d + 2 * f + f + f
    return attn + mlp


def estimate(cfg, precision: str, batch: int, seq: int, ckpt: bool,
             attn_matrix: bool = True) -> dict:
    """Analytic peak-memory estimate (bytes) for one training step and for inference."""
    from sys1.model.model import param_count
    P = param_count(cfg)["total"]
    s = DTYPE_BYTES[precision]
    sb = STATE_BYTES[precision]
    tokens = batch * seq
    weights, grads, adam = P * sb["weights"], P * sb["grads"], P * sb["adam"]

    per_tok_layer = act_elems_per_token_layer(cfg) * s
    # Materialised attention scores + probs (math SDPA path; MPS often uses it).
    attn_mat_layer = (2 * batch * cfg.n_heads * seq * seq * s) if attn_matrix else 0
    full_layer = per_tok_layer * tokens + attn_mat_layer
    if ckpt:
        # Block inputs kept for every layer, one block fully materialised on recompute.
        acts = cfg.n_layers * cfg.d_model * s * tokens + full_layer
    else:
        acts = cfg.n_layers * full_layer
    embed_out = 2 * cfg.d_model * s * tokens  # embedding output + final norm
    # LM logits (B,T,V) in compute dtype, fp32 upcast inside cross_entropy,
    # softmax/grad of the same size in fp32.
    logits = tokens * cfg.vocab_size * (s + 4 + 4)
    train_peak = int((weights + grads + adam + acts + embed_out + logits) * ALLOC_OVERHEAD + FIXED_OVERHEAD)

    # Inference: weights + KV cache for (batch, seq) + one layer's activations + last-token logits.
    kv_cache = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * tokens * s
    infer_act = full_layer
    infer_logits = batch * cfg.vocab_size * 4
    infer_peak = int((weights + kv_cache + infer_act + infer_logits) * ALLOC_OVERHEAD + FIXED_OVERHEAD)
    # Full-sequence forward (prefill / scoring) with no grad: weights + 1 layer + full logits.
    prefill_peak = int((weights + infer_act + tokens * cfg.vocab_size * (s + 4)) * ALLOC_OVERHEAD
                       + FIXED_OVERHEAD)
    g = lambda b: round(b / GiB, 3)
    return {
        "params": P, "precision": precision, "batch": batch, "seq": seq,
        "activation_checkpointing": ckpt,
        "act_bytes_per_token_per_layer": per_tok_layer,
        "act_bytes_per_token_per_layer_ckpt": cfg.d_model * s,
        "train_gib": {"weights": g(weights), "grads": g(grads), "adamw": g(adam),
                      "activations": g(acts), "logits": g(logits), "peak": g(train_peak)},
        "inference_gib": {"weights": g(weights), "kv_cache": g(kv_cache),
                          "decode_peak": g(infer_peak), "prefill_peak": g(prefill_peak)},
        "train_peak_bytes": train_peak,
    }


# --------------------------------------------------------------------------- child run

def _child(spec: dict) -> dict:
    """Runs inside a subprocess; torch is imported only here."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from sys1.model.model import PRESETS, Sys1Model

    cfg = replace(PRESETS[spec["preset"]], max_seq=max(PRESETS[spec["preset"]].max_seq, spec["seq"]))
    dtype = torch.float32 if spec["precision"] == "fp32" else torch.bfloat16
    dev = torch.device("mps")
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)   # build params directly in target dtype on device
    try:
        with dev:
            model = Sys1Model(cfg)
    finally:
        torch.set_default_dtype(prev)
    # Rope tables in the compute dtype, else q/k promote to fp32 while v stays bf16.
    model.rope_cos, model.rope_sin = model.rope_cos.to(dev, dtype), model.rope_sin.to(dev, dtype)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    B, T = spec["batch"], spec["seq"]
    gen = torch.Generator(device="cpu").manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (B, T + 1), generator=gen).to(dev)
    peak = {"current": 0, "driver": 0}

    def sample():
        torch.mps.synchronize()
        peak["current"] = max(peak["current"], torch.mps.current_allocated_memory())
        peak["driver"] = max(peak["driver"], torch.mps.driver_allocated_memory())

    def forward_hidden(x_ids):
        x = model.embed(x_ids)
        for block in model.blocks:
            if spec["ckpt"]:
                x = checkpoint(block, x, model.rope_cos, model.rope_sin, use_reentrant=False)
            else:
                x = block(x, model.rope_cos, model.rope_sin)
        return model.norm(x)

    def step():
        h = forward_hidden(ids[:, :-1])
        logits = model.lm_logits(h)
        loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1))
        sample()
        loss.backward()
        sample()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sample()
        return float(loss)

    for _ in range(spec["warmup"]):
        step()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    losses = [step() for _ in range(spec["steps"])]
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    train_tps = B * T * spec["steps"] / dt

    # Forward-only (no grad) throughput on the same batch.
    model.eval()
    with torch.no_grad():
        model.lm_logits(forward_hidden(ids[:, :-1]))
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(spec["steps"]):
            model.lm_logits(forward_hidden(ids[:, :-1]))
        torch.mps.synchronize()
        infer_tps = B * T * spec["steps"] / (time.perf_counter() - t0)
    sample()
    g = lambda b: round(b / GiB, 3)
    return {"status": "RAN", "train_tokens_per_s": round(train_tps, 1),
            "forward_tokens_per_s": round(infer_tps, 1),
            "step_seconds": round(dt / spec["steps"], 4), "losses": [round(l, 4) for l in losses],
            "mps_current_allocated_peak_gib": g(peak["current"]),
            "mps_driver_allocated_peak_gib": g(peak["driver"]),
            "torch": torch.__version__}


def run_gated(preset: str, cfg, precision: str, ckpt: bool, args) -> dict:
    est = estimate(cfg, precision, args.batch, args.seq, ckpt)
    rec = {"preset": preset, "precision": precision, "activation_checkpointing": ckpt, "estimate": est}
    safety = int(args.safety_gib * GiB)
    avail = available_bytes()
    # Other jobs share this machine: optionally wait for memory to free up, but only
    # when the estimate could fit at all under current physical RAM.
    waitable = not args.estimate_only and est["train_peak_bytes"] <= args.wait_max_gib * GiB
    deadline = time.time() + (args.wait_seconds if waitable else 0)
    while est["train_peak_bytes"] >= avail - safety and time.time() < deadline:
        time.sleep(10)
        avail = available_bytes()
    budget = avail - safety
    rec["available_gib_at_check"] = round(avail / GiB, 3)
    rec["budget_gib"] = round(budget / GiB, 3)
    if args.estimate_only:
        rec["status"] = "ESTIMATE_ONLY"
        return rec
    if est["train_peak_bytes"] >= budget:
        rec["status"] = "SKIPPED"
        rec["reason"] = (f"estimated peak {est['train_gib']['peak']} GiB >= budget "
                         f"{rec['budget_gib']} GiB (available {rec['available_gib_at_check']} - "
                         f"{args.safety_gib} safety)")
        return rec
    # Hard cap on the MPS allocator: never grow past the budget.
    import torch  # parent only needs recommended_max_memory
    rec_max = torch.mps.recommended_max_memory() if torch.backends.mps.is_available() else 0
    if not rec_max:
        rec["status"] = "SKIPPED"
        rec["reason"] = "MPS unavailable"
        return rec
    ratio = max(0.05, min(1.0, budget / rec_max))
    env = dict(os.environ, PYTORCH_MPS_HIGH_WATERMARK_RATIO=f"{ratio:.4f}",
               # Low watermark well below the cap makes the allocator release cached
               # buffers early; at 0.9x the cache alone tripped the cap on sys1-tiny.
               PYTORCH_MPS_LOW_WATERMARK_RATIO=f"{ratio * 0.4:.4f}")
    spec = {"preset": preset, "precision": precision, "ckpt": ckpt, "batch": args.batch,
            "seq": args.seq, "steps": args.steps, "warmup": args.warmup}
    # Final re-check right before the child allocates.
    avail2 = available_bytes()
    if est["train_peak_bytes"] >= avail2 - safety:
        rec["status"] = "SKIPPED"
        rec["reason"] = f"available dropped to {avail2 / GiB:.2f} GiB before launch"
        return rec
    rec["mps_high_watermark_ratio"] = round(ratio, 4)
    rec["mps_low_watermark_ratio"] = round(ratio * 0.4, 4)
    try:
        p = subprocess.run([sys.executable, "-m", "sys1.model.bench", "--child", json.dumps(spec)],
                           capture_output=True, text=True, env=env, timeout=args.timeout,
                           cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    except subprocess.TimeoutExpired:
        rec["status"] = "TIMEOUT"
        return rec
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode == 0 and lines:
        rec.update(json.loads(lines[-1]))
        meas = rec.get("mps_driver_allocated_peak_gib", 0)
        rec["estimate_vs_driver_peak_ratio"] = round(est["train_gib"]["peak"] / meas, 3) if meas else None
    else:
        rec["status"] = "FAILED"
        rec["stderr_tail"] = p.stderr[-1500:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", nargs="*", default=None, help="default: every preset in PRESETS")
    ap.add_argument("--precisions", nargs="*", default=list(PRECISIONS), choices=PRECISIONS)
    ap.add_argument("--ckpt", choices=["off", "on", "both"], default="both")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--safety-gib", type=float, default=4.0)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--wait-seconds", type=int, default=0,
                    help="poll vm_stat up to this long for a run whose estimate does not fit yet")
    ap.add_argument("--wait-max-gib", type=float, default=3.0,
                    help="only wait for runs whose estimated peak is at most this")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--out", default="runs/bench/report.json")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        print(json.dumps(_child(json.loads(args.child))), flush=True)
        return

    from sys1.model.model import PRESETS, param_count
    names = args.presets or list(PRESETS)
    ckpts = {"off": [False], "on": [True], "both": [False, True]}[args.ckpt]
    report = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "batch": args.batch, "seq": args.seq,
              "steps": args.steps, "safety_gib": args.safety_gib,
              "available_gib_start": round(available_bytes() / GiB, 3), "presets": {}}
    for name in names:
        cfg = PRESETS[name]
        entry = {"config": asdict(cfg), "params": param_count(cfg), "runs": []}
        for prec in args.precisions:
            for ck in ckpts:
                rec = run_gated(name, cfg, prec, ck, args)
                entry["runs"].append(rec)
                extra = (f"train {rec['train_tokens_per_s']} tok/s, driver peak "
                         f"{rec['mps_driver_allocated_peak_gib']} GiB") if rec["status"] == "RAN" \
                    else rec.get("reason", rec.get("stderr_tail", ""))[-200:]
                print(f"{name:22s} {prec:4s} ckpt={'on ' if ck else 'off'} est "
                      f"{rec['estimate']['train_gib']['peak']:7.2f} GiB  {rec['status']:8s} {extra}",
                      flush=True)
        report["presets"][name] = entry
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
