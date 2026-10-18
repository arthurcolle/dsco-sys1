"""Semantic initialisation of reserved-token embeddings.

    python -m sys1.model.embed_init --selftest [--ckpt runs/x/model.pt]

Reserved op-code and compositional tokens start as random rows, so the model
has to learn what <|op:ctx.mask_batch|> means from scratch. This module seeds
each reserved row from the BPE embeddings of text that already describes it:

  op-code        IDF-weighted mean of BPE embeddings of its mnemonic words,
                 `desc` and `when` cue (sys1/vocab/opcodes.json)
  compositional  <|fam:x|> / <|verb:x|> / <|obj:x|>: mean of (a) the BPE
                 embedding of its name words and (b) the text vectors of every
                 fused op that decomposes into it
  fused v+o op   (1 - alpha) * text vector + alpha * (fam + verb + obj) sum,
                 so fused ops sit near their compositional parts

Every written row is rescaled to the median norm of the BPE rows, so the
reserved region matches the scale the rest of the table was trained at.
Tokens missing from the tokenizer, or whose id is outside the model's
embedding table, are skipped and counted in the returned report.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import torch

from sys1.paths import COMPOSITIONAL_JSON, DATA, OPCODES_JSON

_WORD_SPLIT = re.compile(r"[._\-]+")


class _Tok:
    """Minimal adapter over HF `tokenizers.Tokenizer`, transformers tokenizers,
    or any object exposing encode(text) and token_to_id(token)."""

    def __init__(self, tokenizer):
        self.t = tokenizer

    def encode(self, text: str) -> list[int]:
        t = self.t
        try:
            out = t.encode(text, add_special_tokens=False)
        except TypeError:
            out = t.encode(text)
        return list(getattr(out, "ids", out))

    def token_to_id(self, token: str):
        t = self.t
        if hasattr(t, "token_to_id"):
            return t.token_to_id(token)
        if hasattr(t, "convert_tokens_to_ids"):
            i = t.convert_tokens_to_ids(token)
            return None if i is None or i == getattr(t, "unk_token_id", object()) else i
        return t.get_vocab().get(token)

    def special_ids(self) -> set[int]:
        t = self.t
        ids = set()
        if hasattr(t, "get_added_tokens_decoder"):          # tokenizers >= 0.15
            ids |= set(t.get_added_tokens_decoder())
        if hasattr(t, "all_special_ids"):                   # transformers
            ids |= set(t.all_special_ids)
        if hasattr(t, "added_tokens_decoder"):
            ids |= set(t.added_tokens_decoder)
        return ids


def _load(p):
    return json.loads(Path(p).read_text()) if not isinstance(p, dict) else p


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-12)


def _name_words(name: str) -> str:
    return " ".join(w for w in _WORD_SPLIT.split(name.lower()) if w)


@torch.no_grad()
def init_reserved_embeddings(model, tokenizer, opcodes_json=OPCODES_JSON, compositional_json=COMPOSITIONAL_JSON,
                             alpha: float = 0.3, name_weight: float = 0.5, include_mnemonic: bool = True,
                             idf: bool = True, target_norm: float | None = None) -> dict:
    """Overwrite reserved op-code / compositional rows of model.embed in place.

    alpha        blend weight of the (fam + verb + obj) compositional sum into
                 fused verb+object op-codes (0 = text only, 1 = parts only)
    name_weight  weight of the name-word vector vs the containing-op vector
                 for compositional tokens
    idf          weight BPE tokens by inverse document frequency over all op
                 texts so filler words ("the", "a", ".") do not dominate
    target_norm  row norm to write; default = median norm of BPE rows
    Returns a report dict (counts, norms, skipped tokens).
    """
    tok = _Tok(tokenizer)
    ops = _load(opcodes_json)["ops"]
    comp = _load(compositional_json)
    W = model.embed.weight
    V = W.shape[0]
    dev, dt = W.device, W.dtype

    def tid(t):
        i = tok.token_to_id(t)
        return i if i is not None and 0 <= i < V else None

    reserved = {t for t in comp["tokens"]} | {op["token"] for op in ops}
    special = tok.special_ids() | {i for i in (tid(t) for t in reserved) if i is not None}
    bpe_mask = torch.ones(V, dtype=torch.bool, device=dev)
    bpe_mask[torch.tensor(sorted(i for i in special if i < V), dtype=torch.long, device=dev)] = False
    Wf = W.detach().float()
    norm = float(target_norm) if target_norm else float(Wf[bpe_mask].norm(dim=-1).median())

    def ids_of(text):
        return [i for i in tok.encode(text) if 0 <= i < V and bpe_mask[i]]

    # Op texts and IDF over them.
    texts = {}
    for op in ops:
        parts = [_name_words(op["mnemonic"])] if include_mnemonic else []
        parts += [op.get("desc") or "", op.get("when") or ""]
        texts[op["token"]] = ids_of(" ".join(p for p in parts if p))
    df = Counter(i for ids in texts.values() for i in set(ids))
    n_docs = max(len(texts), 1)
    w_of = (lambda i: math.log((1 + n_docs) / (1 + df.get(i, 0))) + 1.0) if idf else (lambda i: 1.0)

    def bag(ids):
        if not ids:
            return None
        w = torch.tensor([w_of(i) for i in ids], device=dev)
        return _unit((Wf[torch.tensor(ids, device=dev)] * w[:, None]).sum(0) / w.sum())

    text_vec = {t: v for t, v in ((t, bag(ids)) for t, ids in texts.items()) if v is not None}

    # Compositional tokens: name words + mean of containing ops' text vectors.
    members: dict[str, list[str]] = {t: [] for t in comp["tokens"]}
    for op_tok, parts in comp["decomposition"].items():
        for p in parts:
            members.setdefault(p, []).append(op_tok)
    comp_vec = {}
    for t in comp["tokens"]:
        name = t[2:-2].split(":", 1)[-1]                    # <|verb:dry_run|> -> dry_run
        nv = bag(ids_of(_name_words(name)))
        mv = [text_vec[o] for o in members.get(t, []) if o in text_vec]
        mv = _unit(torch.stack(mv).mean(0)) if mv else None
        if nv is not None and mv is not None:
            comp_vec[t] = _unit(name_weight * nv + (1 - name_weight) * mv)
        elif nv is not None or mv is not None:
            comp_vec[t] = nv if nv is not None else mv

    # Fused verb+object ops: blend in the sum of their parts.
    op_vec = {}
    for t, v in text_vec.items():
        parts = comp["decomposition"].get(t)
        if parts and alpha > 0 and all(p in comp_vec for p in parts):
            v = _unit((1 - alpha) * v + alpha * _unit(sum(comp_vec[p] for p in parts)))
        op_vec[t] = v

    written, skipped = {"op": 0, "op_blended": 0, "comp": 0}, []
    for group, vecs in (("op", op_vec), ("comp", comp_vec)):
        for t, v in vecs.items():
            i = tid(t)
            if i is None:
                skipped.append(t)
                continue
            W[i] = (v * norm).to(dt)
            written[group] += 1
            if group == "op" and t in comp["decomposition"] and alpha > 0:
                written["op_blended"] += 1
    missing_text = [op["token"] for op in ops if op["token"] not in text_vec]
    skipped += [t for t in comp["tokens"] if t not in comp_vec]
    return {"written": written, "skipped": len(skipped), "skipped_sample": skipped[:10],
            "no_text": len(missing_text), "target_norm": norm, "alpha": alpha, "idf": idf}


# --------------------------------------------------------------------------- self-test

def _selftest(ckpt: str | None, alpha: float) -> int:
    from tokenizers import Tokenizer
    from sys1.model.model import PRESETS, Sys1Config, Sys1Model

    torch.manual_seed(0)
    tok = Tokenizer.from_file(str(DATA / "decisions" / "tokenizer.json"))
    ops = json.loads(OPCODES_JSON.read_text())["ops"]
    comp = json.loads(COMPOSITIONAL_JSON.read_text())
    base_vocab = tok.get_vocab_size()
    want = [op["token"] for op in ops] + comp["tokens"]
    missing = [t for t in want if tok.token_to_id(t) is None]
    tok.add_special_tokens(missing)                          # in memory only
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        cfg = Sys1Config(**state["config"])
    else:
        cfg = Sys1Config(**{**PRESETS["sys1-tiny"].__dict__})
    cfg.vocab_size = max(cfg.vocab_size, tok.get_vocab_size())
    model = Sys1Model(cfg)
    if ckpt:
        sd = state["model"]
        e = sd["embed.weight"]
        sd["embed.weight"] = torch.cat([e, model.embed.weight[e.shape[0]:].detach()]) if e.shape[0] < cfg.vocab_size else e
        model.load_state_dict(sd, strict=False)
    model.eval()

    report = init_reserved_embeddings(model, tok, OPCODES_JSON, COMPOSITIONAL_JSON, alpha=alpha)
    W = model.embed.weight.detach().float()
    print(json.dumps({"base_vocab": base_vocab, "added_in_memory": len(missing), "model_vocab": cfg.vocab_size,
                      "ckpt": ckpt, **report}))

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    check(torch.isfinite(W).all().item(), "embedding table has no NaN/Inf")

    op_ids = {op["mnemonic"]: tok.token_to_id(op["token"]) for op in ops}
    res_ids = sorted({tok.token_to_id(t) for t in want})
    res_set = set(res_ids)
    bpe_ids = [i for i in range(base_vocab) if i not in res_set and i not in tok.get_added_tokens_decoder()]
    bpe_norm = W[bpe_ids].norm(dim=-1)
    res_norm = W[res_ids].norm(dim=-1)
    ratio = (res_norm.mean() / bpe_norm.mean()).item()
    print(f"     norms: bpe mean {bpe_norm.mean():.4f} sd {bpe_norm.std():.4f} | reserved mean "
          f"{res_norm.mean():.4f} sd {res_norm.std():.4f} | ratio {ratio:.3f}")
    check(0.9 <= ratio <= 1.1, "reserved norms match BPE embedding scale (ratio in [0.9, 1.1])")

    # Nearest neighbours among op-codes (cosine).
    mn = [op["mnemonic"] for op in ops]
    E = torch.nn.functional.normalize(W[torch.tensor([op_ids[m] for m in mn])], dim=-1)
    S = E @ E.T
    S.fill_diagonal_(-2)
    idx = {m: k for k, m in enumerate(mn)}

    def nn_of(m, k=6):
        return [(mn[j], round(S[idx[m], j].item(), 3)) for j in S[idx[m]].topk(k).indices.tolist()]

    probes = ["CTX.MASK_BATCH", "CTX.UNMASK.OBSERVATION_BLOCK", "ACT.RESULT_ERR", "CTRL.FINAL"]
    for m in probes:
        if m in idx:
            print(f"     NN {m}: {nn_of(m)}")
    a, b, c = idx["CTX.MASK_BATCH"], idx["CTX.UNMASK.OBSERVATION_BLOCK"], idx["ACT.RESULT_ERR"]
    print(f"     cos(MASK_BATCH, UNMASK.OBS_BLOCK)={S[a, b]:.3f}  cos(MASK_BATCH, RESULT_ERR)={S[a, c]:.3f}  "
          f"cos(UNMASK.OBS_BLOCK, RESULT_ERR)={S[b, c]:.3f}")
    check(S[a, b] > S[a, c] and S[a, b] > S[b, c], "masking ops are closer to each other than to ACT.RESULT_ERR")
    top_mask = [x for x, _ in nn_of("CTX.MASK_BATCH", 5)]
    check(sum(x.startswith("CTX.") for x in top_mask) >= 3, "CTX.MASK_BATCH top-5 neighbours are mostly CTX.*")
    top_err = [x for x, _ in nn_of("ACT.RESULT_ERR", 5)]
    check(any(re.search(r"ERR|FAIL|RESULT", x) for x in top_err), "ACT.RESULT_ERR neighbours include error/result ops")

    # Global structure: same-family rate of top-5 neighbours vs chance.
    fam = [m.split(".")[0] for m in mn]
    top5 = S.topk(5, dim=-1).indices
    same = sum(fam[j] == fam[i] for i in range(len(mn)) for j in top5[i].tolist()) / (5 * len(mn))
    cnt = Counter(fam)
    chance = sum((v / len(mn)) * ((v - 1) / (len(mn) - 1)) for v in cnt.values())
    print(f"     same-family rate of top-5 NN: {same:.3f} (chance {chance:.3f})")
    check(same > 5 * chance, "op-code neighbourhoods respect families (> 5x chance)")

    # Fused ops sit near their compositional parts; random rows would not.
    sims = []
    for op_tok, parts in comp["decomposition"].items():
        v = W[tok.token_to_id(op_tok)]
        s = sum(W[tok.token_to_id(p)] for p in parts)
        sims.append(torch.nn.functional.cosine_similarity(v, s, dim=0).item())
    mean_sim = sum(sims) / len(sims)
    print(f"     mean cos(fused op, fam+verb+obj) over {len(sims)} ops: {mean_sim:.3f}")
    check(mean_sim > 0.5, "fused ops align with their compositional sum")

    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", default=None, help="optional trained Sys1 checkpoint (model.pt) for BPE rows")
    ap.add_argument("--alpha", type=float, default=0.3)
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest(args.ckpt, args.alpha)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
