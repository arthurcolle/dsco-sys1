"""Deep-and-thin decoder with typed decision heads ("System One").

Body: pre-norm decoder, RMSNorm, rotary positions, grouped-query attention,
SwiGLU MLP, tied input/output embeddings.

Heads, all read at a [DECIDE] position:
  choice  pointer scores over option spans marked by [OPT] tokens in the input,
          so one head serves any option set (tools, plans, routes, answers)
  noul    P(yes) for a binary question, e.g. "will the chosen action succeed"
  value   P(goal reached) for the episode from this state
The LM head (tied embeddings) is used for pretraining and distillation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Sys1Config:
    vocab_size: int = 32768
    d_model: int = 1152
    n_layers: int = 44
    n_heads: int = 18
    n_kv_heads: int = 6
    ffn_dim: int = 3072
    max_seq: int = 4096
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


PRESETS = {
    # ~666M deep-thin target (see param_count); 44 layers x 1152 wide.
    "sys1-666m": Sys1Config(),
    # Pipeline smoke test on a laptop GPU.
    "sys1-tiny": Sys1Config(d_model=384, n_layers=12, n_heads=6, n_kv_heads=2, ffn_dim=1024, max_seq=2048),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(cfg: Sys1Config, device):
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=device).float() / cfg.head_dim))
    t = torch.arange(cfg.max_seq, device=device).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    T = x.shape[-2]
    cos, sin = cos[:T][None, None], sin[:T][None, None]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: Sys1Config):
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * hd, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.n_kv_heads * hd, bias=False)
        self.o = nn.Linear(cfg.n_heads * hd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, attn_mask=None):
        B, T, _ = x.shape
        c = self.cfg
        q = self.q(x).view(B, T, c.n_heads, c.head_dim).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, c.n_kv_heads, c.head_dim).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = c.n_heads // c.n_kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: Sys1Config):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, cfg: Sys1Config, layer_index: int):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model, cfg.norm_eps), RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn, self.mlp = Attention(cfg), SwiGLU(cfg)
        # LayerNorm Scaling (arXiv:2502.05795): pre-norm output variance grows
        # with depth and leaves deep layers underused; scale by 1/sqrt(l).
        self.ln_scale = 1.0 / math.sqrt(layer_index + 1)

    def forward(self, x, cos, sin, attn_mask=None):
        x = x + self.attn(self.n1(x) * self.ln_scale, cos, sin, attn_mask)
        return x + self.mlp(self.n2(x) * self.ln_scale)


class Sys1Model(nn.Module):
    def __init__(self, cfg: Sys1Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        d = cfg.d_model
        # Typed decision heads (small; the body does the work).
        self.choice_q = nn.Linear(d, d, bias=False)   # query from [DECIDE]
        self.choice_k = nn.Linear(d, d, bias=False)   # keys from [OPT] positions
        self.noul = nn.Linear(d, 1)
        self.value = nn.Linear(d, 1)
        cos, sin = rope_cache(cfg, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # Depth-scaled init on residual projections keeps deep stacks stable.
        for block in self.blocks:
            for proj in (block.attn.o, block.mlp.down):
                nn.init.normal_(proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def hidden(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        return self.norm(x)

    def lm_logits(self, h):
        return h @ self.embed.weight.T  # tied

    def decide(self, h, decide_pos, opt_pos, opt_mask):
        """h: (B,T,D); decide_pos: (B,); opt_pos: (B,K) indices; opt_mask: (B,K) bool.
        Returns choice logits (B,K) with -inf on padding, noul and value logits (B,)."""
        b = torch.arange(h.shape[0], device=h.device)
        hd = h[b, decide_pos]                                   # (B,D)
        ho = torch.gather(h, 1, opt_pos[..., None].expand(-1, -1, h.shape[-1]))  # (B,K,D)
        scores = torch.einsum("bd,bkd->bk", self.choice_q(hd), self.choice_k(ho)) / math.sqrt(h.shape[-1])
        scores = scores.masked_fill(~opt_mask, float("-inf"))
        return scores, self.noul(hd).squeeze(-1), self.value(hd).squeeze(-1)


def param_count(cfg: Sys1Config) -> dict:
    """Exact arithmetic, matching the module definitions above."""
    d, hd = cfg.d_model, cfg.head_dim
    attn = d * cfg.n_heads * hd + d * 2 * cfg.n_kv_heads * hd + cfg.n_heads * hd * d
    mlp = d * 2 * cfg.ffn_dim + cfg.ffn_dim * d
    per_layer = attn + mlp + 2 * d
    embed = cfg.vocab_size * d
    heads = 2 * d * d + 2 * (d + 1)
    total = embed + cfg.n_layers * per_layer + d + heads
    return {"embedding": embed, "per_layer": per_layer, "layers": cfg.n_layers * per_layer,
            "heads": heads, "total": total, "non_embedding": total - embed}


if __name__ == "__main__":
    import json
    for name, cfg in PRESETS.items():
        counts = param_count(cfg)
        with torch.device("meta"):
            model = Sys1Model(cfg)
        actual = sum(p.numel() for p in model.parameters())
        assert actual == counts["total"], (name, actual, counts["total"])
        print(name, json.dumps({**{k: f"{v/1e6:.1f}M" for k, v in counts.items()}, "config": asdict(cfg)}))
