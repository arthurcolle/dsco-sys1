"""System One in MLX: causal GQA, adjacent-pair RoPE, SwiGLU and typed heads."""
from __future__ import annotations

from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass(frozen=True)
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

    def __post_init__(self):
        fields = (self.vocab_size, self.d_model, self.n_layers, self.n_heads, self.n_kv_heads, self.ffn_dim, self.max_seq)
        if any(type(value) is not int or value < 1 for value in fields):
            raise ValueError("model dimensions must be positive integers")
        if self.d_model % self.n_heads or self.n_heads % self.n_kv_heads or self.head_dim % 2:
            raise ValueError("invalid grouped-query or rotary dimensions")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0 or not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("invalid rotary base or normalization epsilon")
        if self.dropout != 0:
            raise ValueError("System One architecture parity requires dropout=0")

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


PRESETS = {
    "sys1-666m": Sys1Config(),
    "sys1-999m": Sys1Config(d_model=1280, n_layers=54, n_heads=20, n_kv_heads=5, ffn_dim=3456),
    "sys1-999m-v100k": Sys1Config(vocab_size=100352, d_model=1280, n_layers=46, n_heads=20, n_kv_heads=5, ffn_dim=3840),
    "sys1-666m-v100k": Sys1Config(vocab_size=100352, d_model=1152, n_layers=47, n_heads=18, n_kv_heads=6, ffn_dim=2368),
    "sys1-tiny": Sys1Config(d_model=384, n_layers=12, n_heads=6, n_kv_heads=2, ffn_dim=1024, max_seq=2048),
    "sys1-mlx-dev": Sys1Config(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=192, max_seq=512),
    "sys1-mlx-small": Sys1Config(d_model=256, n_layers=8, n_heads=4, n_kv_heads=2, ffn_dim=768, max_seq=2048),
}


def param_count(cfg):
    d, hd = cfg.d_model, cfg.head_dim
    attention = d*cfg.n_heads*hd + d*2*cfg.n_kv_heads*hd + cfg.n_heads*hd*d
    per_layer = attention + 3*d*cfg.ffn_dim + 2*d
    embedding, heads = cfg.vocab_size*d, 2*d*d+2*(d+1)
    total = embedding + cfg.n_layers*per_layer + d + heads
    return {"embedding": embedding, "per_layer": per_layer, "layers": cfg.n_layers*per_layer,
            "heads": heads, "total": total, "non_embedding": total-embedding}


def _initialize(linear, std=.02):
    linear.weight = mx.random.normal(linear.weight.shape) * std
    if "bias" in linear:
        linear.bias = mx.zeros_like(linear.bias)


def _linear(layer, x):
    """Cast operands for compute, retaining FP32 master weights and gradients."""
    value = x @ layer.weight.astype(x.dtype).T
    return value + layer.bias.astype(x.dtype) if "bias" in layer else value


def _norm(layer, x, eps):
    return mx.fast.rms_norm(x,layer.weight.astype(x.dtype),eps)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q = nn.Linear(cfg.d_model, cfg.n_heads*cfg.head_dim, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2*cfg.n_kv_heads*cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_heads*cfg.head_dim, cfg.d_model, bias=False)
        self.rope = nn.RoPE(cfg.head_dim, traditional=True, base=cfg.rope_theta)

    def __call__(self, x):
        b, t, _ = x.shape
        c = self.cfg
        q = _linear(self.q,x).reshape(b,t,c.n_heads,c.head_dim).transpose(0,2,1,3)
        kv = _linear(self.kv,x).reshape(b,t,2,c.n_kv_heads,c.head_dim)
        k, v = kv[:,:,0].transpose(0,2,1,3), kv[:,:,1].transpose(0,2,1,3)
        q, k = self.rope(q), self.rope(k)
        # MLX handles compact K/V heads directly; never repeat them to N_q.
        y = mx.fast.scaled_dot_product_attention(q,k,v,scale=c.head_dim**-.5,mask="causal")
        return _linear(self.o,y.transpose(0,2,1,3).reshape(b,t,-1))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2*cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def __call__(self, x):
        gate, up = mx.split(_linear(self.gate_up,x), 2, axis=-1)
        return _linear(self.down,nn.silu(gate)*up)


class Block(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(cfg.d_model,eps=cfg.norm_eps), nn.RMSNorm(cfg.d_model,eps=cfg.norm_eps)
        self.attn, self.mlp = Attention(cfg), SwiGLU(cfg)
        self.ln_scale = 1/math.sqrt(index+1)
        self.norm_eps = cfg.norm_eps

    def __call__(self, x):
        x = x + self.attn(_norm(self.n1,x,self.norm_eps)*self.ln_scale)
        return x + self.mlp(_norm(self.n2,x,self.norm_eps)*self.ln_scale)


class Sys1Model(nn.Module):
    def __init__(self, cfg, gradient_checkpointing=False, compute_dtype=mx.float32):
        super().__init__()
        self.cfg, self.gradient_checkpointing = cfg, gradient_checkpointing
        self.compute_dtype = compute_dtype
        self.embed = nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.blocks = [Block(cfg,i) for i in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.d_model,eps=cfg.norm_eps)
        self.choice_q, self.choice_k = nn.Linear(cfg.d_model,cfg.d_model,bias=False), nn.Linear(cfg.d_model,cfg.d_model,bias=False)
        self.noul, self.value = nn.Linear(cfg.d_model,1), nn.Linear(cfg.d_model,1)
        self.embed.weight = mx.random.normal(self.embed.weight.shape)*.02
        for block in self.blocks:
            for linear in (block.attn.q,block.attn.kv,block.mlp.gate_up):
                _initialize(linear)
            for linear in (block.attn.o,block.mlp.down):
                _initialize(linear,.02/math.sqrt(2*cfg.n_layers))
        for head in (self.choice_q,self.choice_k,self.noul,self.value):
            _initialize(head)

    def hidden(self, ids):
        if ids.ndim != 2 or ids.shape[1] > self.cfg.max_seq:
            raise ValueError("input must be [batch, sequence] within max_seq")
        x = self.embed(ids).astype(self.compute_dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                def recompute(params, value, layer=block):
                    layer.update(params)
                    return layer(value)
                x = mx.checkpoint(recompute)(block.trainable_parameters(), x)
            else:
                x = block(x)
        return _norm(self.norm,x,self.cfg.norm_eps)

    def lm_logits(self, h):
        return h @ self.embed.weight.astype(h.dtype).T

    def __call__(self, ids):
        return self.lm_logits(self.hidden(ids))

    def decide(self, h, decide_pos, opt_pos, opt_mask):
        b = mx.arange(h.shape[0])
        hd = h[b,decide_pos]
        ho = h[b[:,None],opt_pos]
        scores = mx.sum(_linear(self.choice_q,hd)[:,None,:]*_linear(self.choice_k,ho),axis=-1)/math.sqrt(self.cfg.d_model)
        return mx.where(opt_mask,scores,-mx.inf), _linear(self.noul,hd).squeeze(-1), _linear(self.value,hd).squeeze(-1)

    def freeze_decision_heads(self):
        for head in (self.choice_q,self.choice_k,self.noul,self.value):
            head.freeze()

    def num_parameters(self):
        return sum(value.size for _,value in tree_flatten(self.parameters()))


def lm_loss(model, windows):
    logits = model(windows[:,:-1]).astype(mx.float32)
    per_token = nn.losses.cross_entropy(logits,windows[:,1:],reduction="none")
    per_sequence = mx.mean(per_token,axis=-1)
    return mx.mean(per_sequence), per_sequence
