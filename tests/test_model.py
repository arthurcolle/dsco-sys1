"""Model presets: analytic parameter counts and a tiny CPU forward pass."""
from __future__ import annotations

import dataclasses

import pytest
import torch

from sys1.model.model import PRESETS, Sys1Model, param_count


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_param_count_matches_meta_instantiation(name):
    cfg = PRESETS[name]
    with torch.device("meta"):
        model = Sys1Model(cfg)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == param_count(cfg)["total"]


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_shapes_consistent(name):
    cfg = PRESETS[name]
    assert cfg.d_model % cfg.n_heads == 0
    assert cfg.n_heads % cfg.n_kv_heads == 0
    assert cfg.head_dim % 2 == 0


def test_tiny_forward_shapes():
    torch.manual_seed(0)
    cfg = dataclasses.replace(PRESETS["sys1-tiny"], max_seq=64)
    model = Sys1Model(cfg).eval()
    B, T, K = 2, 16, 3
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        h = model.hidden(ids)
        assert h.shape == (B, T, cfg.d_model)
        logits = model.lm_logits(h)
        assert logits.shape == (B, T, cfg.vocab_size)
        assert torch.isfinite(logits).all()
        decide_pos = torch.tensor([T - 1, T - 2])
        opt_pos = torch.tensor([[2, 5, 8], [3, 6, 0]])
        opt_mask = torch.tensor([[True, True, True], [True, True, False]])
        scores, noul, value = model.decide(h, decide_pos, opt_pos, opt_mask)
    assert scores.shape == (B, K)
    assert noul.shape == (B,) and value.shape == (B,)
    assert torch.isneginf(scores[1, 2])
    assert torch.isfinite(scores[opt_mask]).all()
