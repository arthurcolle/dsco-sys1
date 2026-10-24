from __future__ import annotations

import copy
import importlib.util
import json
import math
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from sys1.model import pretrain as pt
from sys1.model.model import PRESETS, Sys1Config, Sys1Model, param_count
from sys1.model.validation import ValidationGuard


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tokenizer():
    tokens = ["[UNK]", "[EOS]", "alpha", "beta", "gamma", "delta", "held", "only"]
    tok = Tokenizer(models.WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["[UNK]", "[EOS]"])
    return tok


def corpus(root):
    for name, text in [("en", "alpha beta gamma"), ("code", "gamma delta alpha")]:
        d = root / name
        d.mkdir(parents=True)
        (d / "train.txt").write_text((text + "\n") * 128)
        (d / "heldout.txt").write_text("held only\n" * 128)


def data_args(root):
    return SimpleNamespace(corpus=str(root), episodes=str(root / "missing.jsonl"),
                           mix="en=0.5,code=0.5", seq=8, eval_seqs=4,
                           txt_sep="<|endoftext|>", seed=1234)


def tiny_cfg():
    return Sys1Config(vocab_size=8, d_model=16, n_layers=2, n_heads=2,
                      n_kv_heads=1, ffn_dim=32, max_seq=32)


def test_causal_mask_blocks_future_tokens():
    torch.manual_seed(7)
    model = Sys1Model(tiny_cfg()).eval()
    a = torch.tensor([[2, 3, 4, 5, 2, 3]])
    b = a.clone()
    b[:, 4:] = torch.tensor([6, 7])
    with torch.no_grad():
        x = model.lm_logits(model.hidden(a))
        y = model.lm_logits(model.hidden(b))
    torch.testing.assert_close(x[:, :4], y[:, :4], rtol=0, atol=0)
    assert not torch.equal(x[:, 4:], y[:, 4:])


def test_next_token_objective_learns_without_decision_heads():
    torch.manual_seed(3)
    model = Sys1Model(tiny_cfg())
    x = torch.tensor([[2, 3, 4, 2, 3, 4, 2, 3, 4]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = pt.lm_loss(model, x).item()
    for _ in range(64):
        optimizer.zero_grad(set_to_none=True)
        loss = pt.lm_loss(model, x)
        loss.backward()
        optimizer.step()
    final = pt.lm_loss(model, x).item()
    assert math.isfinite(final)
    assert final < initial * 0.25
    assert model.choice_q.weight.grad is None
    assert model.noul.weight.grad is None
    assert model.value.weight.grad is None


def test_training_excludes_holdout_and_evaluation_is_fixed(tmp_path):
    corpus(tmp_path)
    args, tok = data_args(tmp_path), tokenizer()
    mix, held, _ = pt.build_data(args, tok, 1)
    _, other, _ = pt.build_data(args, tok, 1)
    batch = mix.batch(16)
    assert not bool(torch.isin(batch, torch.tensor([6, 7])).any())
    assert set(held) == {"en", "code"}
    for name in held:
        assert torch.equal(held[name], other[name])
        assert bool(torch.isin(held[name], torch.tensor([6, 7])).any())


def test_validation_fingerprint_binds_tokens_and_tokenizer(tmp_path):
    tok = tmp_path / "tokenizer.json"
    tokenizer().save(str(tok))
    held = {"en": torch.tensor([[2, 3, 4]])}
    first = pt.validation_fingerprint(held, str(tok))
    assert first == pt.validation_fingerprint(held, str(tok))
    held["en"][0, 0] = 5
    assert first != pt.validation_fingerprint(held, str(tok))
    second = pt.validation_fingerprint(held, str(tok))
    tok.write_text(tok.read_text() + "\n")
    assert second != pt.validation_fingerprint(held, str(tok))


def test_mixture_restores_the_exact_next_batches(tmp_path):
    corpus(tmp_path)
    args, tok = data_args(tmp_path), tokenizer()
    original, _, _ = pt.build_data(args, tok, 1)
    original.batch(5)
    state = copy.deepcopy(original.state())
    expected = [original.batch(7) for _ in range(3)]
    restored, _, _ = pt.build_data(args, tok, 1)
    restored.load_state(state)
    for batch in expected:
        assert torch.equal(batch, restored.batch(7))
    assert original.drawn == restored.drawn


def test_full_training_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    data = tmp_path / "corpus"
    corpus(data)
    tok = tmp_path / "tokenizer.json"
    tokenizer().save(str(tok))
    monkeypatch.setitem(PRESETS, "test-causal", tiny_cfg())
    monkeypatch.setattr(pt, "RUNS", tmp_path / "runs")
    common = ["--preset", "test-causal", "--tokenizer", str(tok), "--corpus", str(data),
              "--mix", "en=0.5,code=0.5", "--seq", "8", "--micro-batch", "2",
              "--accum", "2", "--steps", "6", "--warmup", "2", "--lr", "0.01",
              "--device", "cpu", "--precision", "fp32", "--eval-seqs", "4",
              "--eval-every", "2", "--ckpt-every", "2", "--keep", "2", "--log-every", "1"]
    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        pt.main(["--name", "full", *common])
        pt.main(["--name", "resumed", *common, "--stop-after", "3"])
        pt.main(["--name", "resumed", *common])
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    run_root = tmp_path / "runs/pretrain"
    final_states = []
    for name in ("full", "resumed"):
        out = run_root / name
        ckpts = sorted((out / "checkpoints").glob("step_*.pt"))
        assert len(ckpts) == 2
        final_states.append(torch.load(ckpts[-1], map_location="cpu", weights_only=False))
        events = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
        assert events[-1]["event"] == "done"
        assert events[-1]["step"] == 6
        assert all(math.isfinite(r["loss"]) for r in events if r["event"] == "train")
    a, b = final_states
    assert a["step"] == b["step"] == 6
    assert a["data"]["drawn"] == b["data"]["drawn"]
    assert a["validation"] == b["validation"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)


def test_checkpoint_budget_retains_old_best_and_latest(tmp_path):
    data = tmp_path / "corpus"
    corpus(data)
    mix, _, _ = pt.build_data(data_args(data), tokenizer(), 1)
    model = Sys1Model(tiny_cfg())
    optimizer = torch.optim.AdamW(model.parameters())
    guard = ValidationGuard({"en": 0.5, "code": 0.5})
    out = tmp_path / "run"
    for step, loss in [(2, 2.0), (4, 2.1), (6, 2.2)]:
        decision = guard.observe(step, {"en": loss, "code": loss})
        pt.save_checkpoint(out, step, model, optimizer, mix, SimpleNamespace(), "fp32", 2,
                           validation=guard, is_best=decision["improved"])
    ckdir = out / "checkpoints"
    assert sorted(p.name for p in ckdir.glob("step_*.pt")) == ["step_00000002.pt", "step_00000006.pt"]
    assert (ckdir / "best").read_text() == "step_00000002.pt"
    assert (ckdir / "latest").read_text() == "step_00000006.pt"
    state = torch.load(ckdir / "step_00000006.pt", map_location="cpu", weights_only=False)
    assert state["validation"]["bad_evaluations"] == 2


def test_training_stops_on_validation_regression_and_keeps_best(tmp_path, monkeypatch):
    data = tmp_path / "corpus"
    corpus(data)
    tok = tmp_path / "tokenizer.json"
    tokenizer().save(str(tok))
    monkeypatch.setitem(PRESETS, "test-causal", tiny_cfg())
    monkeypatch.setattr(pt, "RUNS", tmp_path / "runs")
    values = iter([2.0, 2.2, 2.4])
    def worsening(*_):
        value = next(values)
        return {"en": value, "code": value}
    monkeypatch.setattr(pt, "evaluate", worsening)
    argv = ["--name", "guarded", "--preset", "test-causal", "--tokenizer", str(tok),
            "--corpus", str(data), "--mix", "en=0.5,code=0.5", "--seq", "8",
            "--micro-batch", "2", "--steps", "20", "--device", "cpu", "--precision", "fp32",
            "--eval-every", "2", "--eval-seqs", "4", "--ckpt-every", "2", "--keep", "2",
            "--early-stop-patience", "2", "--log-every", "1"]
    out = tmp_path / "runs/pretrain/guarded"
    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        pt.main(argv)
        saved = (out / "checkpoints/step_00000006.pt").read_bytes()
        pt.main(argv)
        assert saved == (out / "checkpoints/step_00000006.pt").read_bytes()
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "early_stopped"
    assert rows[-1]["step"] == 6
    assert rows[-1]["best_step"] == 2
    assert rows[-1]["resumed_terminal_guard"] is True
    assert sum(row["event"] == "train" for row in rows) == 6
    assert not any(r["event"] == "done" for r in rows)
    assert (out / "checkpoints/best").read_text() == "step_00000002.pt"
    assert (out / "checkpoints/step_00000002.pt").is_file()


def test_final_step_validation_selects_a_new_best(tmp_path, monkeypatch):
    data = tmp_path / "corpus"
    corpus(data)
    tok = tmp_path / "tokenizer.json"
    tokenizer().save(str(tok))
    monkeypatch.setitem(PRESETS, "test-causal", tiny_cfg())
    monkeypatch.setattr(pt, "RUNS", tmp_path / "runs")
    values = iter([2.0, 1.0])
    def improving(*_):
        value = next(values)
        return {"en": value, "code": value}
    monkeypatch.setattr(pt, "evaluate", improving)
    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        pt.main(["--name", "final-best", "--preset", "test-causal", "--tokenizer", str(tok),
                 "--corpus", str(data), "--mix", "en=0.5,code=0.5", "--seq", "8",
                 "--micro-batch", "2", "--steps", "5", "--device", "cpu", "--precision", "fp32",
                 "--eval-every", "4", "--eval-seqs", "4", "--ckpt-every", "4", "--keep", "2"])
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    out = tmp_path / "runs/pretrain/final-best"
    assert (out / "checkpoints/best").read_text() == "step_00000005.pt"
    saved = torch.load(out / "checkpoints/step_00000005.pt", map_location="cpu", weights_only=False)
    assert saved["validation"]["best_step"] == 5
    assert saved["validation"]["best_score"] == 1.0


def test_live_100k_presets_match_recorded_parameter_counts():
    assert param_count(PRESETS["sys1-999m-v100k"])["total"] == 998_562_562
    assert param_count(PRESETS["sys1-666m-v100k"])["total"] == 669_340_802


def test_run_preparation_is_private_and_never_overwrites(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_pretrain_run.py"
    spec = importlib.util.spec_from_file_location("prepare_pretrain_run", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "sys1/model").mkdir(parents=True)
    for name in ("model.py", "pretrain.py"):
        (tmp_path / "sys1/model" / name).write_text("# fixture\n")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "corpus/en").mkdir(parents=True)
    (tmp_path / "corpus/en/MANIFEST.json").write_text("{}")
    recipe = {"schema": "sys1-pretrain-recipe-v1", "objective": "causal-next-token-cross-entropy",
              "training": {"preset": "sys1-tiny", "tokenizer": "tokenizer.json", "corpus": "corpus",
                           "mix": "en=1", "seq": 128, "micro-batch": 2, "accum": 4, "steps": 10}}
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    result = module.prepare(path, "new-run", tmp_path)
    assert result["planned_tokens"] == 10240
    assert result["launch_performed"] is False
    assert Path(result["manifest"]).stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        module.prepare(path, "new-run", tmp_path)
    with pytest.raises(ValueError):
        module.prepare(path, "../escape", tmp_path)
