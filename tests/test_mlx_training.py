import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from sys1.mlx import train
from sys1.mlx.model import PRESETS,Sys1Config
from sys1.mlx.data import build_data
from sys1.model.corpus_manifest import file_identity
from test_pretrain import tokenizer


@pytest.fixture(autouse=True)
def gpu():
    if not mx.metal.is_available():pytest.skip("native Apple GPU unavailable")
    previous = mx.default_device();mx.set_default_device(mx.gpu)
    yield
    mx.synchronize();mx.set_default_device(previous)


def fixture_data(root):
    root.mkdir()
    tok = root/"tokenizer.json";tokenizer().save(str(tok))
    sources = {}
    for name,text in [("en","alpha beta gamma"),("code","gamma delta alpha")]:
        sources[name] = {}
        for split in ["train","validation"]:
            p = root/f"{name}-{split}.txt";p.write_text((text+"\n")*128+split+"\n")
            sources[name][split] = [{"path":p.name,**file_identity(p)}]
    path = root/"MANIFEST.json"
    path.write_text(json.dumps({"schema":"sys1-corpus-mixture-v1","sources":sources,"objective_weights":{"en":.7,"code":.3}}))
    return path,tok


def test_sampler_roundtrip_including_sorted_json_and_validation_determinism(tmp_path):
    path,tok_path = fixture_data(tmp_path/"data")
    tok = tokenizer()
    original,held,_,identity = build_data(path,tok_path,tok,8,4,1234)
    original.batch(7)
    saved = json.loads(json.dumps(original.state(),sort_keys=True))
    a,labels = original.batch(32)
    restored,other,_,again = build_data(path,tok_path,tok,8,4,1234)
    restored.load_state(saved)
    b,other_labels = restored.batch(32)
    np.testing.assert_array_equal(np.array(a),np.array(b))
    np.testing.assert_array_equal(np.array(labels),np.array(other_labels))
    assert identity==again
    for name in held:np.testing.assert_array_equal(np.array(held[name]),np.array(other[name]))


@pytest.mark.parametrize("adaptive",[False,True])
def test_full_gpu_training_resume_restores_model_optimizer_sampler_and_counters(tmp_path,monkeypatch,adaptive):
    path,tok = fixture_data(tmp_path/"data")
    monkeypatch.setitem(PRESETS,"mlx-test",Sys1Config(vocab_size=8,d_model=16,n_layers=2,n_heads=2,n_kv_heads=1,ffn_dim=32,max_seq=16))
    monkeypatch.setattr(train,"RUNS",tmp_path/"runs")
    argv = ["--preset","mlx-test","--tokenizer",str(tok),"--corpus-manifest",str(path),"--seq","8","--micro-batch","2",
            "--accum","2","--steps","6","--lr","0.01","--warmup","2","--precision","fp32","--eval-every","2",
            "--eval-seqs","4","--log-every","1","--early-stop-patience","0","--max-domain-regression","0"]
    if adaptive:argv += ["--mixture-policy","adaptive","--mixture-min-observations","1","--mixture-warmup","1","--mixture-smoothing","1"]
    train.main(["--name","full",*argv])
    train.main(["--name","resumed",*argv,"--stop-after","3"])
    train.main(["--name","resumed",*argv])
    saved = []
    for name in ["full","resumed"]:
        root = tmp_path/"runs/mlx-pretrain"/name
        latest = root/"checkpoints"/(root/"checkpoints/latest").read_text().strip()
        saved.append((json.loads((latest/"state.json").read_text()),mx.load(str(latest/"model.safetensors")),mx.load(str(latest/"optimizer.safetensors"))))
        assert len(list((root/"checkpoints").glob("step_*")))==2
        assert json.loads((root/"log.jsonl").read_text().splitlines()[-1])["event"]=="done"
    a,b = saved
    for key in ("data_id","rng","drawn","sources"):
        assert a[0]["sampler"][key]==b[0]["sampler"][key]
    for key in a[0]["sampler"]["weights"]:
        assert a[0]["sampler"]["weights"][key]==pytest.approx(b[0]["sampler"]["weights"][key],abs=1e-6)
    if adaptive:
        for key in ("policy","last_step","observations"):
            assert a[0]["adaptive"][key]==b[0]["adaptive"][key]
        for field in ("weights","baseline_validation","baseline_training","validation_trend","training_trend"):
            for key in a[0]["adaptive"][field]:
                assert a[0]["adaptive"][field][key]==pytest.approx(b[0]["adaptive"][field][key],abs=1e-6)
        assert a[0]["adaptive"]["weights"] != a[0]["adaptive"]["policy"]["base_weights"]
    else:
        assert a[0]["adaptive"]==b[0]["adaptive"]==None
    assert a[0]["validation"]["evaluations"]==b[0]["validation"]["evaluations"]==3
    for state in (1,2):
        assert a[state].keys()==b[state].keys()
        for key in a[state]:np.testing.assert_allclose(np.array(a[state][key]),np.array(b[state][key]),rtol=5e-5,atol=2e-6)
    # A completed checkpoint can be opened without requiring space for a write
    # that will never happen.
    monkeypatch.setattr(train.shutil,"disk_usage",lambda _:SimpleNamespace(free=0))
    train.main(["--name","resumed",*argv])
    log_path = tmp_path/"runs/mlx-pretrain/resumed/log.jsonl"
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert sum(row["event"]=="train" for row in events)==6


def test_terminal_guard_resume_runs_no_more_optimizer_steps(tmp_path,monkeypatch):
    path,tok = fixture_data(tmp_path/"data")
    monkeypatch.setitem(PRESETS,"mlx-test",Sys1Config(vocab_size=8,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,ffn_dim=32,max_seq=16))
    monkeypatch.setattr(train,"RUNS",tmp_path/"runs")
    values = iter([2.,2.2,2.4])
    monkeypatch.setattr(train,"evaluate",lambda *a:{"en":(value:=next(values)),"code":value})
    argv = ["--name","stopped","--preset","mlx-test","--tokenizer",str(tok),"--corpus-manifest",str(path),"--seq","8",
            "--micro-batch","2","--steps","20","--eval-every","2","--eval-seqs","4","--log-every","1", "--early-stop-patience","2","--max-domain-regression","0"]
    train.main(argv)
    root = tmp_path/"runs/mlx-pretrain/stopped"
    checkpoint = root/"checkpoints/step_00000006/model.safetensors"
    before = checkpoint.read_bytes()
    train.main(argv)
    rows = [json.loads(line) for line in (root/"log.jsonl").read_text().splitlines()]
    assert rows[-1]["event"]=="early_stopped" and rows[-1]["resumed_terminal_guard"]
    assert sum(row["event"]=="train" for row in rows)==6
    assert checkpoint.read_bytes()==before
