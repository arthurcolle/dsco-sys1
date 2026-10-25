from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx.utils import tree_flatten
from sys1.mlx.model import PRESETS,Sys1Config,Sys1Model,lm_loss,param_count
from sys1.mlx.train import make_optimizer,make_step
from types import SimpleNamespace


@pytest.fixture(autouse=True)
def gpu():
    if not mx.metal.is_available():pytest.skip("native Apple GPU unavailable")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu);mx.random.seed(7)
    yield
    mx.synchronize();mx.set_default_device(previous)


def config():
    return Sys1Config(vocab_size=32,d_model=32,n_layers=2,n_heads=4,n_kv_heads=2,ffn_dim=64,max_seq=32)


def test_causal_mask_prevents_future_token_leakage():
    model = Sys1Model(config())
    a = mx.array([[2,3,4,5,6,7]])
    b = mx.array([[2,3,4,5,8,9]])
    first,second = model(a),model(b)
    mx.eval(first,second)
    np.testing.assert_allclose(np.array(first[:,:4]),np.array(second[:,:4]),rtol=0,atol=1e-6)
    assert not np.allclose(np.array(first[:,4:]),np.array(second[:,4:]))


def test_torch_weight_parity_for_body_logits_and_typed_heads():
    import torch
    from sys1.model.model import Sys1Config as TorchConfig,Sys1Model as TorchModel
    torch.manual_seed(11)
    reference = TorchModel(TorchConfig(**asdict(config()))).eval()
    model = Sys1Model(config())
    model.load_weights([(name,mx.array(value.detach().numpy())) for name,value in reference.state_dict().items()],strict=True)
    ids = np.array([[2,3,4,5,6,7],[3,2,5,4,7,6]],dtype=np.int32)
    decide,options,mask = np.array([3,4]),np.array([[0,2,4],[1,3,0]]),np.array([[True,True,True],[True,True,False]])
    with torch.no_grad():
        hidden = reference.hidden(torch.tensor(ids,dtype=torch.long))
        expected = reference.lm_logits(hidden).numpy()
        heads = reference.decide(hidden,torch.tensor(decide),torch.tensor(options),torch.tensor(mask))
    native = model.hidden(mx.array(ids))
    actual = model.lm_logits(native)
    actual_heads = model.decide(native,mx.array(decide),mx.array(options),mx.array(mask))
    mx.eval(actual,actual_heads)
    np.testing.assert_allclose(np.array(actual),expected,rtol=2e-5,atol=2e-6)
    for first,second in zip(actual_heads,heads):
        np.testing.assert_allclose(np.array(first),second.numpy(),rtol=2e-5,atol=2e-6)
    assert model.num_parameters()==param_count(config())["total"]


def test_checkpointed_blocks_preserve_first_layer_gradients():
    normal = Sys1Model(config())
    recomputed = Sys1Model(config(),gradient_checkpointing=True)
    recomputed.load_weights(tree_flatten(normal.parameters()))
    normal.freeze_decision_heads();recomputed.freeze_decision_heads()
    windows = mx.array([[2,3,4,2,3,4,2,3,4]])
    (a,_),ga = nn.value_and_grad(normal,lm_loss)(normal,windows)
    (b,_),gb = nn.value_and_grad(recomputed,lm_loss)(recomputed,windows)
    mx.eval(a,b,ga,gb)
    assert float(a)==pytest.approx(float(b),abs=1e-7)
    first,second = dict(tree_flatten(ga)),dict(tree_flatten(gb))
    assert first.keys()==second.keys()
    assert float(mx.sum(mx.abs(first["blocks.0.attn.q.weight"])))>0
    for key in first:
        np.testing.assert_allclose(np.array(first[key]),np.array(second[key]),rtol=3e-5,atol=2e-6)


def test_bf16_compute_fp32_masters_and_compiled_optimizer_learn():
    model = Sys1Model(config(),compute_dtype=mx.bfloat16)
    model.freeze_decision_heads()
    before = np.array(model.choice_q.weight)
    args = SimpleNamespace(lr=.01,beta1=.9,beta2=.95,wd=0.,decay_embeddings=False)
    optimizer = make_optimizer(model,args)
    step = make_step(model,optimizer,1,1,1.,compiled=True)
    x = mx.array([[[2,3,4,2,3,4,2,3,4]]])
    initial = float(lm_loss(model,x[0])[0])
    assert model.hidden(x[0][:,:-1]).dtype==mx.bfloat16
    for _ in range(40):
        loss,norm,sums,counts = step(x,mx.array([[0]]),mx.array(.01))
        mx.eval(model.state,optimizer.state,loss,norm,sums,counts)
    assert float(lm_loss(model,x[0])[0]) < initial*.3
    assert model.embed.weight.dtype==mx.float32
    assert all(value.dtype==mx.float32 for key,value in tree_flatten(optimizer.state) if key.endswith((".m",".v")))
    np.testing.assert_array_equal(np.array(model.choice_q.weight),before)


def test_full_presets_match_system_one_parameter_counts_without_allocating_them():
    from sys1.model.model import PRESETS as torch_presets,param_count as torch_count
    for name in ["sys1-666m","sys1-999m","sys1-999m-v100k","sys1-666m-v100k","sys1-tiny"]:
        assert param_count(PRESETS[name])==torch_count(torch_presets[name])
    assert param_count(PRESETS["sys1-999m-v100k"])["total"]==998562562


def test_native_training_import_does_not_load_torch():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable,"-c","import sys;import sys1.mlx.train;assert 'torch' not in sys.modules;print('mlx-only')"],
                            cwd=repo,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=="mlx-only"
