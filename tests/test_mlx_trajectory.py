import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from sys1.datasets.trajectory_windows import prepare
from sys1.mlx.trajectory import weighted_loss, run
from test_trajectory_windows import fixture_dataset


@pytest.fixture(autouse=True)
def native_gpu():
    if not mx.metal.is_available():pytest.skip('native Apple GPU unavailable')
    previous=mx.default_device();mx.set_default_device(mx.gpu)
    yield
    mx.synchronize();mx.set_default_device(previous)


def test_weighted_objective_excludes_padding_and_context_gradients():
    class Toy(nn.Module):
        def __init__(self):super().__init__();self.logits=mx.array([[[1.,2.,3.],[3.,2.,1.],[0.,0.,0.]]])
        def __call__(self,ids):return self.logits
    model=Toy();ids=mx.array([[0,1,2,0]]);valid=mx.array([[0,1,1,0]]);target=mx.array([[0,1,0,0]])
    value,grads=nn.value_and_grad(model,lambda m:weighted_loss(m,ids,valid,target,0.))(model)
    mx.eval(value,grads)
    expected=np.log(np.exp([1.,2.,3.]).sum())-2.
    assert float(value)==pytest.approx(expected,abs=1e-6)
    assert np.any(np.array(grads['logits'])[0,0])
    assert not np.any(np.array(grads['logits'])[0,1:])
    uniform=weighted_loss(model,ids,valid,target,1.)
    expected_uniform=(expected+np.log(np.exp([3.,2.,1.]).sum())-1.)/2
    assert float(uniform)==pytest.approx(expected_uniform,abs=1e-6)


def test_native_gpu_study_pairs_initialization_inputs_and_immutable_receipts(tmp_path):
    parent,tok=fixture_dataset(tmp_path/'source')
    windows,_=prepare(parent,tmp_path/'windows',tok,seq=32,target_chunk=8)
    out=tmp_path/'study'
    receipt=run(windows,out,steps=4,seeds=(17,),precision='fp32',memory_gib=.25,ttl_seconds=60,legacy_panel=windows)
    uniform,targeted=receipt['runs']
    assert uniform['initial_parameters_sha256']==targeted['initial_parameters_sha256']
    assert uniform['predicted_nonpadding_tokens']==targeted['predicted_nonpadding_tokens']
    assert uniform['assistant_target_tokens']==targeted['assistant_target_tokens']
    assert json.loads((out/'plan.json').read_text())['device']=='gpu'
    assert not receipt['summary']['diagnostic_gate_passed']
    assert receipt['legacy_secondary_summary']['primary']['uniform_nll']==pytest.approx(receipt['summary']['primary']['uniform_nll'])
    assert not receipt['torch_imported'] and not receipt['matrix_contacted']
    with pytest.raises(FileExistsError):run(windows,out,steps=4,seeds=(17,))
