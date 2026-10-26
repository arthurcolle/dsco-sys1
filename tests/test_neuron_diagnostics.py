from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from sys1.model.model import Sys1Config, Sys1Model
from sys1.neurons.cli import main
from sys1.neurons.engine import encode_probe, screen, validate_options, validate_probes
from sys1.neurons.registry import Registry, digest, identity
from sys1.neurons.report import render, validate
from sys1.neurons.torch_backend import TorchAdapter


@pytest.fixture
def model():
    torch.manual_seed(7)
    return Sys1Model(Sys1Config(vocab_size=16, d_model=16, n_layers=2, n_heads=2, n_kv_heads=1, ffn_dim=24, max_seq=64)).eval()


def files(tmp_path, model):
    vocab = {'[UNK]':0, '[EOS]':1, 'a':2, 'b':3, 'c':4, 'return':5}
    vocab.update({'token'+str(i): i for i in range(6, 16)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = tmp_path/'tokenizer.json';tokenizer.save(str(tok))
    ckpt = tmp_path/'checkpoint.pt'
    torch.save({'step':100, 'config':asdict(model.cfg), 'model':model.state_dict(),
                'rng':{'numpy':np.random.get_state()}, 'data':{'buffer':np.array([2,3], dtype=np.int32)}}, ckpt)
    probes = tmp_path/'probes.json'
    probes.write_text(json.dumps({'schema':'sys1-neuron-probes-v1', 'role':'diagnostic', 'probes':[
        {'id':'fixture','stratum':'code','source':'synthetic test fixture', 'prompt':'a b c', 'continuation':'return a b'}]}))
    return tokenizer, tok, ckpt, probes


def options():
    return {'layers':[0,1], 'neurons':1, 'fixed_units':{}, 'controls':1, 'doses':[.1],
            'seed':1234, 'max_tokens':64, 'max_forwards':16, 'eos_prefix':True}


def test_torch_baseline_matches_direct_model_and_full_target(model):
    adapter = TorchAdapter(model)
    ids, pos, targets = [2,3,4,5], [1,2,3], [4,5,6]
    result = adapter.measure(ids, pos, targets, [0,1])
    with torch.inference_mode():
        lp = torch.log_softmax(model.lm_logits(model.hidden(torch.tensor([ids])))[:,pos].float(), -1)[0]
    expected = lp[torch.arange(3),torch.tensor(targets)]
    np.testing.assert_allclose(result['target_logprobs'], expected.numpy(), atol=1e-7)
    assert result['mean_nll'] == pytest.approx(float(-expected.mean()))
    assert result['activations'][0].shape == (4,24)


def test_add_at_neuron_equals_its_down_column_output_push(model):
    adapter = TorchAdapter(model)
    ids, pos, targets = [2,3,4], [0,1,2], [3,4,5]
    before = {k:v.clone() for k,v in model.state_dict().items()}
    value, n = .3, 4
    neuron = adapter.measure(ids,pos,targets,[],{'kind':'neuron','layer':0,'neuron':n,'mode':'add','value':value})
    column = model.blocks[0].mlp.down.weight[:,n].detach().numpy()
    residual = adapter.measure(ids,pos,targets,[],{'kind':'residual','layer':0,'delta':np.tile(value*column,(3,1)).tolist()})
    np.testing.assert_allclose(neuron['target_logprobs'], residual['target_logprobs'], atol=1e-6)
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())


def test_exception_removes_temporary_hooks(model,monkeypatch):
    adapter = TorchAdapter(model)
    def fail(*a,**k):raise RuntimeError('synthetic failure')
    monkeypatch.setattr(model,'hidden',fail)
    with pytest.raises(RuntimeError,match='synthetic failure'):adapter.measure([2,3],[1],[4],[0])
    assert not model.blocks[0].mlp.down._forward_pre_hooks
    assert not model.blocks[0].mlp.down._forward_hooks
    assert not model.blocks[0]._forward_hooks


def test_forward_and_calibration_are_serialized(model,monkeypatch):
    adapter = TorchAdapter(model)
    original = model.hidden;active = 0;maximum = 0;lock = threading.Lock();results=[]
    def monitored(x):
        nonlocal active,maximum
        with lock:active+=1;maximum=max(maximum,active)
        time.sleep(.02)
        try:return original(x)
        finally:
            with lock:active-=1
    monkeypatch.setattr(model,'hidden',monitored)
    def worker():results.append(adapter.measure([2,3],[1],[4],[0]))
    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    assert maximum==1 and len(results)==2
    assert results[0]['residual_norm']==results[1]['residual_norm']


def test_registry_refuses_relabeling_and_detects_corrupted_cases(tmp_path):
    root=tmp_path/'run';plan={'model':'first','tokenizer':'a','prompts':'frozen'}
    registry=Registry(root,plan)
    with registry.locked():registry.save({'probe':'one'},{'loss':2.})
    before=(root/'plan.json').read_bytes()
    with pytest.raises(ValueError,match='identity differs'):
        with Registry(root,{**plan,'model':'second'},True).locked():pass
    assert (root/'plan.json').read_bytes()==before
    path=next(root.glob('case-*.json'));row=json.loads(path.read_text());row['result']['loss']=0.;path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='digest differs'):registry.case({'probe':'one'})


def test_registry_requires_explicit_resume_and_single_writer(tmp_path):
    root=tmp_path/'run';registry=Registry(root,{'model':'first'})
    with pytest.raises(FileExistsError):Registry(root,{'model':'first'})
    with registry.locked():
        with pytest.raises(BlockingIOError):
            with Registry(root,{'model':'first'},True).locked():pass


def test_engine_deterministic_controls_fixed_units_and_resume(tmp_path,model):
    tokenizer,_,_,probe_path=files(tmp_path,model)
    probes=validate_probes(json.loads(probe_path.read_text()))
    opts=options();opts['fixed_units']={'0':[2],'1':[3]}
    adapter=TorchAdapter(model);registry=Registry(tmp_path/'run',{'frozen':True})
    with registry.locked():
        first=screen(adapter,tokenizer,probes,opts,registry)
        second=screen(adapter,tokenizer,probes,opts,registry)
    assert first==second and first[0]['forwards']==9
    assert [(r['layer'],r['neuron']) for r in first[0]['neurons']]==[(0,2),(1,3)]
    assert len(first[0]['baseline']['target_logprobs'])==3
    other=Registry(tmp_path/'other',{'frozen':True})
    with other.locked():third=screen(adapter,tokenizer,probes,opts,other)
    for a,b in zip(first[0]['neurons'],third[0]['neurons']):assert a['random_direction_controls']==b['random_direction_controls']


def test_limits_reject_before_forward(model,tmp_path):
    tokenizer,_,_,probe_path=files(tmp_path,model)
    probes=validate_probes(json.loads(probe_path.read_text()))
    with pytest.raises(ValueError,match='forward-pass'):validate_options(asdict(model.cfg),probes,{**options(),'max_forwards':2})
    with pytest.raises(ValueError,match='never silently truncated'):encode_probe(tokenizer,probes[0],2,16)
    with pytest.raises(ValueError,match='final holdout'):validate_probes({'schema':'sys1-neuron-probes-v1','role':'final_test','probes':[]})
    with pytest.raises(ValueError,match='valid capture layers|invalid capture layers'):
        TorchAdapter(model).measure([2,3],[1],[4],[99])


def test_cli_actual_checkpoint_complete_resume_and_mismatch(tmp_path,model):
    _,tok,ckpt,probes=files(tmp_path,model)
    root=tmp_path/'run'
    argv=['--backend','torch','--checkpoint',str(ckpt),'--tokenizer',str(tok),'--probes',str(probes),'--output',str(root),'--layers','0','--neurons','1','--max-tokens','32']
    result=main(argv)
    assert result['status']=='complete' and (root/'report.html').is_file()
    assert result['results'][0]['forwards']==3
    before=(root/'plan.json').read_bytes();complete=(root/'complete.json').read_bytes()
    main(argv+['--resume'])
    assert (root/'complete.json').read_bytes()==complete
    with pytest.raises(ValueError,match='identity differs'):main(argv+['--resume','--seed','9876'])
    assert (root/'plan.json').read_bytes()==before
    changed=json.loads(probes.read_text());changed['probes'][0]['prompt']='c b a';probes.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='identity differs'):main(argv+['--resume'])
    assert (root/'plan.json').read_bytes()==before


def test_report_escapes_probe_html_and_refuses_incompatible_identity(tmp_path,model):
    _,tok,ckpt,probes=files(tmp_path,model)
    data=json.loads(probes.read_text());data['probes'][0]['prompt']='</script><img src=x onerror=alert(1)>';probes.write_text(json.dumps(data))
    record=main(['--backend','torch','--checkpoint',str(ckpt),'--tokenizer',str(tok),'--probes',str(probes),'--output',str(tmp_path/'run'),'--layers','0','--neurons','1','--max-tokens','64'])
    html=(tmp_path/'run/report.html').read_text()
    assert data['probes'][0]['prompt'] not in html and '\\u003c/script' in html
    broken=json.loads(json.dumps(record));broken['results'][0]['baseline']['mean_nll']=0
    with pytest.raises(ValueError,match='digest differs'):validate(broken)
    other=json.loads(json.dumps(record));other['plan']['comparison']['precision']='different';other['plan']['comparison_key']=digest(other['plan']['comparison']);other['fingerprint']=digest(other['plan'])
    with pytest.raises(ValueError,match='identical probes'):render([record,other],tmp_path/'bad.html')


def test_mlx_native_parity_and_no_torch_dependency(tmp_path,model):
    mx=pytest.importorskip('mlx.core')
    from sys1.mlx.model import Sys1Config as Config,Sys1Model as Model
    from sys1.neurons.mlx_backend import MLXAdapter
    previous=mx.default_device();mx.set_default_device(mx.cpu)
    try:
        native=Model(Config(**asdict(model.cfg)))
        native.load_weights([(k,mx.array(v.detach().numpy())) for k,v in model.state_dict().items()],strict=True)
        a,b=TorchAdapter(model),MLXAdapter(native)
        for intervention in [None,{'kind':'neuron','layer':0,'neuron':2,'mode':'mul','value':0.}, {'kind':'neuron','layer':1,'neuron':3,'mode':'add','value':.2}]:
            first=a.measure([2,3,4],[1,2],[4,5],[0,1],intervention)
            second=b.measure([2,3,4],[1,2],[4,5],[0,1],intervention)
            np.testing.assert_allclose(first['target_logprobs'],second['target_logprobs'],rtol=2e-5,atol=2e-6)
            for layer in [0,1]:
                np.testing.assert_allclose(first['activations'][layer],second['activations'][layer],rtol=3e-5,atol=2e-6)
                assert second['residual_norm'][layer]==pytest.approx(first['residual_norm'][layer],rel=2e-5)
        ids=mx.array([[2,3,4]],dtype=mx.int32)
        direct=native(ids)
        lp=direct[0,2]-mx.logsumexp(direct[0,2])
        measurement=b.measure([2,3,4],[2],[5],[])
        assert measurement['target_logprobs'][0]==pytest.approx(float(lp[5].item()),abs=2e-6)
    finally:mx.set_default_device(previous)
    script='''import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname=="torch" or fullname.startswith("torch."):raise RuntimeError("Torch import forbidden")
sys.meta_path.insert(0,Block())
import mlx.core as mx
mx.set_default_device(mx.cpu)
from sys1.mlx.model import Sys1Config,Sys1Model
from sys1.neurons.mlx_backend import MLXAdapter
m=Sys1Model(Sys1Config(vocab_size=8,d_model=8,n_layers=1,n_heads=2,n_kv_heads=1,ffn_dim=16,max_seq=8))
assert MLXAdapter(m).measure([2,3],[1],[4],[0])["mean_nll"]>0
assert "torch" not in sys.modules
'''
    subprocess.run([sys.executable,'-c',script],check=True)


def test_matched_checkpoint_report_and_javascript_controls(tmp_path,model):
    _,tok,ckpt,probes=files(tmp_path,model)
    def run(checkpoint,name):
        return main(['--backend','torch','--checkpoint',str(checkpoint),'--tokenizer',str(tok),'--probes',str(probes),
                     '--output',str(tmp_path/name),'--layers','0','--units','0:3','--max-tokens','32'])
    first=run(ckpt,'first')
    with torch.no_grad():model.embed.weight[4,1]+=.1
    second_ckpt=tmp_path/'second.pt'
    torch.save({'step':101,'config':asdict(model.cfg),'model':model.state_dict()},second_ckpt)
    second=run(second_ckpt,'second')
    assert first['plan']['comparison_key']==second['plan']['comparison_key']
    assert first['fingerprint']!=second['fingerprint']
    assert first['results'][0]['baseline']['mean_nll']!=second['results'][0]['baseline']['mean_nll']
    path=tmp_path/'compare.html';render([first,second],path)
    html=path.read_text()
    assert 'step ' in html and first['fingerprint'] in html and second['fingerprint'] in html
    if not shutil.which('node'):pytest.skip('Node unavailable for pure JavaScript DOM-harness check')
    scripts=re.findall(r'<script(?: [^>]*)?>(.*?)</script>',html,re.S)
    assert len(scripts)==2
    # Executes the real report script against a small DOM test double; not a visual/browser check.
    program='''const vm=require('node:vm'),assert=require('node:assert/strict');
class Element {constructor(){this.children=[];this.style={};this.textContent='';}append(x){this.children.push(x)}replaceChildren(){this.children=[]}}
const nodes={},document={getElementById(id){return nodes[id]??=new Element()},createElement(){return new Element()}};
document.getElementById('records').textContent=DATA;
const context=vm.createContext({document});vm.runInContext(SCRIPT,context);
assert.equal(nodes['run'].children.length,2);assert.equal(nodes['probe'].children.length,1);
assert.match(nodes['unit-title'].textContent,/L0 · N3/);assert.ok(nodes['trace'].children.length>0);
const before=nodes['loss'].textContent;nodes['run'].onchange({target:{value:'1'}});
assert.notEqual(nodes['loss'].textContent,before);assert.match(nodes['pair'].textContent,/change from first checkpoint/);
nodes['neurons'].children[0].onclick();assert.match(nodes['unit-title'].textContent,/N3/);
'''.replace('DATA',json.dumps(scripts[0])).replace('SCRIPT',json.dumps(scripts[1]))
    subprocess.run(['node','-e',program],check=True)


def test_native_mlx_cli_checkpoint_loading_without_torch(tmp_path):
    pytest.importorskip('mlx.core')
    script='''import importlib.abc,sys,json
from pathlib import Path
from dataclasses import asdict
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname=="torch" or fullname.startswith("torch."):raise RuntimeError("Torch import forbidden")
sys.meta_path.insert(0,Block())
import mlx.core as mx
mx.set_default_device(mx.cpu)
from sys1.mlx.model import Sys1Config,Sys1Model
from sys1.neurons.cli import main
from sys1.neurons.registry import identity
from tokenizers import Tokenizer,models,pre_tokenizers
root=Path(sys.argv[1]);ckpt=root/"checkpoint";ckpt.mkdir()
model=Sys1Model(Sys1Config(vocab_size=8,d_model=8,n_layers=1,n_heads=2,n_kv_heads=1,ffn_dim=16,max_seq=16))
model.save_weights(str(ckpt/"model.safetensors"))
tok=Tokenizer(models.WordLevel({"[UNK]":0,"[EOS]":1,"a":2,"b":3,"c":4,"x":5,"y":6,"z":7},unk_token="[UNK]"));tok.pre_tokenizer=pre_tokenizers.Whitespace();tok.save(str(root/"tokenizer.json"))
(ckpt/"state.json").write_text(json.dumps({"schema":"sys1-mlx-checkpoint-v1","step":0,"config":asdict(model.cfg),"policy":{"precision":"fp32","tokenizer_sha256":identity(root/"tokenizer.json")["sha256"]}}))
(root/"probes.json").write_text(json.dumps({"schema":"sys1-neuron-probes-v1","role":"diagnostic","probes":[{"id":"demo","stratum":"synthetic","source":"untrained fixture","prompt":"a b","continuation":"c x"}]}))
record=main(["--backend","mlx","--device","cpu","--checkpoint",str(ckpt),"--tokenizer",str(root/"tokenizer.json"),"--probes",str(root/"probes.json"),"--output",str(root/"run"),"--layers","0","--neurons","1","--max-tokens","16"])
assert record["status"]=="complete" and "torch" not in sys.modules
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path)],check=True)
