from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest
from tokenizers import Tokenizer,models,pre_tokenizers

mx=pytest.importorskip('mlx.core')
import mlx.nn as nn
from mlx.utils import tree_flatten
from sys1.datasets.http_stream import publish,server,set_policy
from sys1.mlx.streaming import Reader,build_stream,fetch,CHUNK_BYTES
from sys1.mlx.model import Sys1Config,Sys1Model,lm_loss
from sys1.mlx.train import make_optimizer,make_step
from sys1.model.corpus_manifest import file_identity
from types import SimpleNamespace


@pytest.fixture
def served(tmp_path):
    tok=Tokenizer(models.WordLevel({'[UNK]':0,'[EOS]':1,'alpha':2,'beta':3,'gamma':4},unk_token='[UNK]'))
    tok.pre_tokenizer=pre_tokenizers.Whitespace();tp=tmp_path/'tokenizer.json';tok.save(str(tp))
    entries={}
    for name in ['en','code']:
        entries[name]={}
        for split in ['train','validation']:
            path=tmp_path/(name+'-'+split+'.jsonl')
            with path.open('w') as f:
                for i in range(32):f.write(json.dumps({'text':('alpha beta gamma '*64)+str(i)+split})+'\n')
            entries[name][split]=[{'path':path.name,**file_identity(path)}]
    p=tmp_path/'MANIFEST.json';p.write_text(json.dumps({'schema':'sys1-corpus-mixture-v1','sources':entries,
                  'objective_weights':{'en':.6,'code':.4},'tokenizer_identity':file_identity(tp)}))
    index=tmp_path/'index';receipt=publish(p,tp,index);http=server(index,0)
    thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
    try:yield tmp_path,tok,tp,index,receipt,'http://127.0.0.1:'+str(http.server_port)+'/manifest.json'
    finally:http.shutdown();http.server_close();thread.join()


def test_full_pass_validation_zero_disk_cache_and_deterministic_resume(served):
    root,tok,tp,index,receipt,url=served;before=set(root.rglob('*'))
    mix,held,origin,vid=build_stream(url,receipt['manifest_sha256'],tp,tok,16,8,7)
    assert set(root.rglob('*'))==before
    assert all(v['validation_sampling']['complete_pass'] for v in origin.values())
    mix.batch(9);state=json.loads(json.dumps(mix.state()));a,labels=mix.batch(17)
    restored,other,_,again=build_stream(url,receipt['manifest_sha256'],tp,tok,16,8,7)
    restored.load_state(state);b,other_labels=restored.batch(17)
    np.testing.assert_array_equal(np.array(a),np.array(b));np.testing.assert_array_equal(np.array(labels),np.array(other_labels))
    assert vid==again
    for name in held:np.testing.assert_array_equal(np.array(held[name]),np.array(other[name]))
    assert receipt['dataset_bytes_copied']==0


def test_dynamic_policy_applies_at_batch_boundary_and_resumes(served):
    root,tok,tp,index,receipt,url=served;path=root/'policy.json';sha=receipt['manifest_sha256']
    set_policy(path,sha,{'en':.9,'code':.1},1)
    mix,_,_,_=build_stream(url,sha,tp,tok,16,2,7,path)
    assert mix.weights=={'en':.6,'code':.4}
    mix.batch(2);assert mix.weights=={'en':.9,'code':.1} and mix.policy_version==1
    set_policy(path,sha,{'en':.2,'code':.8},2);mix.batch(2)
    saved=mix.state();a,l=mix.batch(8)
    other,_,_,_=build_stream(url,sha,tp,tok,16,2,7,path);other.load_state(saved);b,k=other.batch(8)
    np.testing.assert_array_equal(np.array(a),np.array(b));np.testing.assert_array_equal(np.array(l),np.array(k))
    assert other.policy_version==2
    with pytest.raises(ValueError,match='increase'):set_policy(path,sha,{'en':.9,'code':.1},1)
    p=json.loads(path.read_text());p['version']=1;path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='rollback'):mix.batch(1)


def test_source_mutation_and_manifest_tampering_fail_closed(served):
    root,tok,tp,index,receipt,url=served
    with pytest.raises(ValueError,match='identity'):build_stream(url,'0'*64,tp,tok,16,2,7)
    mix,_,_,_=build_stream(url,receipt['manifest_sha256'],tp,tok,16,2,7)
    p=root/'en-train.jsonl';p.write_text(p.read_text().replace('alpha','omega'))
    with pytest.raises(Exception):mix.sources['en'].next_window()


def test_reader_handles_utf8_document_across_chunk_boundaries(tmp_path):
    from sys1.mlx import streaming
    text='é'*(CHUNK_BYTES//2+10);raw=(json.dumps({'text':text},ensure_ascii=False)+'\n').encode()
    chunks=[raw[i:i+CHUNK_BYTES] for i in range(0,len(raw),CHUNK_BYTES)]
    record={'resource':'chunk/id','bytes':len(raw),'chunks':[hashlib.sha256(c).hexdigest() for c in chunks]}
    original=streaming.fetch
    streaming.fetch=lambda url,limit:chunks[int(url.rsplit('/',1)[1])]
    try:
        r=Reader('http://127.0.0.1',[record],repeat=False);assert r.next_doc()==text
        with pytest.raises(StopIteration):r.next_doc()
        with pytest.raises(ValueError):r.load_state({'file':0,'chunk':0,'offset':CHUNK_BYTES+1,'epoch':0})
    finally:streaming.fetch=original


def test_plain_remote_http_rejected():
    with pytest.raises(ValueError,match='HTTPS'):fetch('http://example.com/private',10)


def test_zero_weight_pauses_source_and_survives_resume(served):
    root,tok,tp,index,receipt,url=served;path=root/'policy.json';sha=receipt['manifest_sha256']
    set_policy(path,sha,{'en':1,'code':0},1)
    mix,_,_,_=build_stream(url,sha,tp,tok,16,2,7,path);mix.batch(20)
    assert mix.drawn['code']==0
    other,_,_,_=build_stream(url,sha,tp,tok,16,2,7,path);other.load_state(mix.state());other.batch(20)
    assert other.weights['code']==0 and other.drawn['code']==0
    with pytest.raises(ValueError):set_policy(path,sha,{'en':0,'code':0},2)


@pytest.mark.parametrize('compiled',[False,True])
def test_chunked_head_loss_preserves_loss_and_gradients(compiled):
    if not mx.metal.is_available():pytest.skip('Metal GPU unavailable')
    mx.set_default_device(mx.gpu);mx.random.seed(19)
    cfg=Sys1Config(vocab_size=32,d_model=16,n_layers=2,n_heads=2,n_kv_heads=1,ffn_dim=32,max_seq=32)
    a=Sys1Model(cfg,gradient_checkpointing=True);b=Sys1Model(cfg,gradient_checkpointing=True)
    b.load_weights(tree_flatten(a.parameters()));a.freeze_decision_heads();b.freeze_decision_heads()
    x=mx.array([[2,3,4,5,6,7,8,9,10],[3,4,5,6,7,8,9,10,11]])
    (la,_),ga=nn.value_and_grad(a,lm_loss)(a,x)
    def chunked(model,values):return lm_loss(model,values,3)
    (lb,_),gb=nn.value_and_grad(b,chunked)(b,x)
    mx.eval(la,lb,ga,gb)
    assert float(la)==pytest.approx(float(lb),abs=1e-6)
    for (ka,va),(kb,vb) in zip(tree_flatten(ga),tree_flatten(gb)):
        assert ka==kb;np.testing.assert_allclose(np.array(va),np.array(vb),rtol=2e-4,atol=2e-6)
    opts=SimpleNamespace(lr=.001,beta1=.9,beta2=.95,wd=0.,decay_embeddings=False)
    step=make_step(b,make_optimizer(b,opts),1,1,1.,compiled=compiled,chunk_tokens=3)
    loss,norm,sums,counts=step(x[None],mx.array([[0,0]]),mx.array(.001))
    mx.eval(b.state,loss,norm,sums,counts);assert np.isfinite(float(loss)) and counts.tolist()==[2]


def test_imports_do_not_load_torch():
    r=subprocess.run([sys.executable,'-c',"import sys; import sys1.mlx.streaming,sys1.mlx.train,sys1.mlx.legacy_weights; assert 'torch' not in sys.modules"],capture_output=True,text=True)
    assert r.returncode==0,r.stderr


def test_stream_training_resume_with_policy_change_and_fixed_validation(served,monkeypatch):
    if not mx.metal.is_available():pytest.skip('Metal GPU unavailable')
    from sys1.mlx import train
    root,tok,tp,index,receipt,url=served;policy=root/'weights.json';sha=receipt['manifest_sha256']
    set_policy(policy,sha,{'en':.9,'code':.1},1)
    monkeypatch.setitem(train.PRESETS,'stream-test',Sys1Config(vocab_size=5,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,ffn_dim=32,max_seq=16))
    monkeypatch.setattr(train,'RUNS',root/'runs')
    args=['--name','stream-test','--preset','stream-test','--tokenizer',str(tp),'--stream-manifest',url,
          '--stream-manifest-sha256',sha,'--stream-policy',str(policy),'--seq','8','--micro-batch','1','--accum','1',
          '--steps','4','--warmup','1','--eval-every','2','--eval-seqs','2','--log-every','1',
          '--early-stop-patience','0','--max-domain-regression','0','--loss-chunk-tokens','3','--precision','fp32']
    train.main(args+['--stop-after','2'])
    set_policy(policy,sha,{'en':.2,'code':.8},2);train.main(args)
    folder=root/'runs/mlx-pretrain/stream-test';checkpoint=folder/'checkpoints/step_00000004/state.json'
    saved=json.loads(checkpoint.read_text())
    assert saved['step']==4 and saved['sampler']['weights']=={'en':.2,'code':.8}
    assert saved['sampler']['policy_version']==2
    assert saved['policy']['weights']=={'en':.6,'code':.4}
    events=[json.loads(line) for line in (folder/'log.jsonl').read_text().splitlines()]
    assert [r['version'] for r in events if r['event']=='stream_policy']==[1,2]
    assert sum(r['event']=='train' for r in events)==4
    assert events[-1]['event']=='done'


def test_stream_initialization_uses_pinned_weights_not_random_reinitialization(served,monkeypatch):
    if not mx.metal.is_available():pytest.skip('Metal GPU unavailable')
    from sys1.mlx import train
    root,tok,tp,index,receipt,url=served
    cfg=Sys1Config(vocab_size=5,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,ffn_dim=32,max_seq=16)
    mx.random.seed(831);initial=Sys1Model(cfg);initial_path=root/'initial.safetensors'
    mx.eval(initial.parameters());initial.save_weights(str(initial_path));expected=np.array(initial.embed.weight)
    init=root/'initialization.json';init.write_text(json.dumps({'schema':'sys1-mlx-initialization-v1','parent_completed':True,
        'config':asdict(cfg),'tokenizer_identity':file_identity(tp),'model_identity':file_identity(initial_path)}))
    monkeypatch.setitem(train.PRESETS,'init-test',cfg);monkeypatch.setattr(train,'RUNS',root/'runs')
    train.main(['--name','init-test','--preset','init-test','--tokenizer',str(tp),'--stream-manifest',url,
        '--stream-manifest-sha256',receipt['manifest_sha256'],'--init-model',str(initial_path),'--init-receipt',str(init),
        '--seq','8','--micro-batch','1','--steps','1','--warmup','1','--lr','1e-20','--wd','0','--eval-seqs','2',
        '--early-stop-patience','0','--max-domain-regression','0','--precision','fp32'])
    saved=mx.load(str(root/'runs/mlx-pretrain/init-test/checkpoints/step_00000001/model.safetensors'))
    np.testing.assert_array_equal(np.array(saved['embed.weight']),expected)
