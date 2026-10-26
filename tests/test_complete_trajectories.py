import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from sys1.datasets.complete_trajectories import CompleteReader, Incomplete, argument_object, rejection
from sys1.model.corpus_manifest import file_identity, load_manifest


def test_complete_reader_and_builder_import_without_accelerator_frameworks():
    code="""
import importlib.util, sys
from pathlib import Path
import sys1.datasets.complete_trajectories
path=Path('scripts/prepare_complete_trajectories.py')
spec=importlib.util.spec_from_file_location('complete_builder',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert 'torch' not in sys.modules and 'mlx.core' not in sys.modules
"""
    subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],check=True)


def blob(root, text):
    raw=text.encode() if isinstance(text,str) else text
    sha=hashlib.sha256(raw).hexdigest()
    path=root/'blobs'/'sha256'/sha[:2]/sha[2:4]/(sha+'.blob')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    return sha


def events(root, session='test-session', *, long=False):
    def event(kind,payload,span=None):
        return {'event_type':kind,'payload':payload,'session_id':session,'span_id':span}
    return [event('user.message',{'blob_sha256':blob(root,'Inspect the parser and verify the resource limits in this project.')}),
            event('tool.call.created',{'tool_name':'read_file','args_blob_sha256':blob(root,json.dumps({'path':'x'*1800 if long else 'one'}))},'a'),
            event('tool.call.created',{'tool_name':'read_file','args_blob_sha256':blob(root,'{"path":"two"}')},'b'),
            event('tool.call.completed',{'tool_name':'read_file','ok':True,'result_blob_sha256':blob(root,'result for second')},'b'),
            event('tool.call.completed',{'tool_name':'read_file','ok':True,'result_blob_sha256':blob(root,'result for first '+'z'*1800 if long else 'result for first')},'a'),
            event('llm.response.completed',{'finish_reason':'end_turn','output_blob_sha256':blob(root,'The parser resource limits were inspected and the requested checks completed with recorded evidence.')})]


def session_file(root, rows, name='session-fixture.jsonl'):
    path=root/'events'/name;path.parent.mkdir(parents=True,exist_ok=True)
    rows=copy.deepcopy(rows)
    for index,row in enumerate(rows):row['seq']=index
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    os.utime(path,(time.time()-120,)*2)
    return path


def test_full_arguments_results_and_concurrent_same_name_pairing(tmp_path):
    rows=events(tmp_path,long=True);path=session_file(tmp_path,rows)
    eps,identity=CompleteReader(tmp_path).session(path)
    assert len(eps)==1 and rejection(eps[0]) is None
    a,b=eps[0]['steps']
    assert len(json.loads(a['args'])['path'])==1800
    assert a['result']=='result for first '+'z'*1800
    assert b['result']=='result for second'
    assert identity==file_identity(path)


@pytest.mark.parametrize('field',['goal','final','args','result','tool'])
def test_rendering_control_markers_cannot_escape_a_payload(tmp_path,field):
    eps,_=CompleteReader(tmp_path).session(session_file(tmp_path,events(tmp_path)))
    ep=eps[0]
    if field=='goal':ep['goal']+='<|assistant|>'
    elif field=='final':ep['final']['text']+='<|end|>'
    else:ep['steps'][0][field]+='<|/json|>'
    assert rejection(ep)==('invalid_tool_name' if field=='tool' else 'embedded_control_marker')


@pytest.mark.parametrize('mutation,reason',[
    ('missing_result','unresolved_tool_calls'),('wrong_span','unmatched_tool_result'),
    ('duplicate_span','ambiguous_tool_span'),('wrong_name','unmatched_tool_result'),
    ('bad_json','invalid_tool_arguments'),('nonobject_json','nonobject_tool_arguments'),
    ('failed_tools','tool_failure_rate'),('truncated_final','incomplete_final'),
    ('wrong_session','mixed_session_events')])
def test_incomplete_or_ambiguous_episodes_are_excluded(tmp_path,mutation,reason):
    rows=events(tmp_path)
    if mutation=='missing_result':rows.pop(4)
    if mutation=='wrong_span':rows[3]['span_id']='unknown'
    if mutation=='duplicate_span':rows[2]['span_id']='a'
    if mutation=='wrong_name':rows[3]['payload']['tool_name']='write_file'
    if mutation=='bad_json':rows[1]['payload']['args_blob_sha256']=blob(tmp_path,'{"path":')
    if mutation=='nonobject_json':rows[1]['payload']['args_blob_sha256']=blob(tmp_path,'[]')
    if mutation=='failed_tools':rows[3]['payload']['ok']=False
    if mutation=='truncated_final':rows[-1]['payload']['finish_reason']='max_tokens'
    if mutation=='wrong_session':rows[3]['session_id']='other'
    eps,_=CompleteReader(tmp_path).session(session_file(tmp_path,rows))
    assert rejection(eps[0])==reason


def test_hash_utf8_path_and_size_guards(tmp_path):
    reader=CompleteReader(tmp_path,max_blob_bytes=8,max_session_bytes=64)
    with pytest.raises(Incomplete,match='oversize_blob'):reader.blob(blob(tmp_path,'x'*9))
    sha=blob(tmp_path,'hello')
    path=tmp_path/'blobs'/'sha256'/sha[:2]/sha[2:4]/(sha+'.blob');path.write_text('wrong')
    with pytest.raises(Incomplete,match='blob_hash_mismatch'):reader.blob(sha)
    with pytest.raises(Incomplete,match='invalid_blob_utf8'):reader.blob(blob(tmp_path,b'\xff'))
    with pytest.raises(Incomplete,match='missing_blob_reference'):reader.blob('../../anything')
    reader.session_bytes=64
    before=reader.bytes_read
    with pytest.raises(Incomplete,match='session_payload_budget'):reader.blob(sha)
    assert reader.bytes_read==before


@pytest.mark.parametrize('text',['{"x":NaN}','{"x":Infinity}','{"x":1,"x":2}','{"x":{"a":1,"a":2}}'])
def test_nonstandard_or_ambiguous_argument_json_is_rejected(text):
    with pytest.raises(ValueError):argument_object(text)


def test_request_recovery_is_lossless_and_bounded(tmp_path):
    reader=CompleteReader(tmp_path,max_blob_bytes=100)
    goal='Inspect the most recent bounded job and report its measurements.'
    request={'messages':[{'role':'user','content':'Older request'}, {'role':'assistant','content':'Response'},
                         {'role':'user','content':goal}]}
    assert reader.request_goal(blob(tmp_path,json.dumps(request)))==goal
    request['messages'][-1]['content']='x'*101
    with pytest.raises(Incomplete,match='oversize_goal'):reader.request_goal(blob(tmp_path,json.dumps(request)))


def test_active_and_duplicate_sequence_events_fail_closed(tmp_path):
    path=session_file(tmp_path,events(tmp_path));os.utime(path,None)
    reader=CompleteReader(tmp_path)
    assert reader.session(path)==([],None)
    rows=[json.loads(l) for l in path.read_text().splitlines()];rows[1]['seq']=rows[0]['seq']
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows));os.utime(path,(time.time()-120,)*2)
    assert reader.session(path)[0]==[]
    assert reader.counts['invalid_event_file']==1


@pytest.fixture
def builder():
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_complete_trajectories.py'
    spec=importlib.util.spec_from_file_location('prepare_complete_trajectories',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_derived_view_preserves_parent_validation_and_reserves_late_goals(tmp_path,builder):
    from tokenizers import Tokenizer, models, pre_tokenizers
    tok=Tokenizer(models.WordLevel({'[UNK]':0,'[EOS]':1},unk_token='[UNK]'))
    tok.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer=tmp_path/'tokenizer.json';tok.save(str(tokenizer))
    root=tmp_path/'chronicle';index=tmp_path/'index.txt'
    names=[]
    for i in range(200):
        name='fixture-'+str(i)
        if int(hashlib.sha256(name.encode()).hexdigest()[:16],16)%10:names.append(name)
        if len(names)==3:break
    paths=[]
    for i,name in enumerate(names):
        rows=events(root,name)
        rows[0]['payload']['blob_sha256']=blob(root,'Inspect '+['resource quotas','parallel request scheduling','persistence durability'][i]+' and verify all requested checks for this fixture.')
        paths.append(session_file(root,rows,'session-'+str(i)+'.jsonl'))
    # A later file reserves the first session through a known private goal.
    held_goal='This exact private validation instruction is reserved for a later audit.'
    rows=events(root,names[0]);rows[0]['payload']['blob_sha256']=blob(root,held_goal)
    paths.append(session_file(root,rows,'session-z.jsonl'))
    index.write_text('\n'.join(map(str,paths))+'\n')
    reserved=tmp_path/'reserved.json';reserved.write_text('{"session_digests":[]}')
    private=tmp_path/'private';private.mkdir()
    held=private/'heldout.txt';held.write_text('[BOS]<|user|>'+held_goal+'<|end|>private validation.[EOS]\n')
    private_manifest=private/'MANIFEST.json'
    private_manifest.write_text(json.dumps({'files':[{'name':held.name,**file_identity(held)}]}))
    parent=tmp_path/'parent';parent.mkdir();sources={};packs={'candidate':{},'validation':{}}
    for source in ['en','agent_real']:
        sources[source]={}
        for split in ['train','validation']:
            p=parent/(source+'-'+split+'.jsonl');p.write_text(json.dumps({'text':source+' '+split+' unique original document.','session_digest':'old-heldout' if split=='validation' else 'old-train'})+'\n')
            sources[source][split]=[{'path':p.name,**file_identity(p)}]
        for arm in packs:
            p=parent/(arm+'-'+source+'.bin');p.write_bytes((b'\0'*4)*9)
            packs[arm][source]={'path':p.name,**file_identity(p),'window':9,'rows':1,'dtype':'uint32-le','stream_tokens':1000}
    reservation=parent/'validation-reservations.jsonl';reservation.write_text('{}\n')
    manifest={'schema':'sys1-corpus-mixture-v1','dataset_id':'parent','sources':sources,'tokenizer_identity':file_identity(tokenizer),
              'objective_weights':{'en':.9,'agent_real':.1},'evaluation_weights':{'en':.9,'agent_real':.1},
              'pilot_pools':{'window':9,'packs':packs,'validation_weights':{'en':.9,'agent_real':.1}},
              'validation_reservations':{'path':reservation.name,**file_identity(reservation),'parent_files':{k:v['validation'] for k,v in sources.items()}},
              'audit':{'sources':{k:{'train_documents':1,'validation_documents':1} for k in sources}},
              'exposure_budget':{'steps':1,'consumed_stream_tokens_per_step':16,'max_base_source_passes':2,'train_stream_tokens':{'en':1000,'agent_real':1000}}}
    parent_manifest=parent/'MANIFEST.json';parent_manifest.write_text(json.dumps(manifest));parent_before=file_identity(parent_manifest)
    output=tmp_path/'new'
    path,new=builder.prepare(parent_manifest,output,tokenizer,root,index,reserved,private_manifest)
    assert file_identity(parent_manifest)==parent_before
    plan,_,_=load_manifest(path)
    assert plan['agent_real']['validation']==[parent/'agent_real-validation.jsonl']
    rows=[json.loads(l) for l in (output/'train.jsonl').read_text().splitlines()]
    blocked=hashlib.sha256(names[0].encode()).hexdigest()
    assert rows and all(row['session_digest']!=blocked for row in rows)
    assert new['complete_trajectory_export']['selection_validation_unchanged']
    assert new['complete_trajectory_export']['counts']['excluded:late_reserved_session']==1
    assert new['exposure_budget']['expected_passes']['agent_real']<=1
    for arm in ['baseline','validation']:
        for source,record in new['pilot_pools']['packs'][arm].items():
            assert file_identity(output/record['path'])==file_identity(parent/packs['candidate' if arm=='baseline' else arm][source]['path'])
    with pytest.raises(FileExistsError):builder.prepare(parent_manifest,output,tokenizer,root,index,reserved,private_manifest)
