from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from sys1.model.corpus_manifest import file_identity,load_manifest
from sys1.tokenizer.corpus_private import Redactor,render
from sys1.datasets.complete_trajectories import CompleteReader,argument_object
from test_complete_trajectories import events,session_file


@pytest.fixture
def repair():
    path=Path(__file__).resolve().parents[1]/'scripts/reconstruct_trajectory_selection.py'
    spec=importlib.util.spec_from_file_location('reconstruct_selection',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixture_source(tmp_path):
    chronicle=tmp_path/'chronicle';event=session_file(chronicle,events(chronicle,'reserved-session',long=True))
    ep=CompleteReader(chronicle).session(event)[0][0]
    ep['steps'][0]['args']='{"path":"clipped…[+1000 bytes]'
    text=render(ep,{}, {},set(),Redactor(),Counter()).strip().removeprefix('[BOS]').removesuffix('[EOS]')
    root=tmp_path/'source';root.mkdir()
    legacy=root/'heldout.jsonl';legacy.write_text(json.dumps({'text':text,'session_digest':hashlib.sha256(b'reserved-session').hexdigest(),
                                                           'event_file_sha256':file_identity(event)['sha256']})+'\n')
    train=root/'train.jsonl';train.write_text(json.dumps({'text':'unchanged private training fixture'})+'\n')
    snapshot=root/'events.jsonl';snapshot.write_text(json.dumps({'path':str(event.relative_to(chronicle)),**file_identity(event)})+'\n')
    reserved=root/'reserved.json';reserved.write_text(json.dumps({'session_digests':[hashlib.sha256(b'reserved-session').hexdigest()]}))
    fresh=root/'fresh.jsonl';fresh.write_text('')
    validation_record={'path':legacy.name,**file_identity(legacy)}
    manifest={'schema':'sys1-corpus-mixture-v1','sources':{'agent_real':{'train':[{'path':train.name,**file_identity(train)}],'validation':[validation_record]}},
              'validation_reservations':{'path':reserved.name,**file_identity(reserved),'parent_files':{'agent_real':[validation_record]}},
              'complete_trajectory_export':{name:{'path':path.name,**file_identity(path)} for name,path in [('event_snapshot',snapshot),('reserved_sessions',reserved),('fresh_reserved',fresh)]},
              'audit':{'sources':{'agent_real':{'train_documents':1,'validation_documents':1}}}}
    path=root/'MANIFEST.json';path.write_text(json.dumps(manifest))
    return path,chronicle,event


def test_structured_redaction_preserves_json_escapes(repair):
    raw=json.dumps({'value':chr(0xABCD)+'Abc123'*12})
    red=Redactor()
    with pytest.raises(ValueError):argument_object(red(raw))
    repaired,changed=repair.safe_argument(raw,Redactor())
    assert changed and isinstance(argument_object(repaired),dict)
    assert 'Abc123'*12 not in repaired


def test_redacted_argument_keys_cannot_silently_merge(repair):
    with pytest.raises(ValueError,match='merged argument keys'):
        repair.safe_argument(json.dumps({'a@example.com':1,'b@example.com':2}),Redactor())


def test_same_heldout_episode_reconstruction_preserves_training_and_parent(tmp_path,repair):
    source,chronicle,_=fixture_source(tmp_path);before=file_identity(source)
    old,_,_=load_manifest(source)
    path,manifest=repair.reconstruct(source,tmp_path/'repaired',chronicle)
    plan,_,_=load_manifest(path)
    assert file_identity(source)==before
    assert plan['agent_real']['train']==old['agent_real']['train']
    assert manifest['selection_reconstruction']['complete']
    assert manifest['selection_reconstruction']['reconstructed_sessions']==1
    result=json.loads(plan['agent_real']['validation'][0].read_text())
    assert 'clipped…[+' not in result['text']
    assert result['session_digest']==hashlib.sha256(b'reserved-session').hexdigest()
    assert not manifest['selection_reconstruction']['fresh_reserved_used']
    with pytest.raises(FileExistsError):repair.reconstruct(source,path.parent,chronicle)


def test_changed_pinned_event_is_not_silently_replaced(tmp_path,repair):
    source,chronicle,event=fixture_source(tmp_path)
    event.write_text(event.read_text()+'\n')
    import os,time
    os.utime(event,(time.time()-120,)*2)
    with pytest.raises(ValueError,match='pinned selection event changed'):repair.reconstruct(source,tmp_path/'bad',chronicle)
