import copy
import hashlib
import json
from pathlib import Path
import random
import string

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from sys1.datasets import trajectory_windows as tw
from sys1.model.corpus_manifest import file_identity


def tokenizer():
    vocab={char:index for index,char in enumerate(string.printable)}
    vocab.update({'[UNK]':len(vocab),'[EOS]':len(vocab)+1})
    tok=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'));tok.pre_tokenizer=pre_tokenizers.Split('',behavior='isolated')
    tok.add_special_tokens(['[EOS]','<|user|>','<|assistant|>','<|tool|>','<|end|>','<|json|>','<|/json|>','[ACT]'])
    return tok


def row(seed):
    rng=random.Random(seed)
    def text(n):return ''.join(rng.choices(string.ascii_letters+' ',k=n))
    body='<|user|>'+text(80)+'<|end|>\n<|assistant|>[ACT]read_file<|json|>'+json.dumps({'path':text(50)})+'<|/json|><|end|>\n<|tool|>'+text(160)+'<|end|>\n<|assistant|>'+text(200)+'<|end|>'
    return {'text':body,'id':hashlib.sha256(body.encode()).hexdigest(),'session_digest':hashlib.sha256(str(seed).encode()).hexdigest()}


def fixture_dataset(root):
    root.mkdir();tok=root/'tokenizer.json';tokenizer().save(str(tok));sources={'agent_real':{}}
    for split,seeds in [('train',[1,2,3]),('validation',[10,11])]:
        path=root/(split+'.jsonl');path.write_text(''.join(json.dumps(row(seed))+'\n' for seed in seeds))
        sources['agent_real'][split]=[{'path':path.name,**file_identity(path)}]
    manifest=root/'MANIFEST.json';manifest.write_text(json.dumps({'schema':'sys1-corpus-mixture-v1','sources':sources,
        'tokenizer_identity':file_identity(tok),'license_ref':'synthetic unit-test fixture'}))
    return manifest,tok


def test_role_spans_are_contiguous_and_targets_never_include_padding():
    source=row(1);tok=tokenizer();items,tokens,bad=tw.windows(source,tok,32,8)
    original=[tok.token_to_id('[EOS]')]+tok.encode(source['text'],add_special_tokens=False).ids+[tok.token_to_id('[EOS]')]
    assert bad==0 and set(tokens)=={1,2,3,4}
    assert {item['target_role'] for item in items}=={'assistant_call','assistant_final'}
    for item in items:
        n=item['valid_tokens']+1
        assert item['ids'][:n]==original[item['start_token']:item['stop_token']]
        assert not item['valid'][0] and not item['target'][0]
        assert all(t<=v for t,v in zip(item['target'],item['valid']))
        assert 1<=sum(item['target'])<=8
        assert {r for r,t in zip(item['roles'],item['target']) if t}=={tw.ROLES[item['target_role']]}


def test_legacy_argument_damage_is_counted_without_changing_validation_text():
    text=row(1)['text'].replace('"path":','"path"')
    with pytest.raises(ValueError,match='argument'):tw.parse(text)
    _,bad=tw.parse(text,strict_arguments=False)
    assert bad==1
    with pytest.raises(ValueError):tw.parse(text.replace('<|tool|>','<|user|>'))


def test_hierarchical_caps_preserve_roles_and_ignore_input_order():
    rows=[]
    for role in ('assistant_call','assistant_final'):
        for task in ('execution','editing','inspection','other'):
            for i in range(10):rows.append({'session':'s','target_role':role,'task_stratum':task,'item_id':f'{role}-{task}-{i:02}'})
    selected=tw.balanced(rows,2,4)
    assert {r['target_role'] for r in selected}=={'assistant_call','assistant_final'}
    assert selected==tw.balanced(list(reversed(rows)),2,4)
    incremental=[]
    for chunk in [rows[:17],rows[17:39],rows[39:]]:incremental=tw.balanced([*incremental,*chunk],None,4)
    assert tw.balanced(incremental,8,4)==tw.balanced(rows,8,4)


def test_session_balanced_schedule_visits_sessions_before_replaying():
    items=[{'session':str(i),'target_role':role} for i in range(8) for role in ('assistant_call','assistant_final')]
    schedule=tw.training_schedule(items,8,123)
    assert len({items[i]['session'] for i in schedule})==8
    assert [items[i]['target_role'] for i in schedule]==['assistant_call','assistant_final']*4
    assert schedule==tw.training_schedule(items,8,123)


def test_versioned_window_build_verifies_masks_hashes_and_session_splits(tmp_path):
    parent,tok=fixture_dataset(tmp_path/'source');before=file_identity(parent)
    path,manifest=tw.prepare(parent,tmp_path/'windows',tok,seq=32,target_chunk=8)
    parts,_,loaded=tw.load(path)
    assert file_identity(parent)==before and loaded==manifest
    assert manifest['coverage']['train']['maximum_windows_per_session']<=8
    assert manifest['coverage']['validation']['retained_sessions']==2
    assert not manifest['fresh_reserved_used'] and not manifest['final_test_used']
    assert not ({r['session'] for r in parts['train']['items']}&{r['session'] for r in parts['validation']['items']})
    with pytest.raises(FileExistsError):tw.prepare(parent,path.parent,tok)
    record=manifest['splits']['train']['target'];target=path.parent/record['path']
    payload=bytearray(target.read_bytes());payload[0]=1;target.write_bytes(payload)
    with pytest.raises(ValueError,match='identity'):tw.load(path)
    # Hashes alone must not admit a semantically invalid mask.
    record.update(file_identity(target));path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='target mask'):tw.load(path)


def test_same_session_cannot_enter_both_window_splits(tmp_path):
    parent,tok=fixture_dataset(tmp_path/'source');manifest=json.loads(parent.read_text())
    path=parent.parent/'validation.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['session_digest']=row(1)['session_digest'];path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest['sources']['agent_real']['validation'][0].update(file_identity(path));parent.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='session split overlap'):tw.prepare(parent,tmp_path/'bad',tok,seq=32,target_chunk=8)
