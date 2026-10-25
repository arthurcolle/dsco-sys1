import ast
import hashlib
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from sys1.datasets.quality_mixture import (NearIndex, Pool, documents, exposure_weights,
                                          prepare, probe_match, quality, sketch, unescape)
from sys1.model.corpus import DocReader
from sys1.model.corpus_manifest import file_identity, load_manifest


def escaped(text):
    return text.replace('\\','\\\\').replace('\n','\\n').replace('\r','\\r')


def tokenizer():
    result=Tokenizer(models.WordLevel({'[UNK]':0,'[EOS]':1},unk_token='[UNK]'))
    result.pre_tokenizer=pre_tokenizers.Whitespace()
    return result


def test_lossless_code_round_trip(tmp_path):
    original='def render_message(value):\n    """Render a message with an escaped newline suffix."""\n    return str(value) + "\\n"\n'
    text='<lang=python> '+original
    path=tmp_path/'python.txt';path.write_text(escaped(text)+'\n')
    row=next(documents(path,'code'))
    assert row['text']==text and row['legacy']!=text
    ast.parse(original)
    assert quality(row['text'],'code','python')[0] is None
    proper_tree=ast.parse(original);legacy_tree=ast.parse(row['legacy'].split('> ',1)[1])
    assert ast.dump(proper_tree)!=ast.dump(legacy_tree)
    target=tmp_path/'correct.jsonl';target.write_text(json.dumps({'text':text})+'\n')
    # Both backend readers use this same JSONL implementation.
    reader=DocReader([target],'[EOS]');assert reader.next_doc()==text
    assert unescape(escaped('a\\n\\r\\t\n\r\\'))=='a\\n\\r\\t\n\r\\'


@pytest.mark.parametrize('text',[
    'def combine(left, right):\n    return left + right\n',
    'def renamed(x, y):\n    """Different name."""\n    return y + x\n',
])
def test_failed_probe_family_is_reserved(text):
    assert probe_match(text,ast.parse(text))
    assert not probe_match('def subtract(x, y):\n    return x - y\n')


def test_quality_rejects_damage_and_credentials():
    assert quality('<lang=python> '+'def broken(:\n'+'# description '*8,'code','python')[0]=='python_syntax'
    assert quality('repeated words in a loop '*100,'en',None)[0]=='repeated_word_sequence'
    assert quality('Documentation for a service using api_key = "abcdefghijklmnop123456" and other settings.','en',None)[0]=='credential_pattern'


def test_near_duplicates_and_bounded_candidates():
    text=' '.join('token'+str(i) for i in range(150))
    index=NearIndex();index.add(sketch(text))
    assert index.related(sketch(text+' appendix'))
    for _ in range(200):index.add(sketch(text))
    assert index.saturated and max(map(len,index.bands.values()))==128


def test_real_session_reservation_uses_parent_digest(tmp_path):
    path=tmp_path/'real.jsonl'
    path.write_text(json.dumps({'text':'A real trajectory.','session_digest':'heldout-session'})+'\n')
    assert next(documents(path,'agent_real'))['group']=='heldout-session'


def test_exposure_caps_and_infeasible_budget():
    weights,caps=exposure_weights({'en':.8,'agent_real':.2},{'en':1000,'agent_real':10},1000)
    assert sum(weights.values())==pytest.approx(1)
    assert weights['agent_real']==pytest.approx(.01)
    assert weights['agent_real']*1000/10<=1+1e-9
    with pytest.raises(ValueError,match='insufficient'):exposure_weights({'en':1.},{'en':10},21)


def test_finite_reservoir_reproducible_and_no_padding(tmp_path):
    def make(name):
        pool=Pool(tokenizer(),5,4,123);pool.add(' '.join('word'+str(i) for i in range(43)))
        return pool.save(tmp_path/name)
    first,second=make('a.bin'),make('b.bin')
    assert first['sha256']==second['sha256'] and first['rows']==4
    assert first['stream_windows']==8 and first['tail_tokens']==4
    tiny=Pool(tokenizer(),100,10,1);tiny.add('short')
    with pytest.raises(ValueError,match='empty'):tiny.save(tmp_path/'tiny.bin')


def test_builder_preserves_parent_and_validation_priority(tmp_path):
    train='A description of pressure measurements recorded by a reliable scientific instrument in the laboratory.'
    heldout='An account of maritime trade through ancient ports carrying spices and pottery across distant seas.'
    python='def render_message(value):\n    """Render a message with an escaped newline suffix."""\n    return str(value) + "\\n"\n'
    pyval='def squared_magnitude(vector):\n    """Compute the squared magnitude of a coordinate vector."""\n    return sum(item * item for item in vector)\n'
    sources={}
    for name,training,validation in [('en',[train,heldout],[heldout]),('code',['<lang=python> '+python],['<lang=python> '+pyval])]:
        sources[name]={}
        for split,texts in [('train',training),('validation',validation)]:
            path=tmp_path/(name+'-'+split+'.txt');path.write_text(''.join(escaped(t)+'\n' for t in texts))
            sources[name][split]=[{'path':path.name,**file_identity(path)}]
    sources['agent_real']={}
    for split,rows in [('train',[
        {'text':'The operator inspected filesystem capacity and read the monitoring logs before preparing a new release.','session_digest':'heldout-session'},
        {'text':'A scheduler routed the query to a worker and received the completed response after verifying its signature.','session_digest':'training-session'}]),
        ('validation',[{'text':'The repository review checked separate source files and reported a concrete defect in request parsing.','session_digest':'heldout-session'}])]:
        path=tmp_path/('real-'+split+'.jsonl');path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        sources['agent_real'][split]=[{'path':path.name,**file_identity(path)}]
    parent=tmp_path/'parent.json';parent.write_text(json.dumps({'schema':'sys1-corpus-mixture-v1','sources':sources,'objective_weights':{'en':.4,'code':.4,'agent_real':.2}}))
    before=file_identity(parent);tok=tmp_path/'tokenizer.json';tokenizer().save(str(tok))
    result,manifest=prepare(parent,tmp_path/'derived',tok,steps=1,seq=4,accum=1,pool_seq=4,pool_windows=8,eval_windows=4,min_free_bytes=0,max_output_bytes=1024**2)
    plan,_,_=load_manifest(result)
    assert file_identity(parent)==before
    assert set(plan)=={'en','code_python','agent_real'}
    assert manifest['audit']['counts']['train_shared_excluded:validation_exact_overlap']==1
    assert manifest['audit']['counts']['train_shared_excluded:validation_group_overlap']==1
    assert manifest['audit']['counts']['train_lossless_decoding_changed']==1
    assert manifest['row_count']==6
    assert all(Path(result.parent/r['path']).exists() for entries in manifest['validation_reservations']['parent_files'].values() for r in entries)
    assert next(documents(plan['code_python']['train'][0],'code_python'))['text']=='<lang=python> '+python
    with pytest.raises(FileExistsError):prepare(parent,result.parent,tok)
