"""Versioned foundation corpus: lossless decoding, quality gates and bounded replay.

No remote execution or training-job control. The parent remains immutable.
"""
from __future__ import annotations

from array import array
import ast
from collections import Counter, defaultdict
import datetime as dt
import gzip
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
import warnings

from sys1.datasets.export_episodes import SECRET_PATTERNS
from sys1.model.corpus_manifest import file_identity, load_manifest

SYNTHETIC = {'agent_opcodes', 'agent_decisions', 'agent_transcripts', 'agent_schemas'}
LANGUAGE_GROUP = {**dict.fromkeys(('c','cpp','go','rust'),'code_native'),
                  **dict.fromkeys(('javascript','typescript'),'code_web'),
                  **dict.fromkeys(('json','yaml','toml'),'code_config'),
                  'python':'code_python','shell':'code_shell','markdown':'code_docs'}
PREFERRED = {'en':.50,'code_python':.15,'code_native':.22,'code_web':.08,
             'code_shell':.02,'code_docs':.01,'code_config':.01,
             'agent_real':.003,'systems_native':.005,'systems_otp':.002}
PROBE = re.compile(r'def\s+\w+\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*:\s*return\s+(\w+)\s*\+\s*(\w+)\b')


def unescape(text):
    """Inverse of corpus_en/code escape(), including literal backslash-n."""
    out=[];i=0
    while i<len(text):
        if text[i]=='\\' and i+1<len(text):
            following=text[i+1]
            out.append({'n':'\n','r':'\r','\\':'\\'}.get(following,'\\'+following));i+=2
        else:out.append(text[i]);i+=1
    return ''.join(out)


def documents(path, source):
    opener=gzip.open if path.suffix=='.gz' else open
    jsonl=path.name.removesuffix('.gz').endswith('.jsonl')
    with opener(path,'rt',encoding='utf-8',errors='strict') as stream:
        for number,line in enumerate(stream,1):
            if not line.strip():continue
            if len(line)>4*1024**2:raise ValueError('oversized parent document')
            group=None;language=None
            if jsonl:
                row=json.loads(line)
                if not isinstance(row,dict) or not isinstance(row.get('text'),str):
                    raise ValueError('parent JSONL requires a text string')
                proper=legacy=row['text'];group=row.get('component') or row.get('session_digest') or row.get('session')
            else:
                raw=line.rstrip('\r\n')
                legacy=raw.replace('\\n','\n').removeprefix('[BOS]').removesuffix('[EOS]')
                proper=unescape(raw) if source in {'en','code'} else legacy
            if source=='code':
                match=re.match(r'<lang=([a-z_]+)> ',proper)
                if not match or match[1] not in LANGUAGE_GROUP:raise ValueError('unknown parent code language')
                language=match[1]
            yield {'text':proper,'legacy':legacy,'line':number,'group':group,'language':language}


def body(text):
    return re.sub(r'^<lang=[a-z_]+> ','',text,count=1)


def normalized_key(text):
    return hashlib.sha256(' '.join(body(text).split()).encode()).hexdigest()


def probe_match(text, tree=None):
    # Reserve the renamed two-argument addition family; do not inject its answer.
    for match in PROBE.finditer(text):
        if {match[1],match[2]}=={match[3],match[4]}:return True
    if tree is not None:
        for node in ast.walk(tree):
            if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
            statements=[s for s in node.body if not (isinstance(s,ast.Expr) and isinstance(s.value,ast.Constant) and isinstance(s.value.value,str))]
            if len(node.args.args)!=2 or len(statements)!=1 or not isinstance(statements[0],ast.Return):continue
            value=statements[0].value
            if isinstance(value,ast.BinOp) and isinstance(value.op,ast.Add) and isinstance(value.left,ast.Name) and isinstance(value.right,ast.Name):
                if {a.arg for a in node.args.args}=={value.left.id,value.right.id}:return True
    return False


def sketch(text):
    words=re.findall(r'\w+',body(text))
    if len(words)<32:return ()
    step=max(1,(len(words)-3)//8192)
    hashes={hashlib.blake2b(' '.join(words[i:i+4]).encode(),digest_size=8).digest()
            for i in range(0,len(words)-3,step)}
    return tuple(heapq.nsmallest(8,hashes)) if len(hashes)>=8 else ()


class NearIndex:
    """Compact bottom-eight shingle sketches; heuristic, not semantic dedup."""
    def __init__(self):self.rows=[];self.bands=defaultdict(list);self.saturated=set()
    def related(self,value):
        if not value:return False
        candidates=set()
        for i in range(4):candidates.update(self.bands.get((i,value[2*i:2*i+2]),()))
        return any(len(set(value)&set(self.rows[index]))>=6 for index in candidates)
    def add(self,value):
        if not value:return
        index=len(self.rows);self.rows.append(value)
        for i in range(4):
            key=(i,value[2*i:2*i+2])
            if len(self.bands[key])<128:self.bands[key].append(index)
            else:self.saturated.add(key)


def quality(text,source,language):
    if '\x00' in text:return 'nul',None
    if not 64<=len(text)<=200_000:return 'length',None
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):return 'credential_pattern',None
    words=re.findall(r'\w+',text)
    if len(words)>=32:
        grams=Counter(zip(words,words[1:],words[2:],words[3:]))
        if max(grams.values())*4/len(words)>.40:return 'repeated_word_sequence',None
    lines=[line.strip() for line in text.splitlines() if len(line.strip())>=16]
    if lines:
        counts=Counter(lines);total=sum(map(len,lines))
        if sum((n-1)*len(line) for line,n in counts.items())/total>.50:return 'repeated_lines',None
    tree=None
    if language=='python':
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',SyntaxWarning);tree=ast.parse(body(text))
        except (SyntaxError,ValueError,RecursionError):return 'python_syntax',None
        if not tree.body:return 'python_no_code',None
    if source=='en' and sum(c.isalpha() for c in text)/len(text)<.40:return 'low_prose_content',None
    if probe_match(text,tree):return 'reserved_probe_family',tree
    return None,tree


class Pool:
    """Seeded full-stream window reservoir; no repeated validation padding."""
    def __init__(self,tokenizer,window,count,seed):
        self.tokenizer,self.window,self.count=tokenizer,window,count
        self.eos=tokenizer.token_to_id('[EOS]');self.rng=random.Random(seed)
        if self.eos is None:raise ValueError('tokenizer requires EOS')
        self.pending=[];self.characters=0;self.buffer=[];self.rows=[];self.tokens=0;self.seen=0
    def add(self,text):
        self.pending.append(text);self.characters+=len(text)
        if len(self.pending)>=16 or self.characters>=1024**2:self.flush()
    def flush(self):
        if not self.pending:return
        for encoded in self.tokenizer.encode_batch(self.pending,add_special_tokens=False):
            ids=encoded.ids+[self.eos];self.tokens+=len(ids);self.buffer.extend(ids)
            cursor=0
            while cursor+self.window<=len(self.buffer):
                self.seen+=1;slot=self.seen-1 if self.seen<=self.count else self.rng.randrange(self.seen)
                if slot<self.count:
                    value=array('I',self.buffer[cursor:cursor+self.window])
                    if slot==len(self.rows):self.rows.append(value)
                    else:self.rows[slot]=value
                cursor+=self.window
            self.buffer=self.buffer[cursor:]
        self.pending=[];self.characters=0
    def save(self,path):
        self.flush()
        if not self.rows:raise ValueError('empty pilot source')
        path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        with path.open('xb') as stream:
            for row in self.rows:
                if sys.byteorder!='little':row=row[:];row.byteswap()
                row.tofile(stream)
        path.chmod(0o600)
        return {'path':str(path),'rows':len(self.rows),'window':self.window,'dtype':'uint32-le',
                'stream_tokens':self.tokens,'stream_windows':self.seen,'tail_tokens':len(self.buffer),**file_identity(path)}


def exposure_weights(preferred,tokens,consumed_tokens,max_passes=2.):
    if consumed_tokens<=0 or max_passes<=0 or set(preferred)!=set(tokens) or any(v<=0 for v in preferred.values()) or any(v<=0 for v in tokens.values()):
        raise ValueError('invalid exposure policy or empty source')
    caps={name:min(1.,tokens[name]*(1. if name in {'agent_real','systems_native','systems_otp'} else max_passes)/consumed_tokens) for name in preferred}
    if math.fsum(caps.values())<1-1e-12:raise ValueError('insufficient unique tokens for the declared replay budget')
    low,high=0.,1.
    while sum(min(caps[k],high*preferred[k]) for k in caps)<1-1e-14:high*=2
    for _ in range(80):
        mid=(low+high)/2
        if sum(min(caps[k],mid*preferred[k]) for k in caps)<1:low=mid
        else:high=mid
    weights={k:min(caps[k],(low+high)/2*preferred[k]) for k in caps}
    return weights,caps


class Writer:
    def __init__(self,root,cap):self.root=root;self.cap=cap;self.bytes=0;self.files=defaultdict(list);self.handles={};self.sizes={}
    def add(self,source,split,row):
        payload=(json.dumps(row,ensure_ascii=False)+'\n').encode()
        if self.bytes+len(payload)>self.cap:raise ValueError('derived corpus byte budget exceeded')
        key=(source,split)
        if key not in self.handles or self.sizes[key]+len(payload)>64*1024**2:
            if key in self.handles:self.handles[key].close()
            folder=self.root/source;folder.mkdir(exist_ok=True,mode=0o700)
            path=folder/(split+'-'+str(len(self.files[key])).zfill(5)+'.jsonl')
            self.files[key].append(path);self.handles[key]=path.open('xb');path.chmod(0o600);self.sizes[key]=0
        self.handles[key].write(payload);self.sizes[key]+=len(payload);self.bytes+=len(payload)
    def close(self):
        for stream in self.handles.values():stream.close()


def prepare(parent_path,output,tokenizer_path,*,steps=30000,seq=1024,micro_batch=1,accum=16,
            max_passes=2.,pool_windows=4096,eval_windows=512,pool_seq=64,max_output_bytes=1536*1024**2,min_free_bytes=16*1024**3,progress=None,source_registers=()):
    from tokenizers import Tokenizer
    parent_path,output,tokenizer_path=map(lambda p:Path(p).resolve(),(parent_path,output,tokenizer_path))
    if output.exists():raise FileExistsError('derived dataset already exists')
    plan,parent_id,parent=load_manifest(parent_path)
    if not {'en','code'}<=set(plan) or not set(plan)<=({'en','code','agent_real','systems_native','systems_otp'}|SYNTHETIC):raise ValueError('unsupported parent sources')
    if any(output==p or p in output.parents or output in p.parents for source in plan.values() for paths in source.values() for p in paths):raise ValueError('output overlaps parent inputs')
    if min(steps,seq,micro_batch,accum,pool_windows,eval_windows,pool_seq)<1:raise ValueError('positive data budgets required')
    if shutil.disk_usage(output.parent).free<max_output_bytes+min_free_bytes:raise ValueError('insufficient disk headroom')
    tokenizer=Tokenizer.from_file(str(tokenizer_path));tokenizer_id=file_identity(tokenizer_path)
    registers=[]
    for register in source_registers:
        register=Path(register).resolve()
        metadata=json.loads(register.read_text())
        registers.append({'path':str(register),**file_identity(register),'declared_provenance':metadata})
    if parent.get('tokenizer_identity',{}).get('sha256',tokenizer_id['sha256'])!=tokenizer_id['sha256']:raise ValueError('parent tokenizer differs')
    staging=output.with_name('.'+output.name+'.pending-'+str(os.getpid()))
    staging.mkdir(mode=0o700)
    writer=Writer(staging,max_output_bytes)
    counts=Counter();per_source=defaultdict(Counter);val_exact=set();val_near=NearIndex();seen_val=set();train_exact=set();train_near=NearIndex();val_groups=set()
    candidate_pools={};baseline_pools={};validation_pools={};created=time.time()
    def pool(target,name,split):
        if name not in target:
            seed=int.from_bytes(hashlib.sha256((split+':'+name).encode()).digest()[:8],'big')
            target[name]=Pool(tokenizer,pool_seq+1,eval_windows if split=='validation' else pool_windows,seed)
        return target[name]
    def group(source,doc):return LANGUAGE_GROUP[doc['language']] if source=='code' else source
    def notify(phase):
        if progress:progress({'phase':phase,'counts':dict(counts),'output_bytes':writer.bytes,'elapsed_seconds':round(time.time()-created,1)})
    reservations=staging/'validation-reservations.jsonl'
    try:
        with reservations.open('x') as reserved:
            for source,entry in plan.items():
                for path in entry['validation']:
                    file_id=file_identity(path)
                    for doc in documents(path,source):
                        counts['validation_parent_documents']+=1
                        text=doc['text'];key=normalized_key(text);val_exact.add(key)
                        if doc['group']:val_groups.add((source,str(doc['group'])))
                        if source not in SYNTHETIC:val_near.add(sketch(text))
                        reason,_=quality(text,source,doc['language'])
                        if source in SYNTHETIC:reason='tokenizer_only_source'
                        if key in seen_val:reason=reason or 'normalized_duplicate'
                        reserved.write(json.dumps({'source':source,'parent_file_sha256':file_id['sha256'],'parent_line':doc['line'],'decoded_sha256':hashlib.sha256(text.encode()).hexdigest(),'selection_exclusion':reason})+'\n')
                        if reason:
                            counts['validation_excluded:'+reason]+=1;continue
                        seen_val.add(key);name=group(source,doc)
                        row={'id':hashlib.sha256(text.encode()).hexdigest(),'text':text,'origin':{'source':source,'parent_file_sha256':file_id['sha256'],'parent_line':doc['line'],'group':doc['group']}}
                        writer.add(name,'validation',row);pool(validation_pools,name,'validation').add(text)
                        counts['validation_selected']+=1;per_source[name]['validation_documents']+=1
        reservations.chmod(0o600);notify('validation_reserved')
        for source,entry in plan.items():
            for path in entry['train']:
                file_id=file_identity(path)
                for doc in documents(path,source):
                    counts['train_parent_documents']+=1
                    if counts['train_parent_documents']%10000==0:notify('training_filtered')
                    text=doc['text'];key=normalized_key(text);signature=sketch(text)
                    reason,tree=quality(text,source,doc['language'])
                    shared=None
                    if any(p.search(text) for p in SECRET_PATTERNS):shared='credential_pattern'
                    elif probe_match(text,tree):shared='reserved_probe_family'
                    elif key in val_exact:shared='validation_exact_overlap'
                    elif (source,str(doc['group'])) in val_groups:shared='validation_group_overlap'
                    elif val_near.related(signature):shared='validation_near_overlap'
                    if shared:
                        counts['train_shared_excluded:'+shared]+=1;continue
                    pool(baseline_pools,source,'baseline').add(doc['legacy'])
                    if source in SYNTHETIC:reason='tokenizer_only_source'
                    if reason:
                        counts['train_excluded:'+reason]+=1;continue
                    if key in train_exact:counts['train_excluded:normalized_duplicate']+=1;continue
                    if train_near.related(signature):counts['train_excluded:near_duplicate']+=1;continue
                    train_exact.add(key);train_near.add(signature);name=group(source,doc)
                    if doc['legacy']!=text:counts['train_lossless_decoding_changed']+=1
                    if doc['language']=='python':
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore',SyntaxWarning)
                            try:ast.parse(body(doc['legacy']))
                            except (SyntaxError,ValueError,RecursionError):counts['retained_python_repaired_by_decoding']+=1
                    row={'id':hashlib.sha256(text.encode()).hexdigest(),'text':text,'origin':{'source':source,'parent_file_sha256':file_id['sha256'],'parent_line':doc['line'],'group':doc['group']}}
                    writer.add(name,'train',row);pool(candidate_pools,name,'candidate').add(text)
                    counts['train_selected']+=1;per_source[name]['train_documents']+=1
        writer.close()
        preferred={name:PREFERRED[name] for name in candidate_pools}
        if set(preferred)!=set(validation_pools):raise ValueError('candidate training/validation coverage differs')
        preferred={k:v/sum(preferred.values()) for k,v in preferred.items()}
        packs={}
        for arm,pools in [('baseline',baseline_pools),('candidate',candidate_pools),('validation',validation_pools)]:
            packs[arm]={}
            for name,stream in pools.items():
                info=stream.save(staging/'pilot'/arm/(name+'.bin'))
                info['path']=str(Path(info['path']).relative_to(staging));packs[arm][name]=info
        consumed=steps*micro_batch*accum*(seq+1)
        token_counts={name:stream.tokens for name,stream in candidate_pools.items()}
        weights,caps=exposure_weights(preferred,token_counts,consumed,max_passes)
        if set(baseline_pools)!=set(parent['objective_weights']):raise ValueError('baseline source coverage differs')
        if not math.isclose(sum(parent['objective_weights'].values()),1.,abs_tol=1e-9):raise ValueError('baseline probabilities do not sum to one')
        sources={name:{split:[{'path':str(path.relative_to(staging)),**file_identity(path)} for path in writer.files[name,split]] for split in ('train','validation')} for name in preferred}
        if load_manifest(parent_path)[1]!=parent_id:raise ValueError('parent drifted during preparation')
        if file_identity(tokenizer_path)!=tokenizer_id:raise ValueError('tokenizer drifted during preparation')
        for register in registers:
            if file_identity(register['path'])!={k:register[k] for k in ('bytes','sha256')}:raise ValueError('source register drifted')
        transform_id=file_identity(Path(__file__))['sha256']
        dag={'parent_manifest':parent_id,'tokenizer':tokenizer_id['sha256'],'transform':transform_id,'source_registers':[r['sha256'] for r in registers]}
        manifest={'schema':'sys1-corpus-mixture-v1','dataset_id':output.name,'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                  'schema_version':1,'source_digest':parent_id,'row_count':counts['train_selected']+counts['validation_selected'],
                  'transform_dag_digest':hashlib.sha256(json.dumps(dag,sort_keys=True).encode()).hexdigest(),
                  'privacy':'PRIVATE: training documents, source metadata and pilot pools must not be committed',
                  'unit_of_analysis':'whole parent document; original heldout and repository-component reservations retained',
                  'parent_manifest_sha256':parent_id,'tokenizer_identity':tokenizer_id,'transform_sha256':transform_id,
                  'sources':sources,'objective_weights':weights,'evaluation_weights':preferred,
                  'phase':'foundation LM; tokenizer-only synthetic fixtures quarantined from training and selection',
                  'license_ref':'inherits parent source licenses and user-directed internal-use basis; no new external data',
                  'provenance':{'parent_path':str(parent_path),'repositories':parent.get('repositories',{}),'parent_private_license_ref':parent.get('license_ref'),'no_external_downloads':True,'source_registers':registers,'transform_dag':dag,
                                'limitations':['upstream per-document attribution not reconstructed where absent from parent','upstream permissions are inherited declarations, not newly verified']},
                  'split_policy':'all original validation documents reserved; quality-selected validation; train-only exact, shingle-near and probe-family exclusion; no final test feedback',
                  'validation_reservations':{'path':reservations.name,**file_identity(reservations),'parent_files':{name:[{'path':os.path.relpath(path,output),**file_identity(path)} for path in entry['validation']] for name,entry in plan.items()}},
                  'audit':{'counts':dict(counts),'sources':{k:dict(v) for k,v in per_source.items()},'raw_examples_in_report':False,
                           'near_duplicate_policy':{'sketch':'bottom eight word four-gram hashes','minimum_intersection':6,'band_candidates_cap':128,'saturated_validation_bands':len(val_near.saturated),'saturated_train_bands':len(train_near.saturated),'limitation':'heuristic; capped common bands can miss related documents'},
                           'unmeasured':['semantic correctness of arbitrary source files','all semantic near-duplicates','unknown benchmark contamination']},
                  'exposure_budget':{'steps':steps,'predicted_tokens_per_step':seq*micro_batch*accum,'consumed_stream_tokens_per_step':(seq+1)*micro_batch*accum,
                                     'max_base_source_passes':max_passes,'max_private_source_passes':1,'train_stream_tokens':token_counts,'probability_caps':caps,
                                     'expected_passes':{k:weights[k]*consumed/token_counts[k] for k in weights}},
                  'pilot_pools':{'window':pool_seq+1,'seed_policy':'SHA256(arm:source); full-pass window reservoir','baseline_weights':parent['objective_weights'],
                                 'validation_weights':preferred,'packs':packs,'baseline_sanitation':'same credential, reserved-probe and validation-overlap exclusions; legacy decoding and tokenizer fixtures retained'}}
        pending=staging/'MANIFEST.pending.json'
        pending.write_text(json.dumps(manifest,indent=2)+'\n');pending.chmod(0o600)
        load_manifest(pending);pending.rename(staging/'MANIFEST.json');staging.rename(output)
        notify('published')
        return output/'MANIFEST.json',manifest
    finally:writer.close()
