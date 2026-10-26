#!/usr/bin/env python3
"""Replace a candidate's real-trajectory training source with complete local traces.

References all other immutable shards. Existing selection validation is unchanged;
fresh validation sessions are reserved separately and never used by the pilot.
No training launch, network access, source mutation or external data transfer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sys1.datasets.complete_trajectories import CompleteReader, argument_object, rejection
from sys1.datasets.quality_mixture import (NearIndex, Pool, documents, exposure_weights,
                                         normalized_key, quality, sketch)
from sys1.model.corpus_manifest import file_identity, load_manifest
from sys1.tokenizer.corpus_private import Redactor, render


def write_json(path, value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    path.chmod(0o600)


def write_rows(path, rows):
    with path.open('x') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    path.chmod(0o600)


def reference(record, old_root, new_root):
    return {**record, 'path':os.path.relpath((old_root/record['path']).resolve(),new_root)}


def prepare(parent_path, output, tokenizer_path, chronicle, index_path, reservations_path,
            private_manifest_path, *, max_files=7500, max_read_bytes=1024**3,
            ttl_seconds=300, cap_bytes=32*1024**2, progress=None):
    from tokenizers import Tokenizer
    parent_path, output, tokenizer_path, chronicle, index_path, reservations_path, private_manifest_path = (
        Path(p).resolve() for p in (parent_path, output, tokenizer_path, chronicle, index_path,
                                  reservations_path, private_manifest_path))
    if output.exists():raise FileExistsError('derived dataset already exists')
    if parent_path.parent in output.parents or chronicle==output or chronicle in output.parents:
        raise ValueError('output overlaps immutable parent or chronicle')
    if not 1<=max_files<=10000 or not 1<=ttl_seconds<=900 or not 1024<=cap_bytes<=64*1024**2:
        raise ValueError('invalid export bounds')
    if not 1024<=max_read_bytes<=2*1024**3:raise ValueError('invalid read bound')
    if shutil.disk_usage(output.parent).free<cap_bytes+8*1024**3:raise ValueError('insufficient disk headroom')
    plan,parent_id,parent = load_manifest(parent_path)
    if 'agent_real' not in plan or 'pilot_pools' not in parent:raise ValueError('requires a quality-mixture parent')
    tokenizer_id = file_identity(tokenizer_path)
    if parent['tokenizer_identity']!=tokenizer_id:raise ValueError('tokenizer differs from parent')
    if index_path.stat().st_size>8*1024**2:raise ValueError('oversized session index')
    paths = sorted({Path(line).resolve() for line in index_path.read_text().splitlines() if line.strip()})
    if any((chronicle/'events') not in p.parents or not p.name.startswith('session-') or p.suffix!='.jsonl' for p in paths):
        raise ValueError('index must contain chronicle session files only')
    if not paths or len(paths)>max_files:raise ValueError('index must fit declared file budget')
    inputs = {str(p):file_identity(p) for p in (index_path,reservations_path,private_manifest_path)}
    fixed_inputs = dict(inputs)
    reserved = set(json.loads(reservations_path.read_text())['session_digests'])
    val_exact = set(); val_near = NearIndex(); goals = set()
    prior_reservations = parent['validation_reservations']
    original_validation = prior_reservations['parent_files']
    def reserve_text(text, group=None):
        val_exact.add(normalized_key(text)); val_near.add(sketch(text))
        if group:reserved.add(str(group))
        if '<|user|>' in text:
            goals.add(normalized_key(text.split('<|user|>',1)[1].split('<|end|>',1)[0]))
    for source, records in original_validation.items():
        for record in records:
            path=(parent_path.parent/record['path']).resolve()
            if file_identity(path)!={k:record[k] for k in ('bytes','sha256')}:raise ValueError('original validation drifted')
            fixed_inputs[str(path)]={k:record[k] for k in ('bytes','sha256')}
            for doc in documents(path,source):
                reserve_text(doc['text'],doc['group'] if source=='agent_real' else None)
    private=json.loads(private_manifest_path.read_text())
    from sys1.model.corpus import DocReader
    private_files=[]
    for record in private['files']:
        if not record['name'].startswith('heldout'):continue
        path=private_manifest_path.parent/record['name']
        if file_identity(path)!={k:record[k] for k in ('bytes','sha256')}:raise ValueError('private validation drifted')
        fixed_inputs[str(path)]={k:record[k] for k in ('bytes','sha256')}
        private_files.append(path)
    if not private_files:raise ValueError('private validation missing')
    reader=DocReader(private_files,'<|endoftext|>',repeat=False)
    try:
        while True:reserve_text(reader.next_doc())
    except StopIteration:pass
    finally:reader.close()
    created=time.time(); counts=Counter(); red=Redactor(); render_stats=Counter()
    reader=CompleteReader(chronicle); candidates=[]; snapshot=[]; total=0
    for index,path in enumerate(paths):
        if time.time()-created>=ttl_seconds or reader.bytes_read>=max_read_bytes:
            counts['scan_budget_exhausted']+=1;break
        try:episodes,event_id=reader.session(path)
        except OSError:
            counts['unreadable_event_file']+=1;continue
        if event_id:snapshot.append({'path':str(path.relative_to(chronicle)),**event_id})
        for ep in episodes:
            counts['episodes_seen']+=1
            if isinstance(ep.get('session'),str) and isinstance(ep.get('goal'),str):
                if normalized_key(red(ep['goal'])) in goals:
                    reserved.add(hashlib.sha256(ep['session'].encode()).hexdigest())
            reason=rejection(ep)
            if reason:counts['excluded:'+reason]+=1;continue
            digest=hashlib.sha256(ep['session'].encode()).hexdigest()
            goal=red(ep['goal'])
            if normalized_key(goal) in goals:reserved.add(digest)
            if digest in reserved:counts['excluded:reserved_session']+=1;continue
            text=render(ep,{}, {},set(),red,render_stats).strip().removeprefix('[BOS]').removesuffix('[EOS]')
            reason,_=quality(text,'agent_real',None)
            if reason:counts['excluded:'+reason]+=1;continue
            # Redaction is also required to preserve the argument object's syntax.
            try:
                for step in ep['steps']:
                    argument_object(red(step['args']))
            except ValueError:counts['excluded:redacted_argument_json']+=1;continue
            if '…[+' in text:counts['excluded:inherited_truncation']+=1;continue
            row={'id':hashlib.sha256(text.encode()).hexdigest(),'text':text,'session_digest':digest,
                 'origin':{'source':'chronicle_complete','group':digest,'event_file_sha256':event_id['sha256']}}
            size=len((json.dumps(row,ensure_ascii=False)+'\n').encode())
            if total+size>cap_bytes:
                counts['excluded:output_cap']+=1;continue
            total+=size
            candidates.append(row)
        if progress and (index+1)%250==0:
            progress({'files_scanned':index+1,'candidate_documents':len(candidates),'bytes_read':reader.bytes_read,
                      'elapsed_seconds':round(time.time()-created,1)})
    fresh_reserved=[]; train=[]; seen=set(); train_near=NearIndex()
    for row in candidates:
        if row['session_digest'] in reserved:counts['excluded:late_reserved_session']+=1;continue
        if int(row['session_digest'][:16],16)%10==0:
            fresh_reserved.append(row);reserve_text(row['text'],row['session_digest'])
    for row in candidates:
        if row['session_digest'] in reserved:continue
        text=row['text'];key=normalized_key(text);sig=sketch(text)
        if key in val_exact or val_near.related(sig):counts['excluded:validation_overlap']+=1;continue
        if key in seen or train_near.related(sig):counts['excluded:training_duplicate']+=1;continue
        seen.add(key);train_near.add(sig);train.append(row)
    if not train:raise ValueError('no complete training trajectories after reservations')
    staging=output.with_name('.'+output.name+'.pending-'+str(os.getpid()));staging.mkdir(mode=0o700)
    train_path=staging/'train.jsonl';write_rows(train_path,train)
    write_rows(staging/'fresh-reserved.jsonl',fresh_reserved)
    write_rows(staging/'event-snapshot.jsonl',snapshot)
    write_json(staging/'reserved-sessions.json',{'session_digests':sorted(reserved)})
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    pool=Pool(tokenizer,parent['pilot_pools']['window'],4096,int.from_bytes(hashlib.sha256(b'candidate:agent_real').digest()[:8],'big'))
    for row in train:pool.add(row['text'])
    pool_record=pool.save(staging/'pilot'/'candidate'/'agent_real.bin')
    pool_record['path']=str(Path(pool_record['path']).relative_to(staging))
    token_counts={**parent['exposure_budget']['train_stream_tokens'],'agent_real':pool.tokens}
    budget=parent['exposure_budget'];consumed=budget['steps']*budget['consumed_stream_tokens_per_step']
    weights,caps=exposure_weights(parent['evaluation_weights'],token_counts,consumed,budget['max_base_source_passes'])
    sources={name:{split:[reference(r,parent_path.parent,output) for r in records] for split,records in source.items()}
             for name,source in parent['sources'].items()}
    sources['agent_real']['train']=[{'path':'train.jsonl',**file_identity(train_path)}]
    packs={}
    for arm,old_arm in [('baseline','candidate'),('candidate','candidate'),('validation','validation')]:
        packs[arm]={}
        for name,record in parent['pilot_pools']['packs'][old_arm].items():
            target=staging/'pilot'/arm/(name+'.bin');target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            if arm=='candidate' and name=='agent_real':
                packs[arm][name]=pool_record;continue
            path=(parent_path.parent/record['path']).resolve()
            if file_identity(path)!={k:record[k] for k in ('bytes','sha256')}:raise ValueError('pilot pool drifted')
            shutil.copyfile(path,target);target.chmod(0o600)
            packs[arm][name]={**record,'path':str(target.relative_to(staging))}
    for path,identity in fixed_inputs.items():
        if file_identity(path)!=identity:raise ValueError('input metadata drifted')
    if load_manifest(parent_path)[1]!=parent_id or file_identity(tokenizer_path)!=tokenizer_id:raise ValueError('parent or tokenizer drifted')
    new_budget={**budget,'train_stream_tokens':token_counts,'probability_caps':caps,
                'expected_passes':{k:weights[k]*consumed/token_counts[k] for k in weights}}
    manifest={**parent,'dataset_id':output.name,'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'parent_manifest_sha256':parent_id,'sources':sources,'objective_weights':weights,'exposure_budget':new_budget,
              'phase':'unpromoted candidate; complete hash-verified real trajectories, unchanged selection validation',
              'unit_of_analysis':'complete user-goal/tool-call/result/final trajectory; other parent documents unchanged',
              'license_ref':'inherits parent licenses; new material is user-directed internal use of own DSCO chronicle',
              'provenance':{'parent_manifest_path':str(parent_path),'parent_manifest_sha256':parent_id,
                            'inherited_parent_provenance':parent.get('provenance'),
                            'new_source':'local owner DSCO chronicle; no external downloads or transfers'},
              'validation_reservations':{**prior_reservations,**reference(prior_reservations,parent_path.parent,output),
                                         'parent_files':{k:[reference(r,parent_path.parent,output) for r in records] for k,records in original_validation.items()}},
              'pilot_pools':{**parent['pilot_pools'],'baseline_weights':parent['objective_weights'],'packs':packs,
                             'baseline_sanitation':'unchanged foundation-quality parent; only real training source replaced'},
              'complete_trajectory_export':{'input_identities':inputs,'bounds':{'max_files':max_files,'max_read_bytes':max_read_bytes,
                    'ttl_seconds':ttl_seconds,'cap_bytes':cap_bytes,'max_blob_bytes':reader.max_blob_bytes},
                    'budget_semantics':'encoded candidate rows capped in bytes; read and time limits checked between bounded sessions',
                    'counts':dict(counts),'reader_counts':dict(reader.counts),'bytes_read':reader.bytes_read,
                    'training_documents':len(train),'training_sessions':len({r['session_digest'] for r in train}),
                    'training_stream_tokens':pool.tokens,'fresh_reserved_documents':len(fresh_reserved),
                    'fresh_reserved_used_for_selection':False,'selection_validation_unchanged':True,'redactions':dict(red.counts),
                    'transform_identities':{str(p.relative_to(REPO)):file_identity(p) for p in [Path(__file__),
                        REPO/'sys1/datasets/complete_trajectories.py',REPO/'sys1/datasets/export_episodes.py',
                        REPO/'sys1/datasets/quality_mixture.py',REPO/'sys1/tokenizer/corpus_private.py',
                        REPO/'sys1/model/corpus.py',REPO/'sys1/model/corpus_manifest.py']},
                    'event_snapshot':{'path':'event-snapshot.jsonl',**file_identity(staging/'event-snapshot.jsonl')},
                    'reserved_sessions':{'path':'reserved-sessions.json',**file_identity(staging/'reserved-sessions.json')},
                    'fresh_reserved':{'path':'fresh-reserved.jsonl',**file_identity(staging/'fresh-reserved.jsonl')},
                    'limitations':['completion and tool status do not establish task correctness','bounded index is not all local history',
                                   'near-duplicate sketches can miss semantic overlap','inherited licensing and attribution limits remain']},
              'audit':{'inherited_parent_audit':parent['audit'],'sources':{**parent['audit']['sources'],
                       'agent_real':{'train_documents':len(train),'validation_documents':parent['audit']['sources']['agent_real']['validation_documents']}},
                       'raw_examples_in_report':False}}
    manifest['row_count']=sum(v['train_documents']+v['validation_documents'] for v in manifest['audit']['sources'].values())
    manifest['source_digest']=hashlib.sha256(json.dumps({'parent':parent_id,'inputs':fixed_inputs,
         'event_snapshot':manifest['complete_trajectory_export']['event_snapshot']},sort_keys=True).encode()).hexdigest()
    manifest['transform_sha256']=file_identity(Path(__file__))['sha256']
    manifest['transform_dag']={'parent_manifest_sha256':parent_id,'tokenizer_identity':tokenizer_id,
                               'transforms':manifest['complete_trajectory_export']['transform_identities'],
                               'inputs':fixed_inputs,'event_snapshot':manifest['complete_trajectory_export']['event_snapshot']}
    manifest['transform_dag_digest']=hashlib.sha256(json.dumps(manifest['transform_dag'],sort_keys=True).encode()).hexdigest()
    write_json(staging/'MANIFEST.json',manifest)
    load_manifest(staging/'MANIFEST.json')
    staging.rename(output)
    return output/'MANIFEST.json',manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--session-index',type=Path,required=True)
    parser.add_argument('--chronicle',type=Path,default=Path.home()/'.dsco/chronicle')
    parser.add_argument('--tokenizer',type=Path,default=REPO/'data/tokenizer-100k/tokenizer.json')
    parser.add_argument('--prior-reservations',type=Path,default=REPO/'data/mixtures/agent-plus-real-v4/reserved-sessions.json')
    parser.add_argument('--private-manifest',type=Path,default=REPO/'data/corpus/private/MANIFEST.json')
    args=parser.parse_args()
    path,manifest=prepare(args.parent_manifest,args.output,args.tokenizer,args.chronicle,args.session_index,
                          args.prior_reservations,args.private_manifest,progress=lambda x:print(json.dumps(x),flush=True))
    export=manifest['complete_trajectory_export']
    print(json.dumps({'manifest':str(path),'documents':export['training_documents'],'sessions':export['training_sessions'],
                      'tokens':export['training_stream_tokens'],'counts':export['counts'],'launch_performed':False},indent=2))


if __name__=='__main__':main()
