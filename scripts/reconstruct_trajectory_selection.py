#!/usr/bin/env python3
"""Repair existing selection traces from pinned local events, preserving their split.

Never consumes fresh reserved/final data. Missing or ambiguous matches are recorded
and prevent the downstream quality gate from passing. No raw samples in reports.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from sys1.datasets.complete_trajectories import CompleteReader, argument_object, rejection
from sys1.datasets.trajectory_windows import write_json
from sys1.model.corpus_manifest import file_identity,load_manifest
from sys1.tokenizer.corpus_private import Redactor,render


def safe_argument(text, redactor):
    redacted=redactor(text)
    try:argument_object(redacted);return redacted,False
    except ValueError:pass
    def visit(value):
        if isinstance(value,str):return redactor(value)
        if isinstance(value,list):return [visit(v) for v in value]
        if isinstance(value,dict):
            result={}
            for key,item in value.items():
                key=redactor(key)
                if key in result:raise ValueError('redaction merged argument keys')
                result[key]=visit(item)
            return result
        return value
    result=json.dumps(visit(argument_object(text)),ensure_ascii=False)
    argument_object(result)
    return result,True


def reconstruct(parent_path,output,chronicle):
    parent_path,output,chronicle=map(lambda p:Path(p).resolve(),(parent_path,output,chronicle))
    if output.exists():raise FileExistsError('selection reconstruction already exists')
    if parent_path.parent in output.parents or chronicle==output or chronicle in output.parents:raise ValueError('output overlaps source')
    plan,parent_id,parent=load_manifest(parent_path)
    selected={}
    for path in plan['agent_real']['validation']:
        for row in map(json.loads,path.open()):selected[hashlib.sha256(row['text'].encode()).hexdigest()]=row
    if not 1<=len(selected)<=1000:raise ValueError('selection repair must remain bounded')
    snapshot=parent['complete_trajectory_export']['event_snapshot'];snapshot_path=parent_path.parent/snapshot['path']
    if file_identity(snapshot_path)!={k:snapshot[k] for k in ('bytes','sha256')}:raise ValueError('event snapshot differs')
    events={row['sha256']:row['path'] for row in map(json.loads,snapshot_path.open())}
    originals=[]
    for record in parent['validation_reservations']['parent_files']['agent_real']:
        path=(parent_path.parent/record['path']).resolve()
        if file_identity(path)!={k:record[k] for k in ('bytes','sha256')}:raise ValueError('original selection differs')
        originals.extend(row for row in map(json.loads,path.open()) if hashlib.sha256(row['text'].encode()).hexdigest() in selected)
    if len(originals)!=len(selected):raise ValueError('selection lineage is incomplete')
    reader=CompleteReader(chronicle,max_blob_bytes=1024**2);cache={};red=Redactor();stats=Counter();counts=Counter();repaired=[];pairs=[];seen=set()
    started=time.time()
    for original in originals:
        if reader.bytes_read>=256*1024**2 or time.time()-started>=120:
            counts['scan_budget_exhausted']+=1;break
        source_sha=original['event_file_sha256'];session=original['session_digest'];legacy_id=hashlib.sha256(original['text'].encode()).hexdigest()
        if source_sha not in events:counts['missing_pinned_event']+=1;continue
        if source_sha not in cache:
            episodes,identity=reader.session(chronicle/events[source_sha])
            if not identity or identity['sha256']!=source_sha:raise ValueError('pinned selection event changed')
            cache[source_sha]=episodes
        old_goal=original['text'].split('<|user|>',1)[1].split('<|end|>',1)[0].strip()
        old_final=original['text'].rsplit('<|assistant|>',1)[1].removesuffix('<|end|>').strip()
        matches=[]
        for episode in cache[source_sha]:
            if hashlib.sha256(str(episode.get('session')).encode()).hexdigest()!=session:continue
            goal=red(episode.get('goal'));final=red((episode.get('final') or {}).get('text'))
            if goal.strip().startswith(old_goal) and final.strip().startswith(old_final):matches.append(episode)
        if len(matches)!=1:counts['missing_or_ambiguous_episode']+=1;continue
        ep=matches[0];reason=rejection(ep)
        if reason:counts['excluded:'+reason]+=1;continue
        try:
            arguments={}
            for step in ep['steps']:
                safe,changed=safe_argument(step['args'],red);arguments[step['args']]=safe
                counts['structured_redaction_repairs']+=changed
        except ValueError:counts['redaction_damaged_arguments']+=1;continue
        def fields(text):return arguments[text] if isinstance(text,str) and text in arguments else red(text)
        text=render(ep,{}, {},set(),fields,stats).strip().removeprefix('[BOS]').removesuffix('[EOS]')
        doc_id=hashlib.sha256(text.encode()).hexdigest()
        if doc_id in seen:counts['duplicate_reconstruction']+=1;continue
        seen.add(doc_id)
        repaired.append({'id':doc_id,'text':text,'session_digest':session,
                         'origin':{'group':session,'source':'reconstructed_existing_selection','event_file_sha256':source_sha,
                                   'legacy_document_sha256':legacy_id}})
        pairs.append({'session_digest':session,'legacy_document_sha256':legacy_id,'complete_document_sha256':doc_id,'event_file_sha256':source_sha})
    if not repaired:raise ValueError('no complete existing selection traces')
    staging=output.with_name('.'+output.name+'.pending-'+str(os.getpid()));staging.mkdir(mode=0o700)
    path=staging/'selection.jsonl'
    with path.open('x') as stream:
        for row in repaired:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    path.chmod(0o600)
    sources={name:{split:[{**r,'path':os.path.relpath((parent_path.parent/r['path']).resolve(),output)} for r in records]
                         for split,records in source.items()} for name,source in parent['sources'].items()}
    sources['agent_real']['validation']=[{'path':path.name,**file_identity(path)}]
    reservations=parent['validation_reservations']
    manifest={**parent,'dataset_id':output.name,'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'parent_manifest_sha256':parent_id,'sources':sources,
              'validation_reservations':{**reservations,'path':os.path.relpath(parent_path.parent/reservations['path'],output),
                    'parent_files':{name:[{**r,'path':os.path.relpath((parent_path.parent/r['path']).resolve(),output)} for r in records]
                                    for name,records in reservations['parent_files'].items()}},
              'complete_trajectory_export':{**parent['complete_trajectory_export'],
                    **{name:{**parent['complete_trajectory_export'][name],
                       'path':os.path.relpath(parent_path.parent/parent['complete_trajectory_export'][name]['path'],output)}
                       for name in ('event_snapshot','reserved_sessions','fresh_reserved')}},
              'selection_reconstruction':{'source_manifest':{'path':str(parent_path),'sha256':parent_id},
                    'source_documents':len(selected),'reconstructed_documents':len(repaired),'complete':len(repaired)==len(selected),
                    'source_sessions':len({r['session_digest'] for r in originals}),'reconstructed_sessions':len({r['session_digest'] for r in repaired}),
                    'pairs':pairs,'counts':dict(counts),'redactions':dict(red.counts),
                    'bounds':{'max_blob_bytes':reader.max_blob_bytes,'read_budget_bytes':256*1024**2,'scan_ttl_seconds':120},
                    'bytes_read':reader.bytes_read,
                    'code':file_identity(Path(__file__)),'fresh_reserved_used':False,'final_test_used':False},
              'phase':'complete reconstruction of existing selection only; training source unchanged; unpromoted'}
    # This is a source view for role-window experiments, not a reusable generic mixture recipe.
    manifest.pop('pilot_pools',None);manifest.pop('provenance',None)
    manifest['audit']={'inherited_parent_audit':parent['audit'],'sources':{**parent['audit']['sources'],
                       'agent_real':{**parent['audit']['sources']['agent_real'],'validation_documents':len(repaired)}}}
    manifest['row_count']=sum(r['train_documents']+r['validation_documents'] for r in manifest['audit']['sources'].values())
    manifest['transform_sha256']=file_identity(Path(__file__))['sha256']
    manifest['transform_dag']={'parent_manifest_sha256':parent_id,'code':manifest['selection_reconstruction']['code'],
          'selection_output':file_identity(path),'episode_pairs':pairs,'method':'unique pinned event/session plus redacted goal/final prefix match'}
    manifest['transform_dag_digest']=hashlib.sha256(json.dumps(manifest['transform_dag'],sort_keys=True).encode()).hexdigest()
    manifest['source_digest']=hashlib.sha256(json.dumps([parent_id,file_identity(path)],sort_keys=True).encode()).hexdigest()
    if load_manifest(parent_path)[1]!=parent_id:raise ValueError('parent drifted during reconstruction')
    write_json(staging/'MANIFEST.json',manifest);load_manifest(staging/'MANIFEST.json');staging.rename(output)
    return output/'MANIFEST.json',manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--chronicle',type=Path,default=Path.home()/'.dsco/chronicle')
    args=parser.parse_args();path,manifest=reconstruct(args.parent_manifest,args.output,args.chronicle)
    report=manifest['selection_reconstruction'];print(json.dumps({'manifest':str(path),**{k:v for k,v in report.items() if k not in {'pairs','code','redactions'}}},indent=2))


if __name__=='__main__':main()
