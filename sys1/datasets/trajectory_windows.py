"""Private, role-aware context windows with bounded session/stratum exposure.

No frameworks, network access, raw-source mutation or final-holdout feedback.
Windows are contiguous token slices; complete source trajectories remain intact.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import random
import shutil
import struct

from sys1.datasets.complete_trajectories import argument_object
from sys1.datasets.quality_mixture import NearIndex
from sys1.model.corpus_manifest import file_identity, load_manifest

ROLES = {'user':1,'assistant_call':2,'tool':3,'assistant_final':4}
MESSAGE = re.compile(r'<\|(user|assistant|tool)\|>(.*?)<\|end\|>\n?',re.S)
CALL = re.compile(r'\[ACT\]([A-Za-z_][A-Za-z0-9_:/.-]*)<\|json\|>(.*?)<\|/json\|>',re.S)


def parse(text, *, strict_arguments=True):
    spans=[];cursor=0;expect='user';bad_arguments=0
    for match in MESSAGE.finditer(text):
        if match.start()!=cursor:raise ValueError('invalid trajectory framing')
        speaker,content=match[1],match[2];tool=None
        if speaker=='user' and expect=='user':role='user';expect='assistant'
        elif speaker=='tool' and expect=='tool':role='tool';expect='assistant'
        elif speaker=='assistant' and expect=='assistant':
            call=CALL.fullmatch(content)
            if call:
                role='assistant_call';tool=call[1];expect='tool'
                try:argument_object(call[2])
                except ValueError:
                    bad_arguments+=1
                    if strict_arguments:raise ValueError('invalid argument object') from None
            else:role='assistant_final';expect='done'
        else:raise ValueError('invalid trajectory role order')
        spans.append({'start':match.start(),'end':match.end(),'role':role,'tool':tool})
        cursor=match.end()
    if cursor!=len(text) or expect!='done' or len(spans)<4:raise ValueError('incomplete trajectory')
    return spans,bad_arguments


def tool_class(name):
    name=name.lower()
    if any(x in name for x in ('exec','shell','command','terminal')):return 'execution'
    if any(x in name for x in ('write','patch','edit','create_file')):return 'editing'
    if any(x in name for x in ('read','search','find','list','glob')):return 'inspection'
    return 'other'


def signature(ids):
    values={hashlib.blake2b(struct.pack('<4I',*ids[i:i+4]),digest_size=8).digest() for i in range(len(ids)-3)}
    return tuple(sorted(values)[:8]) if len(values)>=8 else ()


def windows(row, tokenizer, seq, target_chunk, *, strict_arguments=True):
    text=row['text'];spans,bad=parse(text,strict_arguments=strict_arguments)
    encoded=tokenizer.encode(text,add_special_tokens=False)
    eos=tokenizer.token_to_id('[EOS]')
    if eos is None:raise ValueError('tokenizer requires EOS')
    labels=[];span_numbers=[];cursor=0
    for start,end in encoded.offsets:
        while cursor+1<len(spans) and start>=spans[cursor]['end']:cursor+=1
        if not spans[cursor]['start']<=start<end<=spans[cursor]['end']:
            raise ValueError('token crosses a role boundary')
        labels.append(ROLES[spans[cursor]['role']]);span_numbers.append(cursor)
    ids=[eos]+encoded.ids+[eos];roles=[0]+labels+[4]
    span_numbers=[-1]+span_numbers+[len(spans)-1]
    all_tools=[tool_class(s['tool']) for s in spans if s['tool']]
    primary=Counter(all_tools).most_common(1)[0][0]
    session=row.get('session_digest') or row.get('origin',{}).get('group')
    if not isinstance(session,str) or not re.fullmatch('[0-9a-f]{64}',session):raise ValueError('missing session identity')
    document=hashlib.sha256(text.encode()).hexdigest()
    if row.get('id',document)!=document:raise ValueError('document identity differs')
    ranges={}
    for index,number in enumerate(span_numbers):
        if number not in ranges:ranges[number]=[index,index+1]
        else:ranges[number][1]=index+1
    generated=[]
    for number,span in enumerate(spans):
        role=ROLES[span['role']]
        if role not in (2,4):continue
        first,last=ranges[number]
        for begin in range(first,last,target_chunk):
            stop=min(begin+target_chunk,last);start=max(0,stop-(seq+1))
            values=ids[start:stop];count=len(values);pad=seq+1-count
            target=[int(begin<=i<stop) for i in range(start,stop)]+[0]*pad
            item={'ids':values+[eos]*pad,'roles':roles[start:stop]+[0]*pad,
                  'valid':[0]+[1]*(count-1)+[0]*pad,'target':target,
                  'session':session,'document':document,'target_role':span['role'],
                  'task_stratum':tool_class(span['tool']) if span['tool'] else primary,
                  'context_bucket':'short' if len(ids)<=seq+1 else 'medium' if len(ids)<=1024 else 'long',
                  'start_token':start,'stop_token':stop,'valid_tokens':count-1}
            # The first token is input only, including when a long slice starts mid-message.
            item['target'][0]=0
            item['item_id']=hashlib.sha256(json.dumps([document,start,stop,begin,role],separators=(',',':')).encode()).hexdigest()
            generated.append(item)
    return generated,Counter(labels),bad


def balanced(candidates, per_session, per_stratum):
    buckets=defaultdict(list)
    for row in candidates:
        key=(row['session'],row['target_role'],row['task_stratum'])
        bucket=buckets[key];bucket.append(row)
        bucket.sort(key=lambda r:r['item_id'])
        del bucket[per_stratum:]
    if per_session is None:
        return [row for key in sorted(buckets) for row in buckets[key]]
    sessions=defaultdict(dict)
    for (session,role,task),bucket in buckets.items():sessions[session][role,task]=bucket
    selected=[]
    for session,groups in sorted(sessions.items()):
        queues=defaultdict(list)
        for (role,task),rows in sorted(groups.items()):queues[role].append(list(rows))
        cursors=Counter()
        kept=[]
        while len(kept)<per_session and any(any(q) for q in queues.values()):
            for role,role_queues in sorted(queues.items()):
                for _ in role_queues:
                    queue=role_queues[cursors[role]%len(role_queues)];cursors[role]+=1
                    if queue:
                        if len(kept)<per_session:kept.append(queue.pop(0))
                        break
        selected.extend(kept)
    return selected


def training_schedule(items, steps, seed):
    """Same session-balanced inputs for both arms; alternate requested target roles."""
    rng=random.Random(seed);sessions=defaultdict(lambda:defaultdict(list))
    for index,item in enumerate(items):sessions[item['session']][item['target_role']].append(index)
    if not sessions or steps<1:raise ValueError('empty training schedule')
    for roles in sessions.values():
        for indices in roles.values():rng.shuffle(indices)
    order=[];result=[];drawn=Counter()
    for step in range(steps):
        if not order:order=sorted(sessions);rng.shuffle(order)
        session=order.pop();roles=sessions[session]
        wanted=('assistant_call','assistant_final')[step%2]
        role=wanted if wanted in roles else sorted(roles)[0]
        indices=roles[role];result.append(indices[drawn[session,role]%len(indices)]);drawn[session,role]+=1
    return result


def write_json(path, value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    path.chmod(0o600)


def prepare(parent_path,output,tokenizer_path,*,seq=128,target_chunk=64,per_session=8,per_stratum=4):
    import numpy as np
    from tokenizers import Tokenizer
    parent_path,output,tokenizer_path=map(lambda p:Path(p).resolve(),(parent_path,output,tokenizer_path))
    if output.exists():raise FileExistsError('window dataset already exists')
    if parent_path.parent in output.parents:raise ValueError('output overlaps parent')
    if not 16<=seq<=512 or not 1<=target_chunk<=seq//2 or not 2<=per_session<=16 or not 1<=per_stratum<=8:
        raise ValueError('invalid bounded window policy')
    if shutil.disk_usage(output.parent).free<8*1024**3:raise ValueError('insufficient disk headroom')
    plan,parent_id,parent=load_manifest(parent_path)
    tokenizer_id=file_identity(tokenizer_path)
    if parent.get('tokenizer_identity')!=tokenizer_id:raise ValueError('tokenizer differs')
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    counts=Counter();role_tokens=Counter();parts={};groups={};source_documents={}
    val_hashes=set();near=NearIndex()
    for split in ('validation','train'):
        candidates=[];groups[split]=set();source_documents[split]=0
        for path in plan['agent_real'][split]:
            with path.open() as stream:
                for line in stream:
                    if len(line)>1024**2:raise ValueError('source record exceeds bound')
                    row=json.loads(line);session=row.get('session_digest') or row.get('origin',{}).get('group')
                    groups[split].add(session);source_documents[split]+=1
                    if source_documents[split]>10000:raise ValueError('source document budget exceeded')
                    try:items,tokens,bad=windows(row,tokenizer,seq,target_chunk,strict_arguments=split=='train')
                    except ValueError:
                        counts[split+':unusable_documents']+=1;continue
                    counts[split+':legacy_invalid_arguments']+=bad
                    if split=='train':role_tokens.update(tokens)
                    else:
                        # Reserve every validation anchor, including those not selected for scoring.
                        for item in items:
                            values=item['ids'][:item['valid_tokens']+1]
                            val_hashes.add(hashlib.sha256(struct.pack('<'+'I'*len(values),*values)).digest())
                            near.add(signature(values))
                    # Bound working memory before the final deterministic session selection.
                    candidates=balanced([*candidates,*items],None,per_stratum)
        parts[split]=balanced(candidates,per_session,per_stratum)
    if groups['train']&groups['validation']:raise ValueError('source session split overlap')
    reservations=parent.get('complete_trajectory_export',{}).get('reserved_sessions')
    if reservations:
        path=parent_path.parent/reservations['path']
        if file_identity(path)!={k:reservations[k] for k in ('bytes','sha256')}:raise ValueError('session reservations drifted')
        if groups['train']&set(json.loads(path.read_text())['session_digests']):raise ValueError('reserved session entered training')
    kept=[];seen=set()
    for row in parts['train']:
        ids=row['ids'][:row['valid_tokens']+1];key=hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).digest()
        if key in val_hashes or near.related(signature(ids)):
            counts['train:window_validation_overlap']+=1;continue
        unique=key+bytes(row['target'])
        if unique in seen:counts['train:window_duplicate']+=1;continue
        seen.add(unique);kept.append(row)
    parts['train']=kept
    if not all(parts.values()):raise ValueError('empty train or validation windows')
    staging=output.with_name('.'+output.name+'.pending-'+str(os.getpid()));staging.mkdir(mode=0o700)
    records={};coverage={}
    for split,rows in parts.items():
        records[split]={}
        for key,dtype in [('ids','<u4'),('roles','u1'),('valid','u1'),('target','u1')]:
            path=staging/(split+'-'+key+'.bin')
            array=np.array([r[key] for r in rows],dtype=dtype);array.tofile(path);path.chmod(0o600)
            records[split][key]={'path':path.name,'dtype':dtype,'shape':list(array.shape),**file_identity(path)}
        path=staging/(split+'-items.jsonl')
        with path.open('x') as stream:
            for row in rows:stream.write(json.dumps({k:v for k,v in row.items() if k not in {'ids','roles','valid','target'}})+'\n')
        path.chmod(0o600);records[split]['items']={'path':path.name,**file_identity(path)}
        coverage[split]={'source_documents':source_documents[split],'source_sessions':len(groups[split]),
                         'retained_sessions':len({r['session'] for r in rows}),'windows':len(rows),
                         'target_roles':dict(Counter(r['target_role'] for r in rows)),
                         'task_strata':dict(Counter(r['task_stratum'] for r in rows)),
                         'maximum_windows_per_session':max(Counter(r['session'] for r in rows).values())}
    if load_manifest(parent_path)[1]!=parent_id or file_identity(tokenizer_path)!=tokenizer_id:raise ValueError('input drift during window build')
    manifest={'schema':'sys1-trajectory-windows-v1','dataset_id':output.name,'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'parent_manifest':{'path':str(parent_path),'sha256':parent_id},'tokenizer_identity':tokenizer_id,
              'privacy':'PRIVATE: source traces, token windows and session metadata; never transfer or commit',
              'license_ref':parent.get('license_ref'),'unit_of_analysis':'assistant target span within contiguous context; session is inference cluster',
              'policy':{'seq':seq,'target_chunk':target_chunk,'per_session':per_session,'per_stratum':per_stratum,
                        'eos_id':tokenizer.token_to_id('[EOS]'),'vocab_size':tokenizer.get_vocab_size(),
                        'selection':'stable hashed priorities; round robin across assistant role and tool-class strata',
                        'role_codes':ROLES,'padding':'EOS with zero valid and target masks','training_objective':'declared by experiment, not inferred from masks'},
              'splits':records,'coverage':coverage,'counts':dict(counts),
              'training_role_tokens':{name:role_tokens[code] for name,code in ROLES.items()},
              'code_identity':file_identity(Path(__file__)),'fresh_reserved_used':False,'final_test_used':False,
              'selection_reconstruction':parent.get('selection_reconstruction'),
              'validation_policy':'same source selection sessions; role-aware derived annotations; no new reserved data scored',
              'window_overlap_policy':{'validation_anchors_reserved_before_selection':len(val_hashes),
                    'near_duplicate_sketch':'bottom eight token four-grams; at least six shared; capped candidate bands',
                    'saturated_validation_bands':len(near.saturated)},
              'limitations':['legacy selection arguments can be clipped; counted separately','windowed contexts may begin mid-message',
                             'heuristic near-duplicate screening, not semantic deduplication','small-model studies cannot promote the production model'],
              'launch_performed':False}
    write_json(staging/'MANIFEST.json',manifest);staging.rename(output)
    return output/'MANIFEST.json',manifest


def load(path):
    """Verify every byte and mask before exposing read-only memory maps."""
    import numpy as np
    path=Path(path).resolve()
    if path.stat().st_size>8*1024**2:raise ValueError('oversized window manifest')
    manifest=json.loads(path.read_text())
    if manifest.get('schema')!='sys1-trajectory-windows-v1':raise ValueError('unsupported window dataset')
    seq=manifest['policy']['seq'];parts={};groups={};all_ids=set()
    if type(seq) is not int or not 16<=seq<=512:raise ValueError('invalid context length')
    for split in ('train','validation'):
        arrays={};items=None
        for key,record in manifest['splits'][split].items():
            source=(path.parent/record['path']).resolve()
            if path.parent not in source.parents:raise ValueError('window path escaped dataset')
            if file_identity(source)!={k:record[k] for k in ('bytes','sha256')}:raise ValueError('window file identity differs')
            if key=='items':
                items=[json.loads(line) for line in source.open()]
                continue
            shape=record['shape'];dtype='<u4' if key=='ids' else 'u1'
            if key not in {'ids','roles','valid','target'} or record['dtype']!=dtype:
                raise ValueError('invalid window dtype')
            if len(shape)!=2 or any(type(x) is not int for x in shape) or not 1<=shape[0]<=160000 or shape[1]!=seq+1:
                raise ValueError('invalid window shape')
            if record['bytes']!=shape[0]*shape[1]*np.dtype(dtype).itemsize:raise ValueError('window byte size differs')
            arrays[key]=np.memmap(source,dtype=dtype,mode='r',shape=tuple(shape))
        if items is None or set(arrays)!={'ids','roles','valid','target'} or len({a.shape for a in arrays.values()})!=1 or len(items)!=len(arrays['ids']):
            raise ValueError('window arrays and metadata disagree')
        if arrays['ids'].max()>=manifest['policy']['vocab_size'] or arrays['roles'].max()>4:
            raise ValueError('token or role outside vocabulary')
        if any(not np.isin(arrays[key],[0,1]).all() for key in ('valid','target')):raise ValueError('nonbinary window mask')
        if np.any(arrays['target']>arrays['valid']) or np.any(arrays['valid'][:,0]) or np.any(arrays['target'].sum(axis=1)==0):
            raise ValueError('invalid causal target mask')
        groups[split]=set()
        for index,item in enumerate(items):
            if item['item_id'] in all_ids:raise ValueError('duplicate window primary key')
            all_ids.add(item['item_id']);groups[split].add(item['session'])
            expected=ROLES[item['target_role']]
            if expected not in (2,4) or np.any(arrays['roles'][index][arrays['target'][index].astype(bool)]!=expected):
                raise ValueError('target role disagrees with mask')
            count=item['valid_tokens']
            if not 1<=count<=seq or not np.array_equal(arrays['valid'][index],np.array([0]+[1]*count+[0]*(seq-count))):
                raise ValueError('noncontiguous valid-token mask')
            if np.any(arrays['ids'][index,count+1:]!=manifest['policy']['eos_id']):raise ValueError('invalid padding tokens')
        parts[split]={'arrays':arrays,'items':items}
    if groups['train']&groups['validation']:raise ValueError('window session split overlap')
    return parts,file_identity(path),manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True);parser.add_argument('--seq',type=int,default=128)
    args=parser.parse_args();path,manifest=prepare(args.parent_manifest,args.output,args.tokenizer,seq=args.seq)
    print(json.dumps({'manifest':str(path),'coverage':manifest['coverage'],'counts':manifest['counts'],'launch_performed':False},indent=2))


if __name__=='__main__':main()
