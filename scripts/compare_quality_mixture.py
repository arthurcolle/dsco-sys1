#!/usr/bin/env python3
"""Bounded, paired CPU pilot on frozen token reservoirs; no production-job access.

This measures the combined successor data policy, not each filter's causal effect.
It never uses the reserved addition probe or final test data for selection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import resource
import signal
import statistics
import sys
import time


def identity(path):
    digest=hashlib.sha256();size=0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):
            size+=len(chunk);digest.update(chunk)
    return {'bytes':size,'sha256':digest.hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-source',type=Path,default=Path(__file__).resolve().parents[1]/'sys1/model/model.py')
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--seeds',default='1234,2345,3456')
    parser.add_argument('--vocab-size',type=int,default=100352)
    parser.add_argument('--ttl-seconds',type=int,default=1800)
    args=parser.parse_args()
    seeds=[int(value) for value in args.seeds.split(',')]
    if not 64<=args.steps<=512 or not 1<=len(seeds)<=3 or len(set(seeds))!=len(seeds):parser.error('use 64..512 steps and 1..3 distinct seeds')
    if any(not 0<=seed<2**32 for seed in seeds) or not 2<=args.vocab_size<=131072 or not 30<=args.ttl_seconds<=3600:parser.error('invalid bounded pilot configuration')
    if 'matrix' in platform.node().lower():parser.error('protected Matrix host is excluded')
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source=args.model_source.resolve()
    spec=importlib.util.spec_from_file_location('quality_pilot_model',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    manifest=json.loads(args.manifest.read_text());policy=manifest['pilot_pools']
    packs={}
    for arm,entries in policy['packs'].items():
        packs[arm]={}
        for name,record in entries.items():
            path=(args.manifest.parent/record['path']).resolve()
            if args.manifest.parent.resolve() not in path.parents:raise ValueError('pool path escaped dataset')
            if identity(path)!={key:record[key] for key in ('bytes','sha256')}:raise ValueError('pool identity differs')
            if record['dtype']!='uint32-le' or record['window']!=policy['window']:raise ValueError('pool shape differs')
            data=np.fromfile(path,dtype='<u4').reshape(record['rows'],record['window'])
            if data.min()<0 or data.max()>=args.vocab_size:raise ValueError('pool token outside vocabulary')
            packs[arm][name]=data
    validation_weights=policy['validation_weights']
    if set(validation_weights)!=set(packs['validation']):raise ValueError('evaluation source mismatch')
    weights={'baseline':policy['baseline_weights'],'candidate':manifest['objective_weights']}
    for arm,values in weights.items():
        if set(values)!=set(packs[arm]) or min(values.values())<=0 or not math.isclose(sum(values.values()),1.,abs_tol=1e-9):raise ValueError('training weights mismatch')
    seq=policy['window']-1
    if not 8<=seq<=128:raise ValueError('pilot sequence exceeds bound')
    cfg=module.Sys1Config(vocab_size=args.vocab_size,d_model=32,n_layers=2,n_heads=4,n_kv_heads=2,ffn_dim=64,max_seq=seq)
    selected={}
    for name,data in packs['validation'].items():
        rng=random.Random('fixed-validation:'+name)
        selected[name]=rng.sample(range(len(data)),min(16,len(data)))
    criteria={'minimum_relative_weighted_gain':.01,'maximum_mean_source_regression':.02,'required_paired_wins':len(seeds)}
    args.output.mkdir(parents=True,exist_ok=False,mode=0o700)
    plan={'created_unix':time.time(),'host':platform.node(),'device':'cpu','threads':2,'runtime':{'python':platform.python_version(),'torch':torch.__version__},
          'seeds':seeds,'steps_per_arm':args.steps,'predicted_tokens_per_arm':args.steps*seq,'batch':1,'model_config':asdict(cfg),
          'manifest':identity(args.manifest),'model_source':identity(source),'script':identity(Path(__file__)),'training_weights':weights,
          'validation_weights':validation_weights,'fixed_validation_rows':selected,'criteria':criteria,'ttl_seconds':args.ttl_seconds,'max_peak_rss_bytes':1536*1024**2,
          'limitations':['tiny CPU model and short token budget; not a production-quality gate','combined policy change; no attribution to individual filters','selection validation only; no final test feedback']}
    plan_path=args.output/'plan.json';plan_path.write_text(json.dumps(plan,indent=2)+'\n');plan_path.chmod(0o600)
    def timeout(signum,frame):raise TimeoutError('bounded pilot TTL exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(args.ttl_seconds)
    started=time.time();runs=[]
    try:
        for seed in seeds:
            for arm in ('baseline','candidate'):
                torch.manual_seed(seed);model=module.Sys1Model(cfg).cpu()
                optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.1)
                rng=random.Random(seed);names=list(weights[arm]);probabilities=list(weights[arm].values());drawn={name:0 for name in names}
                model.train()
                for step in range(args.steps):
                    name=rng.choices(names,weights=probabilities,k=1)[0];data=packs[arm][name]
                    row=torch.from_numpy(data[rng.randrange(len(data))].astype('int64')).unsqueeze(0)
                    optimizer.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(model.lm_logits(model.hidden(row[:,:-1])).reshape(-1,args.vocab_size),row[:,1:].reshape(-1))
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    lr=.003*min(1.,(step+1)/8)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
                    for group in optimizer.param_groups:group['lr']=lr
                    optimizer.step();drawn[name]+=seq
                    if (step+1)%64==0:
                        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
                        if peak>plan['max_peak_rss_bytes']:raise MemoryError('pilot exceeded resident-memory budget')
                        print(json.dumps({'phase':'training','seed':seed,'arm':arm,'step':step+1,'peak_rss_bytes':peak,'elapsed_seconds':round(time.time()-started,1)}),flush=True)
                model.eval();losses={}
                with torch.inference_mode():
                    for name,indices in selected.items():
                        values=[]
                        for index in indices:
                            row=torch.from_numpy(packs['validation'][name][index].astype('int64')).unsqueeze(0)
                            values.append(float(F.cross_entropy(model.lm_logits(model.hidden(row[:,:-1])).reshape(-1,args.vocab_size),row[:,1:].reshape(-1))))
                        losses[name]=statistics.mean(values)
                result={'run_id':args.output.name+'-'+arm+'-'+str(seed),'seed':seed,'arm':arm,'step':args.steps,'predicted_tokens':args.steps*seq,
                        'drawn_tokens':drawn,'losses':losses,'weighted_loss':sum(validation_weights[name]*loss for name,loss in losses.items())}
                runs.append(result);path=args.output/(arm+'-'+str(seed)+'.json');path.write_text(json.dumps(result,indent=2)+'\n');path.chmod(0o600)
                print(json.dumps(result),flush=True)
                del model,optimizer
        means={arm:{name:statistics.mean(run['losses'][name] for run in runs if run['arm']==arm) for name in validation_weights} for arm in weights}
        weighted={arm:sum(validation_weights[name]*loss for name,loss in means[arm].items()) for arm in weights}
        gain=1-weighted['candidate']/weighted['baseline']
        regressions={name:means['candidate'][name]/means['baseline'][name]-1 for name in validation_weights}
        wins=sum(next(r for r in runs if r['arm']=='candidate' and r['seed']==seed)['weighted_loss']<next(r for r in runs if r['arm']=='baseline' and r['seed']==seed)['weighted_loss'] for seed in seeds)
        receipt={'plan':plan,'plan_identity':identity(plan_path),'runs':runs,'mean_source_losses':means,'mean_weighted_losses':weighted,
                 'relative_weighted_gain':gain,'relative_source_regressions':regressions,'paired_wins':wins,
                 'pilot_gate_passed':gain>=criteria['minimum_relative_weighted_gain'] and max(regressions.values())<=criteria['maximum_mean_source_regression'] and wins==len(seeds),
                 'production_quality_improvement_established':False,'matrix_contacted':False,'elapsed_seconds':time.time()-started,
                 'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
        path=args.output/'receipt.json';path.write_text(json.dumps(receipt,indent=2)+'\n');path.chmod(0o600)
        print(json.dumps({key:receipt[key] for key in ('relative_weighted_gain','relative_source_regressions','paired_wins','pilot_gate_passed','elapsed_seconds')}),flush=True)
    finally:signal.alarm(0)


if __name__=='__main__':main()
