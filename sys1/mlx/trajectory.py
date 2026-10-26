"""Bounded native-MLX paired diagnostics for role-aware trajectory objectives.

GPU by default, no CPU fallback. No production checkpoints, network or job control.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

from sys1.datasets.trajectory_windows import load, training_schedule, write_json
from sys1.datasets.trajectory_statistics import summarize
from sys1.mlx.model import PRESETS, Sys1Model, param_count
from sys1.model.corpus_manifest import file_identity


def weighted_loss(model, ids, valid, target, context_weight):
    logits=model(ids[:,:-1]).astype(mx.float32)
    losses=nn.losses.cross_entropy(logits,ids[:,1:],reduction='none')
    weights=valid[:,1:].astype(mx.float32)*(context_weight+(1-context_weight)*target[:,1:])
    return mx.sum(losses*weights)/mx.maximum(mx.sum(weights),1.)


def initial_identity(model):
    digest=hashlib.sha256()
    for name,value in sorted(tree_flatten(model.parameters())):
        digest.update(name.encode());digest.update(np.array(value).tobytes())
    return digest.hexdigest()


def evaluate(model, part, arm, seed):
    model.eval();records=[];arrays=part['arrays']
    for index,item in enumerate(part['items']):
        ids=mx.array(np.array(arrays['ids'][index:index+1],dtype=np.int32))
        losses=nn.losses.cross_entropy(model(ids[:,:-1]).astype(mx.float32),ids[:,1:],reduction='none')
        mx.eval(losses);values=np.array(losses)[0]
        target=arrays['target'][index,1:].astype(bool);valid=arrays['valid'][index,1:].astype(bool)
        if not np.isfinite(values).all():raise FloatingPointError('nonfinite validation NLL')
        row={k:item[k] for k in ('item_id','session','target_role','task_stratum','context_bucket')}
        row.update(arm=arm,seed=seed,target_nll=float(values[target].mean()),all_token_nll=float(values[valid].mean()),
                   target_tokens=int(target.sum()),valid_tokens=int(valid.sum()))
        records.append(row)
    model.train()
    return records


def run(manifest_path,output,*,steps=128,seeds=(1103,2207,3301),context_weight=.1,
        device='gpu',precision='bf16',memory_gib=2.,ttl_seconds=300,compiled=True,legacy_panel=None):
    manifest_path,output=Path(manifest_path).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('study already exists; immutable runs cannot be resumed with new metadata')
    if 'matrix' in platform.node().lower():raise ValueError('protected Matrix host is excluded')
    if not 4<=steps<=512 or not 1<=len(seeds)<=3 or len(set(seeds))!=len(seeds) or any(type(s) is not int or not 0<=s<2**32 for s in seeds):
        raise ValueError('bounded steps and distinct seeds required')
    if not math.isfinite(context_weight) or not 0<=context_weight<1 or not 30<=ttl_seconds<=900 or not math.isfinite(memory_gib) or not .25<=memory_gib<=4:
        raise ValueError('invalid objective or resource budget')
    if device not in ('gpu','cpu') or precision not in ('bf16','fp32'):raise ValueError('invalid backend')
    if device=='gpu' and not mx.metal.is_available():raise RuntimeError('Metal GPU unavailable; no CPU fallback')
    mx.set_default_device(mx.gpu if device=='gpu' else mx.cpu)
    limit=int(memory_gib*1024**3);mx.set_memory_limit(limit);mx.set_cache_limit(min(limit//8,128*1024**2))
    parts,data_id,manifest=load(manifest_path);policy=manifest['policy']
    legacy_parts=legacy_id=legacy_manifest=None
    if legacy_panel:
        legacy_panel=Path(legacy_panel).resolve();legacy_parts,legacy_id,legacy_manifest=load(legacy_panel)
        if any(legacy_manifest['policy'][k]!=policy[k] for k in ('seq','vocab_size')) or legacy_manifest['tokenizer_identity']!=manifest['tokenizer_identity']:
            raise ValueError('legacy panel model/tokenizer policy differs')
    cfg=replace(PRESETS['sys1-mlx-dev'],vocab_size=policy['vocab_size'],max_seq=policy['seq'])
    if param_count(cfg)['total']*16>limit//2:raise ValueError('model state leaves insufficient memory for bounded activations')
    root=Path(__file__).resolve().parents[2]
    code_paths=[Path(__file__),root/'sys1/mlx/model.py',root/'sys1/datasets/trajectory_windows.py',
                root/'sys1/datasets/trajectory_statistics.py',root/'sys1/model/corpus_manifest.py',
                root/'scripts/reconstruct_trajectory_selection.py']
    code={str(path.relative_to(root)):file_identity(path) for path in code_paths}
    if manifest['code_identity']!=code['sys1/datasets/trajectory_windows.py']:raise ValueError('window transform differs from dataset')
    if legacy_manifest and legacy_manifest['code_identity']!=manifest['code_identity']:raise ValueError('legacy panel transform differs')
    reconstruction=manifest.get('selection_reconstruction')
    if reconstruction and reconstruction['code']!=code['scripts/reconstruct_trajectory_selection.py']:
        raise ValueError('selection reconstruction transform differs')
    schedules={str(seed):training_schedule(parts['train']['items'],steps,seed) for seed in seeds}
    model_name=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip() if sys.platform=='darwin' else platform.processor()
    plan={'schema':'sys1-mlx-trajectory-study-plan-v1','run_id':output.name,'created_unix':time.time(),
          'hypothesis':'on identical session-balanced context windows, emphasizing assistant target spans may improve session-macro target NLL',
          'dataset_identity':data_id,'code':code,'model_config':asdict(cfg),'parameters':param_count(cfg)['total'],
          'legacy_panel_identity':legacy_id,'legacy_panel_role':'secondary only; never primary or promotion evidence',
          'backend':'mlx','framework_version':mx.__version__,'device':device,'hardware':model_name,'precision':precision,
          'seeds':list(seeds),'steps_per_arm':steps,'batch':1,'arms':{'uniform':1.,'targeted':context_weight},
          'schedule_indices':schedules,'optimizer':{'name':'AdamW','lr':.003,'weight_decay':.1,'clip_norm':1.,'warmup':8,'cosine_floor':.1},
          'compiled':compiled,'memory_limit_bytes':limit,'ttl_seconds':ttl_seconds,
          'evaluation':{'primary':'assistant role macro within session, then session macro across seeds',
                        'minimum_gain':.01,'maximum_regression':.02,'minimum_seeds':3,'minimum_sessions':20,
                        'bootstrap':2000,'bootstrap_seed':41873,'guard_multiplicity':'Bonferroni over observed role/task/context slices'},
          'stop_rule':'one preregistered uniform-versus-targeted comparison; no automatic data or threshold tuning',
          'final_test_used':False,'fresh_reserved_used':False,'production_launch_performed':False,'matrix_contacted':False}
    output.mkdir(parents=True,mode=0o700);write_json(output/'plan.json',plan)
    plan_id=file_identity(output/'plan.json');started=time.time();all_records=[];legacy_records=[];runs=[];initials={}
    old_handler=signal.getsignal(signal.SIGALRM)
    def timeout(signum,frame):raise TimeoutError('trajectory study TTL exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(ttl_seconds)
    try:
        for seed in seeds:
            for arm,weight in plan['arms'].items():
                mx.random.seed(seed);mx.clear_cache();mx.reset_peak_memory()
                model=Sys1Model(cfg,compute_dtype=mx.bfloat16 if precision=='bf16' else mx.float32)
                model.freeze_decision_heads();mx.eval(model.parameters())
                initial=initial_identity(model)
                if seed in initials and initials[seed]!=initial:raise ValueError('paired model initializations differ')
                initials[seed]=initial
                optimizer=optim.AdamW(learning_rate=.003,weight_decay=.1);optimizer.init(model.trainable_parameters())
                def objective(m,ids,valid,target):return weighted_loss(m,ids,valid,target,weight)
                value_grad=nn.value_and_grad(model,objective)
                def step_fn(ids,valid,target,lr):
                    loss,grads=value_grad(model,ids,valid,target)
                    grads,norm=optim.clip_grad_norm(grads,1.)
                    optimizer.learning_rate=lr;optimizer.update(model,grads)
                    return loss,norm
                state=[model.state,optimizer.state]
                update=mx.compile(step_fn,inputs=state,outputs=state) if compiled else step_fn
                valid_tokens=target_tokens=0;role_draws=Counter();train=parts['train'];begin=time.time()
                for step,index in enumerate(schedules[str(seed)]):
                    values={key:mx.array(np.array(train['arrays'][key][index:index+1],dtype=np.int32)) for key in ('ids','valid','target')}
                    lr=.003*min(1.,(step+1)/8)*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
                    loss,norm=update(values['ids'],values['valid'],values['target'],mx.array(lr,dtype=mx.float32))
                    mx.eval(loss,norm,model.state,optimizer.state)
                    if not math.isfinite(float(loss)) or not math.isfinite(float(norm)):raise FloatingPointError('nonfinite training step')
                    valid_tokens+=int(train['arrays']['valid'][index].sum());target_tokens+=int(train['arrays']['target'][index].sum())
                    role_draws[train['items'][index]['target_role']]+=1
                    if (step+1)%32==0 or step+1==steps:
                        print(json.dumps({'seed':seed,'arm':arm,'step':step+1,'loss':float(loss),'peak_device_bytes':mx.get_peak_memory(),
                                          'elapsed_seconds':round(time.time()-started,2)}),flush=True)
                records=evaluate(model,parts['validation'],arm,seed);all_records.extend(records)
                secondary=evaluate(model,legacy_parts['validation'],arm,seed) if legacy_parts else []
                legacy_records.extend(secondary)
                run={'seed':seed,'arm':arm,'step':steps,'initial_parameters_sha256':initial,
                     'predicted_nonpadding_tokens':valid_tokens,'assistant_target_tokens':target_tokens,'role_draws':dict(role_draws),
                     'elapsed_seconds':time.time()-begin,'peak_device_bytes':mx.get_peak_memory()}
                runs.append(run);write_json(output/(arm+'-'+str(seed)+'.json'),{'run':run,'observations':records,'legacy_observations':secondary})
                del model,optimizer,update,value_grad,objective,step_fn,state,records,secondary,values,loss,norm
                gc.collect();mx.clear_cache()
        _,again,_=load(manifest_path)
        if again!=data_id or any(file_identity(root/path)!=identity for path,identity in code.items()):raise ValueError('dataset or code drifted during study')
        if legacy_panel and load(legacy_panel)[1]!=legacy_id:raise ValueError('legacy panel drifted during study')
        coverage=manifest['coverage']['validation'];items=parts['validation']['items'];roles=defaultdict(set)
        for item in items:roles[item['session']].add(item['target_role'])
        complete=(coverage['retained_sessions']==coverage['source_sessions'] and not manifest['counts'].get('validation:unusable_documents',0)
                  and not manifest['counts'].get('validation:legacy_invalid_arguments',0)
                  and (manifest.get('selection_reconstruction') or {}).get('complete',True)
                  and all(v=={'assistant_call','assistant_final'} for v in roles.values()))
        summary=summarize(all_records,coverage_complete=complete)
        receipt={'schema':'sys1-mlx-trajectory-study-v1','plan_identity':plan_id,'runs':runs,'summary':summary,
                 'legacy_secondary_summary':summarize(legacy_records,coverage_complete=False) if legacy_records else None,
                 'elapsed_seconds':time.time()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                 'torch_imported':'torch' in sys.modules,'production_quality_established':False,'promotion_performed':False,'matrix_contacted':False}
        write_json(output/'receipt.json',receipt)
        print(json.dumps({'run_id':output.name,'summary':summary,'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)
        return receipt
    except BaseException as exc:
        write_json(output/'failed.json',{'error_type':type(exc).__name__,'elapsed_seconds':time.time()-started,'completed_arms':len(runs),'plan_identity':plan_id})
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=128);parser.add_argument('--seeds',default='1103,2207,3301')
    parser.add_argument('--device',choices=['gpu','cpu'],default='gpu');parser.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    parser.add_argument('--context-weight',type=float,default=.1);parser.add_argument('--ttl-seconds',type=int,default=300)
    parser.add_argument('--memory-gib',type=float,default=2.)
    parser.add_argument('--legacy-panel',type=Path)
    args=parser.parse_args()
    run(args.manifest,args.output,steps=args.steps,seeds=tuple(map(int,args.seeds.split(','))),context_weight=args.context_weight,
        device=args.device,precision=args.precision,memory_gib=args.memory_gib,ttl_seconds=args.ttl_seconds,legacy_panel=args.legacy_panel)


if __name__=='__main__':main()
