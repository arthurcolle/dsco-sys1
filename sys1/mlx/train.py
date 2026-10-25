"""Native MLX causal pretraining. GPU default; never controls another process.

    python -m sys1.mlx.train --name mlx-small --preset sys1-mlx-small \
      --corpus-manifest data/mixtures/agent-plus-systems-v1/MANIFEST.json \
      --tokenizer data/tokenizer-100k/tokenizer.json --steps 128
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from tokenizers import Tokenizer

from sys1.mlx.model import PRESETS, Sys1Model, lm_loss, param_count
from sys1.mlx.data import build_data
from sys1.model.mixture import AdaptiveMixture
from sys1.model.validation import ValidationGuard
from sys1.model.corpus_manifest import file_identity
from sys1.paths import RUNS


def learning_rate(step, args):
    if step < args.warmup:
        return args.lr*(step+1)/args.warmup
    fraction = min(1.,(step-args.warmup)/max(1,args.steps-args.warmup))
    return args.lr*(args.min_lr_ratio+(1-args.min_lr_ratio)*.5*(1+math.cos(math.pi*fraction)))


def make_optimizer(model, args):
    options = {"learning_rate":args.lr,"betas":(args.beta1,args.beta2),"eps":1e-8}
    optimizer = optim.MultiOptimizer([optim.AdamW(**options,weight_decay=args.wd),optim.AdamW(**options,weight_decay=0)],
        [lambda path,value:value.ndim >= 2 and (args.decay_embeddings or not path.startswith("embed."))])
    optimizer.init(model.trainable_parameters())
    return optimizer


def make_step(model, optimizer, accumulation, source_count, clip, compiled=True):
    value_grad = nn.value_and_grad(model,lm_loss)
    def step(windows, labels, lr):
        grads = tree_map(mx.zeros_like,model.trainable_parameters())
        loss = mx.array(0.,dtype=mx.float32)
        sums,counts = mx.zeros((source_count,),dtype=mx.float32),mx.zeros((source_count,),dtype=mx.int32)
        for index in range(accumulation):
            (micro_loss,per_sequence),micro_grads = value_grad(model,windows[index])
            loss = loss + micro_loss/accumulation
            grads = tree_map(lambda a,b:a+b/accumulation,grads,micro_grads)
            sums = sums + mx.stack([mx.sum(mx.where(labels[index]==source,per_sequence,0)) for source in range(source_count)])
            counts = counts + mx.stack([mx.sum(labels[index]==source) for source in range(source_count)])
        if clip > 0:
            grads,norm = optim.clip_grad_norm(grads,clip)
        else:
            norm = mx.sqrt(sum(mx.sum(g.astype(mx.float32)**2) for _,g in tree_flatten(grads)))
        optimizer.learning_rate = lr
        optimizer.update(model,grads)
        return loss,norm,sums,counts
    state = [model.state,optimizer.state]
    return mx.compile(step,inputs=state,outputs=state) if compiled else step


def evaluate(model, heldout, batch_size):
    model.eval()
    losses = {}
    try:
        for name,windows in heldout.items():
            total = mx.array(0.,dtype=mx.float32)
            for offset in range(0,len(windows),batch_size):
                loss,_ = lm_loss(model,windows[offset:offset+batch_size])
                total = total + loss*len(windows[offset:offset+batch_size])
            mx.eval(total)
            losses[name] = float(total)/len(windows)
    finally:
        model.train()
    return losses


def checkpoint(root, step, model, optimizer, mixture, policy, validation, adaptive, sums, counts, keep, is_best=False):
    root.mkdir(parents=True,exist_ok=True)
    target = root/f"step_{step:08d}"
    temporary = root/f".step_{step:08d}.{os.getpid()}.tmp"
    if target.exists() or temporary.exists():
        raise FileExistsError("refusing to overwrite an MLX checkpoint")
    # FP32 masters + two FP32 Adam moments; do not start an oversized write.
    required = sum(value.nbytes for _,value in tree_flatten(model.parameters())) + sum(value.nbytes for _,value in tree_flatten(optimizer.state))
    if shutil.disk_usage(root).free < required+64*1024**2:
        raise RuntimeError("insufficient disk for an atomic MLX checkpoint")
    temporary.mkdir(mode=0o700)
    mx.eval(model.state,optimizer.state,sums,counts)
    model.save_weights(str(temporary/"model.safetensors"))
    mx.save_safetensors(str(temporary/"optimizer.safetensors"),dict(tree_flatten(optimizer.state)))
    record = {"schema":"sys1-mlx-checkpoint-v1","step":step,"policy":policy,"config":asdict(model.cfg),
              "sampler":mixture.state(),"validation":validation.state(),"adaptive":adaptive.state() if adaptive else None,
              "training_source_sums":sums.tolist(),"training_source_counts":counts.tolist()}
    (temporary/"state.json").write_text(json.dumps(record)+"\n")
    for path in temporary.iterdir():path.chmod(0o600)
    temporary.rename(target)
    def pointer(name):
        path = root/(name+".tmp")
        path.write_text(target.name+"\n");path.replace(root/name)
    pointer("latest")
    if is_best:pointer("best")
    protected = {target}
    if (root/"best").is_file():
        name = (root/"best").read_text().strip()
        if not re.fullmatch(r"step_[0-9]{8}",name) or not (root/name/"state.json").is_file():
            raise ValueError("invalid best checkpoint pointer; refusing cleanup")
        protected.add(root/name)
    for path in sorted(root.glob("step_*"),reverse=True):
        if len(protected) < keep:protected.add(path)
    for path in root.glob("step_*"):
        if path not in protected and path.is_dir():shutil.rmtree(path)
    return target


def restore(path, model, optimizer, mixture, policy, validation, adaptive):
    record = json.loads((path/"state.json").read_text())
    if record.get("schema") != "sys1-mlx-checkpoint-v1" or record.get("policy") != policy or record.get("config") != asdict(model.cfg):
        raise ValueError("MLX checkpoint policy, configuration or dataset differs")
    if type(record.get("step")) is not int or not 0 <= record["step"] <= policy["steps"]:
        raise ValueError("invalid MLX checkpoint step")
    validation.load_state(record["validation"],step_hint=record["step"])
    if (record.get("adaptive") is not None) != (adaptive is not None):
        raise ValueError("cannot change adaptive mixture mode on resume")
    if adaptive:
        adaptive.load_state(record["adaptive"])
    mixture.load_state(record["sampler"])
    if mixture.weights != (adaptive.weights if adaptive else validation.weights):
        raise ValueError("sampler probabilities disagree with the saved policy")
    sums,counts = record["training_source_sums"],record["training_source_counts"]
    if len(sums) != len(mixture.names) or len(counts) != len(mixture.names) or any(not math.isfinite(v) or v < 0 for v in sums) or any(type(v) is not int or v < 0 for v in counts):
        raise ValueError("invalid saved training-source statistics")
    model.load_weights(str(path/"model.safetensors"),strict=True)
    optimizer.state = tree_unflatten(list(mx.load(str(path/"optimizer.safetensors")).items()))
    mx.eval(model.state,optimizer.state)
    return record["step"],mx.array(sums,dtype=mx.float32),mx.array(counts,dtype=mx.int32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name",required=True)
    parser.add_argument("--preset",choices=sorted(PRESETS),default="sys1-mlx-small")
    parser.add_argument("--tokenizer",required=True)
    parser.add_argument("--corpus-manifest",required=True)
    parser.add_argument("--device",choices=["gpu","cpu"],default="gpu")
    parser.add_argument("--precision",choices=["bf16","fp32"],default="bf16")
    parser.add_argument("--seq",type=int,default=256)
    parser.add_argument("--micro-batch",type=int,default=4)
    parser.add_argument("--accum",type=int,default=1)
    parser.add_argument("--steps",type=int,default=1000)
    parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--warmup",type=int,default=50)
    parser.add_argument("--min-lr-ratio",type=float,default=.1)
    parser.add_argument("--wd",type=float,default=.1)
    parser.add_argument("--beta1",type=float,default=.9)
    parser.add_argument("--beta2",type=float,default=.95)
    parser.add_argument("--clip",type=float,default=1.)
    parser.add_argument("--decay-embeddings",action="store_true")
    parser.add_argument("--gradient-checkpointing",action="store_true")
    parser.add_argument("--no-compile",action="store_true")
    parser.add_argument("--eval-every",type=int,default=100)
    parser.add_argument("--eval-seqs",type=int,default=64)
    parser.add_argument("--early-stop-patience",type=int,default=8)
    parser.add_argument("--early-stop-min-delta",type=float,default=.01)
    parser.add_argument("--max-domain-regression",type=float,default=.15)
    parser.add_argument("--domain-regression-patience",type=int,default=4)
    parser.add_argument("--mixture-policy",choices=["static","adaptive"],default="static")
    parser.add_argument("--mixture-min-observations",type=int,default=32)
    parser.add_argument("--mixture-warmup",type=int,default=3)
    parser.add_argument("--mixture-smoothing",type=float,default=.3)
    parser.add_argument("--ckpt-every",type=int,default=0)
    parser.add_argument("--keep",type=int,default=2)
    parser.add_argument("--log-every",type=int,default=10)
    parser.add_argument("--seed",type=int,default=1234)
    parser.add_argument("--resume",default="auto",help="auto, checkpoint directory, or empty for a fresh run")
    parser.add_argument("--stop-after",type=int,default=0,help="bounded interruption test; saves recovery state")
    parser.add_argument("--memory-limit-gib",type=float,default=8.)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}",args.name):parser.error("unsafe run name")
    if min(args.seq,args.micro_batch,args.accum,args.steps,args.eval_seqs,args.log_every) < 1 or args.keep < 2:
        parser.error("positive dimensions/log interval and at least two retained checkpoints are required")
    if min(args.eval_every,args.ckpt_every,args.warmup,args.stop_after) < 0:
        parser.error("intervals cannot be negative")
    if (args.early_stop_patience or args.max_domain_regression or args.mixture_policy=="adaptive") and not args.eval_every:
        parser.error("validation guards and adaptation require periodic evaluation")
    if not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("positive finite learning rate and memory limit required")
    if not 0 <= args.min_lr_ratio <= 1 or not math.isfinite(args.wd) or not 0 <= args.wd or not 0 <= args.beta1 < 1 or not 0 <= args.beta2 < 1 or not math.isfinite(args.clip):
        parser.error("invalid optimization policy")
    if args.device=="gpu" and not mx.metal.is_available():
        raise RuntimeError("MLX Metal GPU unavailable; no automatic CPU fallback")
    mx.set_default_device(mx.gpu if args.device=="gpu" else mx.cpu)
    memory_limit = int(args.memory_limit_gib*1024**3)
    mx.set_memory_limit(memory_limit)
    mx.set_cache_limit(min(memory_limit//8,256*1024**2))
    mx.random.seed(args.seed)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    cfg = replace(PRESETS[args.preset],vocab_size=tokenizer.get_vocab_size(),max_seq=max(PRESETS[args.preset].max_seq,args.seq))
    if param_count(cfg)["total"]*16 > memory_limit:
        raise ValueError("FP32 master/gradient/Adam minimum exceeds memory limit; declare a larger budget")
    mixture,heldout,origin,validation_id = build_data(args.corpus_manifest,args.tokenizer,tokenizer,args.seq,args.eval_seqs,args.seed)
    validation = ValidationGuard(mixture.weights,args.early_stop_patience,args.early_stop_min_delta,validation_id,
                                 args.max_domain_regression or None,args.domain_regression_patience)
    mixture.weights = dict(validation.weights)
    adaptive = AdaptiveMixture(mixture.weights,validation_id,min_observations=args.mixture_min_observations,
                               warmup=args.mixture_warmup,smoothing=args.mixture_smoothing) if args.mixture_policy=="adaptive" else None
    if adaptive:mixture.weights = dict(adaptive.weights)
    policy = {key:value for key,value in vars(args).items() if key not in {"name","resume","stop_after","ckpt_every","keep","log_every"}}
    policy.update(corpus_manifest_sha256=mixture.data_id,validation_id=validation_id,config=asdict(cfg),
                  framework_version=mx.__version__,compiled=not args.no_compile,weights=dict(mixture.weights))
    root = RUNS/"mlx-pretrain"/args.name
    latest = root/"checkpoints/latest"
    resume = None
    if args.resume=="auto" and latest.is_file():
        name = latest.read_text().strip()
        if not re.fullmatch(r"step_[0-9]{8}",name):raise ValueError("unsafe latest pointer")
        resume = latest.parent/name
    elif args.resume and args.resume!="auto":resume = Path(args.resume)
    if root.exists() and resume is None:
        if {path.name for path in root.iterdir()} != {"run-manifest.json"}:
            raise FileExistsError("run exists without a selected checkpoint; choose a fresh name")
        prepared = json.loads((root/"run-manifest.json").read_text())
        if prepared.get("schema") != "sys1-pretrain-run-v1" or prepared.get("backend") != "mlx" or prepared.get("status") != "prepared" or prepared.get("launch_performed") is not False:
            raise ValueError("unrecognized prepared MLX run")
        if prepared.get("corpus_manifest_sha256") != mixture.data_id or prepared.get("tokenizer_sha256") != file_identity(args.tokenizer)["sha256"]:
            raise ValueError("prepared data identities changed")
        for key,value in prepared["recipe"]["training"].items():
            if key not in {"tokenizer","corpus-manifest"} and getattr(args,key.replace("-","_"),None) != value:
                raise ValueError("prepared training settings changed")
        repo = Path(__file__).resolve().parents[2]
        if any(file_identity(repo/path)["sha256"] != sha for path,sha in prepared["code_sha256"].items()):
            raise ValueError("prepared training source changed")
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    minimum_checkpoint_bytes = param_count(cfg)["total"]*4 + (param_count(cfg)["total"]-param_count(cfg)["heads"])*8
    needs_checkpoint = resume is None or json.loads((resume/"state.json").read_text()).get("step",-1) < args.steps
    if needs_checkpoint and shutil.disk_usage(root).free < minimum_checkpoint_bytes+64*1024**2:
        raise RuntimeError("insufficient disk for the first resumable MLX checkpoint")
    model = Sys1Model(cfg,args.gradient_checkpointing,mx.bfloat16 if args.precision=="bf16" else mx.float32)
    model.freeze_decision_heads()
    optimizer = make_optimizer(model,args)
    step,sums,counts = 0,mx.zeros((len(mixture.names),)),mx.zeros((len(mixture.names),),dtype=mx.int32)
    if resume:step,sums,counts = restore(resume,model,optimizer,mixture,policy,validation,adaptive)
    mx.eval(model.state,optimizer.state)
    log_path = root/"log.jsonl"
    def log(row):
        row = {"t":round(time.time(),3),**row}
        with log_path.open("a") as stream:stream.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
    log({"event":"resume" if resume else "start","step":step,"run_id":args.name,"framework":"mlx","framework_version":mx.__version__,
         "device":str(mx.default_device()),"hardware":mx.device_info() if args.device=="gpu" else {},"precision":args.precision,
         "master_weights":"float32","optimizer_moments":"float32","params":model.num_parameters(),"config":asdict(cfg),
         "compiled":not args.no_compile,"gradient_checkpointing":args.gradient_checkpointing,"weights":dict(mixture.weights),
         "validation_policy":validation.policy(),"sources":origin,"policy_sha256":hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()})
    if validation.should_stop:
        log({"event":"early_stopped","step":step,"stop_reason":validation.stop_reason,"resumed_terminal_guard":True})
        return
    step_fn = make_step(model,optimizer,args.accum,len(mixture.names),args.clip,not args.no_compile)
    last_saved,ran,tokens_since = step,0,0
    started = time.perf_counter()
    mx.reset_peak_memory()
    while step < args.steps:
        batch,labels = mixture.batch(args.micro_batch*args.accum)
        batch = batch.reshape(args.accum,args.micro_batch,args.seq+1)
        labels = labels.reshape(args.accum,args.micro_batch)
        loss,norm,step_sums,step_counts = step_fn(batch,labels,mx.array(learning_rate(step,args),dtype=mx.float32))
        sums,counts = sums+step_sums,counts+step_counts
        mx.eval(model.state,optimizer.state,loss,norm,sums,counts)
        step += 1;ran += 1;tokens_since += args.seq*args.micro_batch*args.accum
        if not math.isfinite(float(loss)) or not math.isfinite(float(norm)):
            log({"event":"abort","step":step,"reason":"nonfinite loss or gradient; no checkpoint saved"})
            raise FloatingPointError("nonfinite MLX training step")
        if step % args.log_every==0 or step==args.steps:
            elapsed = time.perf_counter()-started
            log({"event":"train","step":step,"loss":float(loss),"grad_norm":float(norm),"tokens":step*args.seq*args.micro_batch*args.accum,
                 "tok_per_s":tokens_since/elapsed,"elapsed_interval_seconds":elapsed,"peak_memory_bytes":mx.get_peak_memory(),
                 "active_memory_bytes":mx.get_active_memory(),"drawn":dict(mixture.drawn),"sampling_weights":dict(mixture.weights)})
            started,tokens_since = time.perf_counter(),0
        best,stopped = False,False
        if (args.eval_every and step % args.eval_every==0) or step==args.steps:
            losses = evaluate(model,heldout,args.micro_batch)
            decision = validation.observe(step,losses)
            best,stopped = decision["improved"],decision["should_stop"]
            source_counts = {name:int(value) for name,value in zip(mixture.names,counts.tolist())}
            source_means = {name:float(value)/source_counts[name] if source_counts[name] else None for name,value in zip(mixture.names,sums.tolist())}
            log({"event":"eval","step":step,"heldout_loss":losses,"validation":decision,"training_source_loss":source_means,"training_source_counts":source_counts})
            if adaptive and not stopped:
                feedback = adaptive.observe(step,losses,source_means,source_counts)
                mixture.weights = dict(adaptive.weights)
                log({"event":"mixture_update","step":step,**feedback})
            sums,counts = mx.zeros_like(sums),mx.zeros_like(counts)
            started,tokens_since = time.perf_counter(),0
        if best or stopped or step==args.steps or (args.ckpt_every and step % args.ckpt_every==0):
            saved = checkpoint(root/"checkpoints",step,model,optimizer,mixture,policy,validation,adaptive,sums,counts,args.keep,best)
            last_saved = step
            log({"event":"checkpoint","step":step,"path":str(saved),"best":best})
            started,tokens_since = time.perf_counter(),0
        if stopped:
            log({"event":"early_stopped","step":step,"best_step":validation.best_step,"stop_reason":validation.stop_reason})
            break
        if args.stop_after and ran >= args.stop_after:break
    if last_saved != step:
        saved = checkpoint(root/"checkpoints",step,model,optimizer,mixture,policy,validation,adaptive,sums,counts,args.keep)
        log({"event":"checkpoint","step":step,"path":str(saved),"best":False})
    if step==args.steps and not validation.should_stop:log({"event":"done","step":step,"tokens":step*args.seq*args.micro_batch*args.accum})


if __name__=="__main__":main()
