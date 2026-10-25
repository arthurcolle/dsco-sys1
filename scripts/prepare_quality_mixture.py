#!/usr/bin/env python3
"""Build a losslessly decoded, quality-filtered successor; never modifies Matrix."""
import argparse
import json
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from sys1.datasets.quality_mixture import prepare

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=REPO/'data/mixtures/agent-plus-systems-v1/MANIFEST.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,default=REPO/'data/tokenizer-100k/tokenizer.json')
    parser.add_argument('--steps',type=int,default=30000)
    parser.add_argument('--max-source-passes',type=float,default=2.)
    parser.add_argument('--pool-windows',type=int,default=4096)
    parser.add_argument('--eval-windows',type=int,default=512)
    parser.add_argument('--max-output-mib',type=int,default=1536)
    parser.add_argument('--min-free-gib',type=float,default=16.)
    parser.add_argument('--source-register',type=Path,action='append',default=[])
    args=parser.parse_args()
    os.environ['TOKENIZERS_PARALLELISM']='false'
    path,manifest=prepare(args.parent,args.output,args.tokenizer,steps=args.steps,max_passes=args.max_source_passes,
        pool_windows=args.pool_windows,eval_windows=args.eval_windows,max_output_bytes=args.max_output_mib*1024**2,
        min_free_bytes=int(args.min_free_gib*1024**3),source_registers=args.source_register,progress=lambda row:print(json.dumps(row),flush=True))
    print(json.dumps({'manifest':str(path),'weights':manifest['objective_weights'],'audit':manifest['audit'],'expected_passes':manifest['exposure_budget']['expected_passes']}),flush=True)

if __name__=='__main__':main()
