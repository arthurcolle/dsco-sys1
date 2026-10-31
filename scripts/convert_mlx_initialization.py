#!/usr/bin/env python3
"""Convert the completed legacy parent to MLX tensor storage without Torch."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sys1.mlx.legacy_weights import convert

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--tokenizer',required=True);p.add_argument('--parent-log',required=True)
    p.add_argument('--expected-step',type=int,required=True)
    a=p.parse_args()
    r=convert(a.checkpoint,a.output,a.tokenizer,expected_step=a.expected_step,require_done_log=a.parent_log)
    print(json.dumps({k:r[k] for k in ['schema','parent_completed','parent_step','model_identity','torch_imported']}))
