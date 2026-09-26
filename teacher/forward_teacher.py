"""Teacher stage 2 (GPU): forward the prepped (context + span) sequences from teacher.prepare_contexts
to the RAG teacher distribution. Per span: base forward over full_ids -> softmax(z_ctx) at span positions
-> top-64 (teacher_tok/teacher_prob). Stores the cached targets {span_ids, span_lens, teacher_tok, teacher_prob} consumed
by train.pretrain. No retrieval in the loop, so the GPU stays saturated. mp.spawn over GPUs; resumable per rank.
Length-sorted batching (less padding) + vectorized lm_head/softmax/topk over all span positions in a batch
(one kernel + one D2H copy). z_base is recomputed at train time.
"""
import argparse, sys, os, glob, time
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
SPANS_PER_CHUNK=20000

def order_rows(rows, preserve_order=False):
    if preserve_order:
        return rows
    return sorted(rows, key=lambda r: len(r[0]))

def worker(rank, world, args):
    """Run the frozen base on (context + span) and cache the top-64 teacher distribution
    at every span position."""
    raise NotImplementedError

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prepped", required=True, help="output directory of teacher.prepare_contexts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-v0.3")
    ap.add_argument("--topk", type=int, default=64); ap.add_argument("--batch_size", type=int, default=192)
    ap.add_argument("--world", type=int, default=8); ap.add_argument("--load_stagger", type=int, default=15)
    ap.add_argument("--max_spans", type=int, default=0)
    ap.add_argument("--njobs", type=int, default=1); ap.add_argument("--job_id", type=int, default=0)
    ap.add_argument("--preserve_order", action="store_true")
    a=ap.parse_args()
    if a.world==1: worker(0,1,a)
    else: mp.spawn(worker, args=(a.world,a), nprocs=a.world, join=True)

if __name__=="__main__":
    try: main()
    except Exception:
        import traceback; traceback.print_exc(); sys.exit(1)
