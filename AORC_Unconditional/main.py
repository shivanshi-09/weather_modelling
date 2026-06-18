"""
main.py — entry point.
Run: python main.py
     python main.py --device cpu --batch_size 2
"""

import argparse
from config import CFG
from train import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     default=CFG.device)
    parser.add_argument("--batch_size", type=int,   default=CFG.batch_size)
    parser.add_argument("--lr",         type=float, default=CFG.lr)
    parser.add_argument("--num_epochs", type=int,   default=CFG.num_epochs)
    parser.add_argument("--num_workers", type=int,  default=CFG.num_workers)
    parser.add_argument("--val_every", type=int,    default=CFG.val_every)
    parser.add_argument("--output_dir", default=CFG.output_dir)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    CFG.device     = args.device
    CFG.batch_size = args.batch_size
    CFG.lr         = args.lr
    CFG.num_epochs = args.num_epochs
    CFG.num_workers = args.num_workers
    CFG.val_every  = args.val_every
    CFG.output_dir = args.output_dir
    CFG.amp        = CFG.amp and not args.no_amp

    print("=== Config ===")
    print(f"  device     : {CFG.device}")
    print(f"  batch_size : {CFG.batch_size}")
    print(f"  num_workers: {CFG.num_workers}")
    print(f"  amp        : {CFG.amp} ({CFG.amp_dtype})")
    print(f"  val_every  : {CFG.val_every}")
    print(f"  max_patches: {CFG.max_patches}")
    print(f"  max_val    : {CFG.max_val_patches}")
    print(f"  base_ch    : {CFG.base_ch}")
    print(f"  ch_mult    : {CFG.ch_mult}")
    print(f"  lr         : {CFG.lr}")
    print(f"  num_epochs : {CFG.num_epochs}")
    print(f"  output_dir : {CFG.output_dir}")
    print("==============")

    train()


if __name__ == "__main__":
    main()
