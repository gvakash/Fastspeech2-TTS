"""Make a small, GitHub-friendly checkpoint (NEW).

Training checkpoints hold the model AND the Adam optimiser state in fp32
(~420 MB). For submission/inference only the model weights are needed; stored
in fp16 they take ~70 MB, below GitHub's 100 MB per-file limit.
The inference code converts them back to fp32 on load.

Usage:
  python tools/export_checkpoint.py --lang English            # best checkpoint
  python tools/export_checkpoint.py --lang Telugu --step 60000
Writes <lang>/checkpoints/fastspeech2_<lang>_<step>_fp16.pth.tar
"""
import argparse
import json
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["English", "Telugu"])
    ap.add_argument("--step", default="best", help="best | latest | <step>")
    args = ap.parse_args()
    ckpt_dir = os.path.join(ROOT, args.lang, "checkpoints")
    step = args.step
    if step == "best" and os.path.exists(os.path.join(ckpt_dir, "best.json")):
        step = json.load(open(os.path.join(ckpt_dir, "best.json")))["step"]
    if step in ("best", "latest"):
        steps = [int(f.split(".")[0]) for f in os.listdir(ckpt_dir) if f.split(".")[0].isdigit()]
        step = max(steps)
    src = os.path.join(ckpt_dir, "{}.pth.tar".format(step))
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in ckpt["model"].items()}
    dst = os.path.join(ckpt_dir, "fastspeech2_{}_{}_fp16.pth.tar".format(args.lang.lower(), step))
    torch.save({"model": sd, "step": int(step), "dtype": "float16"}, dst)
    print("{} -> {} ({:.1f} MB)".format(src, dst, os.path.getsize(dst) / 2 ** 20))


if __name__ == "__main__":
    main()
