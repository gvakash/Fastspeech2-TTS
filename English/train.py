"""Train the English (Svarah) FastSpeech 2 model.

Usage:
  python English/train.py --config English/config.yaml                 # from scratch
  python English/train.py --config English/config.yaml --max_hours 7    # stop + save after 7 h
  python English/train.py --config English/config.yaml --restore_step latest   # resume
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fastspeech2_core"))

from config import load_config, split_configs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--restore_step", default="0", help="0 | <step> | latest")
    ap.add_argument("--max_hours", type=float, default=None, help="wall-clock budget for this run")
    ap.add_argument("--total_step", type=int, default=None, help="override train.step.total_step")
    ap.add_argument("--smoke", action="store_true",
                    help="20-step sanity run into checkpoints_smoke/ and logs_smoke/ (does not touch real logs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        for k in ("ckpt_path", "log_path"):
            cfg["train"]["path"][k] = cfg["train"]["path"][k].rstrip("/") + "_smoke"
        cfg["train"]["step"].update(log_step=5, val_step=10, save_step=20, synth_step=10 ** 9)
        args.total_step = 20
    from trainer import train

    restore = args.restore_step if args.restore_step == "latest" else int(args.restore_step)
    train(split_configs(cfg), restore_step=restore, max_hours=args.max_hours, total_step=args.total_step)


if __name__ == "__main__":
    main()
