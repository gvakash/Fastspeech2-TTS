"""Synthesise the English test sentences and plot the variance-adaptor outputs.

Usage:
  python English/inference.py --config English/config.yaml                     # all test sentences
  python English/inference.py --config English/config.yaml --text "Hello there." --id demo
  python English/inference.py --checkpoint 40000 --duration_control 1.1        # slower speech
Outputs go to Results/English/ (test_generated_audios/ and variance_adaptor_plots/).
"""
import argparse
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fastspeech2_core"))

from config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--checkpoint", default=None, help="best | latest | <step> | path to .pth.tar")
    ap.add_argument("--speaker", default=None, help="speaker name or id (default from config)")
    ap.add_argument("--test_sentences", default=None)
    ap.add_argument("--text", default=None, help="synthesise a single sentence instead")
    ap.add_argument("--id", default="custom")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--pitch_control", type=float, default=1.0)
    ap.add_argument("--energy_control", type=float, default=1.0)
    ap.add_argument("--duration_control", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    icfg = cfg["inference"]
    from synthesis import run_test_set

    test_file = args.test_sentences or icfg["test_sentences"]
    if args.text:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        tmp.write("{}|{}\n".format(args.id, args.text))
        tmp.close()
        test_file = tmp.name
    run_test_set(
        cfg,
        test_file,
        args.output_dir or icfg["output_dir"],
        checkpoint=args.checkpoint or icfg.get("checkpoint", "best"),
        speaker=args.speaker or icfg.get("speaker", "auto"),
        p_control=args.pitch_control,
        e_control=args.energy_control,
        d_control=args.duration_control,
    )


if __name__ == "__main__":
    main()
