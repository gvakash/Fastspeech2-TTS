"""Preprocessing for the English (Svarah) FastSpeech 2 model.

Stages (run all with --stage all, or one at a time):
  prepare   download Svarah from Hugging Face, filter, pick ~5 h train / ~1 h val,
            write 22.05 kHz wavs + normalised transcripts (.lab)
  lexicon   build the pronunciation dictionary (CMUdict variants + g2p_en for OOVs)
  align     Montreal Forced Aligner -> phone TextGrids (pretrained english_us_arpa)
  features  mel-spectrogram, phone-level pitch/energy, durations, stats, train/val lists

Usage:
  python English/preprocess.py --config English/config.yaml --stage all
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fastspeech2_core"))

from config import load_config  # noqa: E402

STAGES = ["prepare", "lexicon", "align", "features"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--stage", default="all", choices=STAGES + ["all"])
    ap.add_argument("--overwrite", action="store_true", help="redo a stage even if its output exists")
    ap.add_argument("--num_workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    stages = STAGES if args.stage == "all" else [args.stage]
    language = cfg["preprocess"]["preprocessing"]["text"]["language"]
    print("Language: {} | data root: {}".format(language, cfg["data_root"]))

    frontend = None
    if "prepare" in stages or "lexicon" in stages:
        from text.frontends import get_frontend
        frontend = get_frontend(language)

    if "prepare" in stages:
        from data_prep import prepare_corpus
        prepare_corpus(cfg, frontend, overwrite=args.overwrite)
    if "lexicon" in stages:
        from data_prep import build_lexicon
        build_lexicon(cfg, frontend)
    if "align" in stages:
        from align import run_mfa
        run_mfa(cfg, num_jobs=args.num_workers, overwrite=args.overwrite)
    if "features" in stages:
        from preprocessor import Preprocessor
        nw = args.num_workers or cfg["preprocess"]["preprocessing"].get("num_workers", 4)
        Preprocessor(cfg["preprocess"]).build_from_path(num_workers=nw)


if __name__ == "__main__":
    main()
