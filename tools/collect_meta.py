"""Copy the small preprocessing artefacts into <lang>/data_meta/ (NEW).

The audio and features live outside the repository (FS2_DATA_ROOT), but these
files are needed to document the data split and to run inference from the
repository alone (stats.json = pitch/energy normalisation, speakers.json =
speaker ids):

  data_selection_report.json  preprocess_summary.json  stats.json  speakers.json
  train.txt  val.txt  lexicon.txt  metadata.json

Usage: python tools/collect_meta.py --lang English
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fastspeech2_core"))
from config import load_config  # noqa: E402

FILES = ["data_selection_report.json", "preprocess_summary.json", "stats.json", "speakers.json",
         "train.txt", "val.txt", "metadata.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["English", "Telugu"])
    args = ap.parse_args()
    cfg = load_config(os.path.join(ROOT, args.lang, "config.yaml"))
    pre_dir = cfg["preprocess"]["path"]["preprocessed_path"]
    out = os.path.join(ROOT, args.lang, "data_meta")
    os.makedirs(out, exist_ok=True)
    for f in FILES:
        src = os.path.join(pre_dir, f)
        if os.path.exists(src):
            shutil.copy(src, out)
            print("copied", f)
    lex = cfg["preprocess"]["path"]["lexicon_path"]
    if os.path.exists(lex):
        shutil.copy(lex, os.path.join(out, "lexicon.txt"))
        print("copied lexicon.txt")


if __name__ == "__main__":
    main()
