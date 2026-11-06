"""Generates kaggle_fastspeech2.ipynb (run: python notebooks/make_notebook.py)."""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# FastSpeech 2 — English (Svarah) / Telugu (IndicTTS) on a free Kaggle GPU

This one notebook does **everything** for ONE language: install → download data → select 5 h / 1 h →
forced alignment (MFA) → feature extraction → smoke test → training → loss plots → test-sentence
synthesis + variance-adaptor plots → a zip to download.

**Before you press "Save Version → Save & Run All (Commit)":**
1. Right panel → *Session options* → **Accelerator: GPU P100** (or GPU T4 x2) and **Internet: ON**
   (internet needs a phone-verified Kaggle account).
2. *Add-ons → Secrets* → add `HF_TOKEN` (a Hugging Face *read* token) and tick it for this notebook.
   For English you must first click **"Agree and access repository"** on
   https://huggingface.co/datasets/ai4bharat/Svarah while logged in to Hugging Face.
3. Set `LANG` and `REPO_URL` in the next cell.

Make one copy of this notebook per language and run both (they can run at the same time).
A committed run continues in the background even if you close the browser (max 12 h).
""")

code("""
# ======================= SETTINGS =======================
LANG = "English"          # "English" or "Telugu"
REPO_URL = "https://github.com/YOUR_GITHUB_USERNAME/fastspeech2-english-telugu.git"
TRAIN_HOURS = 7.0         # wall-clock budget for training (keep total < 12 h Kaggle limit)
# =========================================================
import os, subprocess, sys, time
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
PROJECT = os.path.join(WORK, "project")
os.environ["FS2_DATA_ROOT"] = "/tmp/fs2_data"        # audio + features (not saved as output)
os.environ["MFA_ROOT_DIR"] = "/tmp/fs2_data/mfa_root"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TQDM_MININTERVAL"] = "30"                # keep the log readable
T0 = time.time()


def sh(cmd):
    # run a shell command, stream its output, stop the notebook if it fails
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError("command failed (exit %d): %s" % (p.returncode, cmd))


print(LANG, WORK)
""")

code("""
# Hugging Face token (Kaggle secret -> environment variable)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle secrets")
except Exception as e:
    print("No Kaggle secret found:", e)
    # os.environ["HF_TOKEN"] = "hf_..."   # <- or paste it here (do NOT commit it to GitHub)
""")

code("""
# Get the code
if not os.path.isdir(PROJECT):
    sh(f"git clone --depth 1 {REPO_URL} {PROJECT}")
os.chdir(PROJECT)
sh("git log --oneline -1")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv || echo 'NO GPU - enable it in Session options!'")
""")

code("""
# Python dependencies (torch, librosa, numpy... are pre-installed on Kaggle)
sh("pip install -q pyworld tgt g2p_en==2.1.0 inflect unidecode huggingface_hub pyarrow soundfile")
import nltk
for r in ["cmudict", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "punkt"]:
    nltk.download(r, quiet=True)
""")

code("""
# Montreal Forced Aligner in its own conda env (~3 min). Only needed for preprocessing.
MFA_ENV = "/tmp/mfa_env"
if not os.path.exists(MFA_ENV + "/bin/mfa"):
    sh("cd /tmp && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba")
    sh(f"MAMBA_ROOT_PREFIX=/tmp/mamba /tmp/bin/micromamba create -y -q -p {MFA_ENV} "
       "-c conda-forge montreal-forced-aligner python=3.11 > /tmp/mfa_install.log 2>&1 || "
       "(tail -n 30 /tmp/mfa_install.log; exit 1)")
os.environ["MFA_BIN"] = MFA_ENV + "/bin/mfa"
sh(f"{MFA_ENV}/bin/mfa version")
""")

md("""
## 1. Preprocessing
`prepare` (download + select 5 h / 1 h) → `lexicon` → `align` (MFA) → `features`.
English alignment uses the pretrained `english_us_arpa` model (~15 min);
Telugu trains its own MFA model on the selected data (~30–90 min).
""")

code("""
sh(f"python {LANG}/preprocess.py --config {LANG}/config.yaml --stage all")
sh(f"python tools/collect_meta.py --lang {LANG}")
print("elapsed: %.2f h" % ((time.time() - T0) / 3600))
""")

md("""
## 2. Smoke test (2 min)
20 training steps + synthesis of the test sentences, to make sure the whole chain works
*before* spending hours on training. Results go to `/tmp/smoke` and are thrown away.
""")

code("""
sh(f"python {LANG}/train.py --config {LANG}/config.yaml --smoke")
sh(f"python {LANG}/inference.py --config {LANG}/config.yaml "
   f"--checkpoint {LANG}/checkpoints_smoke/20.pth.tar --output_dir /tmp/smoke")
sh(f"rm -rf {LANG}/checkpoints_smoke {LANG}/logs_smoke")
print("smoke test passed")
""")

md("""
## 3. Training
Stops by itself after `TRAIN_HOURS` (or 100k steps), saving a checkpoint. Validation loss is computed
on the whole 1 h validation set every 1000 steps; the best checkpoint is remembered in `best.json`.
""")

code("""
budget = max(0.5, min(TRAIN_HOURS, 11.3 - (time.time() - T0) / 3600))  # stay under Kaggle's 12 h
print("training budget: %.2f h" % budget)
sh(f"python {LANG}/train.py --config {LANG}/config.yaml --max_hours {budget:.2f} --restore_step latest")
""")

md("## 4. Loss curves, test sentences, variance-adaptor plots, small checkpoint")

code("""
sh(f"python tools/plot_losses.py --lang {LANG}")
sh(f"python {LANG}/inference.py --config {LANG}/config.yaml --checkpoint best")
sh(f"python tools/export_checkpoint.py --lang {LANG} --step best")
sh(f"ls -la {LANG}/checkpoints Results/{LANG} Results/{LANG}/test_generated_audios")
""")

code("""
# Listen to the results right here
from IPython.display import Audio, Image, display
import glob
for w in sorted(glob.glob(f"Results/{LANG}/test_generated_audios/*.wav")):
    print(os.path.basename(w)); display(Audio(w))
display(Image(f"Results/{LANG}/training_loss.png"))
display(Image(f"Results/{LANG}/validation_loss.png"))
""")

code("""
# Package everything you need for GitHub into one zip (Output tab -> download)
import shutil, glob
out = f"/kaggle/working/{LANG}_outputs" if os.path.isdir("/kaggle/working") else f"/tmp/{LANG}_outputs"
shutil.rmtree(out, ignore_errors=True)
shutil.copytree(f"Results/{LANG}", f"{out}/Results/{LANG}")
shutil.copytree(f"{LANG}/data_meta", f"{out}/{LANG}/data_meta")
shutil.copytree(f"{LANG}/logs", f"{out}/{LANG}/logs")
os.makedirs(f"{out}/{LANG}/checkpoints", exist_ok=True)
for f in glob.glob(f"{LANG}/checkpoints/*fp16*") + glob.glob(f"{LANG}/checkpoints/*.json"):
    shutil.copy(f, f"{out}/{LANG}/checkpoints/")
shutil.make_archive(out, "zip", out)
print("download:", out + ".zip", "%.1f MB" % (os.path.getsize(out + ".zip") / 2**20))
print("full fp32 checkpoints (with optimizer) stay in", f"{PROJECT}/{LANG}/checkpoints")
print("total elapsed: %.2f h" % ((time.time() - T0) / 3600))
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_fastspeech2.ipynb")
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", path)
