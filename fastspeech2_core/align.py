"""Forced alignment with the Montreal Forced Aligner (NEW wrapper).

FastSpeech 2 needs a duration (in frames) for every input phone. ming024 ships
pre-computed TextGrids for LJSpeech only, so for Svarah and IndicTTS-Telugu we
create them:

  English  `mfa align`  with the pretrained `english_us_arpa` acoustic model and
           a dictionary built by our English front-end (all CMUdict variants +
           g2p_en predictions for OOV words), so MFA picks the variant that was
           actually spoken (e.g. noun vs. verb "conduct").
  Telugu   no pretrained Telugu MFA model exists, so `mfa train` learns a
           GMM-HMM acoustic model on the selected Telugu corpus itself (using the
           rule-based Telugu lexicon) and exports the alignments.
"""
import glob
import os
import shutil
import subprocess


def _mfa_bin(cfg):
    acfg = cfg["preprocess"]["alignment"]
    return os.environ.get("MFA_BIN") or acfg.get("mfa_bin") or "mfa"


def run_mfa(cfg, num_jobs=None, overwrite=False):
    pcfg = cfg["preprocess"]
    acfg = pcfg["alignment"]
    raw_path = pcfg["path"]["raw_path"]
    out_dir = os.path.join(pcfg["path"]["preprocessed_path"], "TextGrid")
    lexicon = pcfg["path"]["lexicon_path"]
    existing = glob.glob(os.path.join(out_dir, "*", "*.TextGrid"))
    if existing and not overwrite:
        print("[align] {} TextGrids already exist - skipping (use --overwrite)".format(len(existing)))
        return out_dir
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    mfa = _mfa_bin(cfg)
    jobs = str(num_jobs or acfg.get("num_jobs") or os.cpu_count() or 2)
    temp_dir = os.path.join(cfg["data_root"], "mfa_tmp")
    env = dict(os.environ)
    env.setdefault("MFA_ROOT_DIR", os.path.join(cfg["data_root"], "mfa_root"))
    if os.path.isabs(mfa):  # MFA calls kaldi/openfst binaries that live next to it
        env["PATH"] = os.path.dirname(mfa) + os.pathsep + env.get("PATH", "")
    common = ["-j", jobs, "--clean", "--overwrite", "--temporary_directory", temp_dir,
              "--beam", str(acfg.get("beam", 100)), "--retry_beam", str(acfg.get("retry_beam", 400))]

    n_speakers = len([d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))])
    if n_speakers == 1:
        # otherwise MFA parallelises per speaker and would use a single CPU core
        common.append("--single_speaker")

    if acfg["mode"] == "align":
        model = acfg["pretrained_acoustic_model"]
        if not os.path.exists(model):
            print("[align] downloading pretrained MFA model:", model)
            subprocess.run([mfa, "model", "download", "acoustic", model], check=True, env=env)
        cmd = [mfa, "align", raw_path, lexicon, model, out_dir] + common
    elif acfg["mode"] == "train":
        model_out = os.path.join(pcfg["path"]["preprocessed_path"], "mfa_acoustic_model.zip")
        cmd = [mfa, "train", raw_path, lexicon, model_out, "--output_directory", out_dir] + common
    else:
        raise ValueError("alignment.mode must be 'align' or 'train'")
    print("[align] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    n = len(glob.glob(os.path.join(out_dir, "*", "*.TextGrid")))
    n_lab = len(glob.glob(os.path.join(raw_path, "*", "*.lab")))
    print("[align] {} / {} utterances aligned".format(n, n_lab))
    if n == 0:
        raise RuntimeError("MFA produced no TextGrids - check the log above")
    return out_dir
