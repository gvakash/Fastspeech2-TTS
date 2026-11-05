"""Dataset download, filtering, sub-sampling and train/validation split (NEW).

Stage 1 of preprocessing. Produces the ming024/MFA corpus layout

    <raw_path>/<speaker>/<utt_id>.wav   (22.05 kHz, mono, 16-bit, peak-normalised)
    <raw_path>/<speaker>/<utt_id>.lab   (normalised word transcript for the aligner)

plus <preprocessed_path>/metadata.json (original text, normalised text,
speaker, duration, split) and data_selection_report.json (all statistics that
the technical report needs).

Selection strategy (identical code for both languages, parameters in config):
  1. read every row of the Hugging Face parquet shards (audio decoded with
     soundfile, so no torchcodec/ffmpeg dependency);
  2. drop clips that are too short/long, have an empty/invalid transcript or
     (English) contain Latin-script problems / (Telugu) contain Latin letters;
  3. compute a cheap SNR proxy (95th - 10th percentile of frame energy, dB) and
     drop the noisiest fraction of clips;
  4. group clips by normalised transcript so the same sentence can never be in
     both splits; shuffle groups with a fixed seed;
  5. fill the validation split first (~1 h) and then the training split (~5 h)
     by round-robin over speakers, so every speaker contributes to both splits
     and no speaker dominates. Clips that are not selected are not used.
"""
import glob
import io
import json
import os
import random
import re
import unicodedata
from collections import defaultdict, OrderedDict

import numpy as np
import soundfile as sf
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Hugging Face download + parquet iteration
# ----------------------------------------------------------------------------
def download_parquet(repo_id, local_dir, token=None):
    from huggingface_hub import snapshot_download

    os.makedirs(local_dir, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["*.parquet", "**/*.parquet", "README.md"],
        local_dir=local_dir,
        token=token,
    )
    files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    if not files:
        raise RuntimeError(
            "No parquet files found for {}. If the dataset is gated, accept its "
            "terms on huggingface.co and set the HF_TOKEN secret.".format(repo_id)
        )
    return files


def iter_rows(parquet_files, columns, batch_size=64):
    import pyarrow.parquet as pq

    for fpath in parquet_files:
        pf = pq.ParquetFile(fpath)
        cols = [c for c in columns if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            for row in batch.to_pylist():
                yield os.path.basename(fpath), row


def parquet_columns(parquet_files):
    import pyarrow.parquet as pq

    return pq.ParquetFile(parquet_files[0]).schema_arrow.names


def decode_audio(cell):
    """HF Audio cell ({'bytes':..., 'path':...}) -> (float32 mono, sr)."""
    if isinstance(cell, dict):
        data = cell.get("bytes")
        if data is None and cell.get("path") and os.path.exists(cell["path"]):
            data = open(cell["path"], "rb").read()
    else:
        data = cell
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return wav.mean(axis=1), sr


def snr_proxy_db(wav, sr):
    """Rough SNR estimate: loud-frame energy minus quiet-frame energy (dB)."""
    win = int(0.025 * sr)
    hop = int(0.010 * sr)
    if len(wav) < win * 4:
        return 0.0
    n = 1 + (len(wav) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = wav[idx]
    e = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
    return float(np.percentile(e, 95) - np.percentile(e, 10))


# ----------------------------------------------------------------------------
# Selection and split
# ----------------------------------------------------------------------------
def select_and_split(cands, train_hours, val_hours, seed):
    """cands: list of dicts with keys id, speaker, duration, norm_text."""
    rng = random.Random(seed)
    groups = defaultdict(list)  # normalised text -> clips
    for c in cands:
        groups[c["norm_text"]].append(c)
    # one group = all clips with the same sentence; assign the group to a speaker
    # bucket by its first clip (most groups have exactly one clip)
    by_spk = defaultdict(list)
    for key in sorted(groups):
        g = groups[key]
        by_spk[g[0]["speaker"]].append(g)
    for s in by_spk:
        rng.shuffle(by_spk[s])
    speakers = sorted(by_spk)
    rng.shuffle(speakers)

    def fill(target_sec, split, min_left=1):
        total, chosen = 0.0, []
        progress = True
        while total < target_sec and progress:
            progress = False
            for s in speakers:
                # keep at least `min_left` groups per speaker for the other split
                if len(by_spk[s]) <= min_left:
                    continue
                g = by_spk[s].pop()
                for c in g:
                    c["split"] = split
                    chosen.append(c)
                    total += c["duration"]
                progress = True
                if total >= target_sec:
                    break
        return chosen, total

    val, val_sec = fill(val_hours * 3600, "val", min_left=2)
    train, train_sec = fill(train_hours * 3600, "train", min_left=0)
    return train, val, train_sec, val_sec


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------
def _speaker_id(row, dcfg):
    cols = dcfg.get("speaker_columns") or []
    if not cols:
        return dcfg.get("single_speaker_name", "spk0")
    vals = []
    for c in cols:
        v = row.get(c)
        if v is None:
            v = "na"
        if isinstance(v, int) and dcfg.get("speaker_label_names"):
            v = dcfg["speaker_label_names"][v]
        vals.append(str(v))
    if len(cols) == 1:
        return re.sub(r"[^A-Za-z0-9]+", "_", vals[0]).strip("_") or "spk0"
    import hashlib
    return "spk_" + hashlib.md5("|".join(vals).encode("utf-8")).hexdigest()[:8]


def prepare_corpus(cfg, frontend, overwrite=False):
    pcfg = cfg["preprocess"]
    dcfg = pcfg["data"]
    raw_path = pcfg["path"]["raw_path"]
    out_dir = pcfg["path"]["preprocessed_path"]
    meta_path = os.path.join(out_dir, "metadata.json")
    if os.path.exists(meta_path) and not overwrite:
        print("[prepare] metadata.json exists - skipping (use --overwrite to redo)")
        return json.load(open(meta_path, encoding="utf-8"))
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    if dcfg.get("local_parquet_glob"):  # already-downloaded shards (or a test fixture)
        files = sorted(glob.glob(dcfg["local_parquet_glob"]))
        print("[prepare] using {} local parquet file(s)".format(len(files)))
    else:
        token = os.environ.get(dcfg.get("hf_token_env", "HF_TOKEN")) or None
        print("[prepare] downloading {} ...".format(dcfg["hf_repo"]))
        files = download_parquet(dcfg["hf_repo"], dcfg["download_dir"], token)
    cols = parquet_columns(files)
    print("[prepare] parquet columns:", cols)
    audio_col, text_col = dcfg["audio_column"], dcfg["text_column"]
    for c in [audio_col, text_col]:
        if c not in cols:
            raise KeyError("column '{}' not in dataset columns {}".format(c, cols))
    spk_cols = [c for c in (dcfg.get("speaker_columns") or []) if c in cols]
    filter_cols = list((dcfg.get("filters") or {}).keys())
    read_cols = list(OrderedDict.fromkeys([audio_col, text_col] + spk_cols + filter_cols))

    stats = defaultdict(int)
    cands, total_sec = [], 0.0
    rows_meta = {}
    for i, (fname, row) in enumerate(tqdm(iter_rows(files, read_cols), desc="scan")):
        stats["rows"] += 1
        ok = True
        for col, allowed in (dcfg.get("filters") or {}).items():
            v = row.get(col)
            if isinstance(v, int) and dcfg.get("speaker_label_names") and col in dcfg.get("speaker_columns", []):
                v = dcfg["speaker_label_names"][v]
            if v not in allowed:
                ok = False
        if not ok:
            stats["drop_filter"] += 1
            continue
        raw_text = (row.get(text_col) or "").strip()
        raw_text = " ".join(unicodedata.normalize("NFC", raw_text).replace("|", " ").split())
        if not raw_text:
            stats["drop_empty_text"] += 1
            continue
        try:
            wav, sr = decode_audio(row[audio_col])
        except Exception:
            stats["drop_decode_error"] += 1
            continue
        dur = len(wav) / sr
        total_sec += dur
        if dur < dcfg["min_duration"] or dur > dcfg["max_duration"]:
            stats["drop_duration"] += 1
            continue
        ok, norm, reason = frontend.check_transcript(raw_text)
        if not ok:
            stats["drop_text_" + reason] += 1
            continue
        utt_id = "{}_{:05d}".format(dcfg["id_prefix"], i)
        spk = _speaker_id(row, dcfg)
        cands.append({
            "id": utt_id, "row": i, "speaker": spk, "duration": dur,
            "snr": snr_proxy_db(wav, sr), "norm_text": norm, "raw_text": raw_text,
            "orig_sr": sr,
        })
    print("[prepare] scanned {} rows ({:.2f} h); {} candidates".format(
        stats["rows"], total_sec / 3600, len(cands)))

    # preferred speakers (e.g. a single Telugu voice) if they have enough data
    pref = dcfg.get("preferred_speakers")
    need = (dcfg["train_hours"] + dcfg["val_hours"]) * 3600 / (1 - float(dcfg.get("drop_noisiest_fraction", 0.0)))
    if pref:
        pc = [c for c in cands if c["speaker"] in pref]
        ph = sum(c["duration"] for c in pc)
        if ph >= need * 1.02:
            print("[prepare] using preferred speaker(s) {} ({:.2f} h available)".format(pref, ph / 3600))
            stats["drop_not_preferred_speaker"] = len(cands) - len(pc)
            cands = pc
        else:
            print("[prepare] preferred speaker(s) {} have only {:.2f} h -> using all speakers".format(pref, ph / 3600))

    # SNR filtering
    frac = float(dcfg.get("drop_noisiest_fraction", 0.0))
    snr_thr = None
    if frac > 0 and cands:
        snr_thr = float(np.quantile([c["snr"] for c in cands], frac))
        before = len(cands)
        cands = [c for c in cands if c["snr"] >= snr_thr]
        stats["drop_snr"] = before - len(cands)

    cand_hours = sum(c["duration"] for c in cands) / 3600
    train, val, train_sec, val_sec = select_and_split(
        cands, dcfg["train_hours"], dcfg["val_hours"], dcfg["seed"])
    if train_sec < 0.8 * dcfg["train_hours"] * 3600:
        print("[prepare] WARNING: only {:.2f} h available for training".format(train_sec / 3600))
    chosen = {c["row"]: c for c in train + val}
    assert not ({c["id"] for c in train} & {c["id"] for c in val})
    assert not ({c["norm_text"] for c in train} & {c["norm_text"] for c in val})

    # second pass: write the selected clips
    target_sr = pcfg["preprocessing"]["audio"]["sampling_rate"]
    import librosa

    metadata = {}
    for i, (fname, row) in enumerate(tqdm(iter_rows(files, [audio_col]), desc="write", total=stats["rows"])):
        if i not in chosen:
            continue
        c = chosen[i]
        wav, sr = decode_audio(row[audio_col])
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        peak = np.max(np.abs(wav)) + 1e-9
        wav = wav / peak * 0.95
        spk_dir = os.path.join(raw_path, c["speaker"])
        os.makedirs(spk_dir, exist_ok=True)
        sf.write(os.path.join(spk_dir, c["id"] + ".wav"), wav, target_sr, subtype="PCM_16")
        with open(os.path.join(spk_dir, c["id"] + ".lab"), "w", encoding="utf-8") as f:
            f.write(" ".join(frontend.words_for_aligner(c["norm_text"])))
        metadata[c["id"]] = {k: c[k] for k in ["speaker", "duration", "snr", "raw_text", "norm_text", "split", "orig_sr"]}

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=1)

    spk_hours = defaultdict(lambda: {"train": 0.0, "val": 0.0})
    for m in metadata.values():
        spk_hours[m["speaker"]][m["split"]] += m["duration"] / 3600
    report = {
        "dataset": dcfg["hf_repo"],
        "rows_scanned": stats["rows"],
        "hours_scanned": round(total_sec / 3600, 3),
        "dropped": {k: v for k, v in stats.items() if k.startswith("drop")},
        "snr_threshold_db": snr_thr,
        "candidate_hours_after_filtering": round(cand_hours, 3),
        "train": {"utterances": len(train), "hours": round(train_sec / 3600, 3),
                  "speakers": len({c["speaker"] for c in train})},
        "val": {"utterances": len(val), "hours": round(val_sec / 3600, 3),
                "speakers": len({c["speaker"] for c in val})},
        "overlap_check": "no shared utterance ids and no shared transcripts between train and val",
        "duration_filter_sec": [dcfg["min_duration"], dcfg["max_duration"]],
        "seed": dcfg["seed"],
        "per_speaker_hours": {k: {s: round(v, 3) for s, v in d.items()} for k, d in sorted(spk_hours.items())},
    }
    with open(os.path.join(out_dir, "data_selection_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "per_speaker_hours"}, indent=1))
    return metadata


# ----------------------------------------------------------------------------
# Pronunciation dictionary for the aligner
# ----------------------------------------------------------------------------
def build_lexicon(cfg, frontend):
    pcfg = cfg["preprocess"]
    raw_path = pcfg["path"]["raw_path"]
    words = set()
    for lab in glob.glob(os.path.join(raw_path, "*", "*.lab")):
        words.update(open(lab, encoding="utf-8").read().split())
    lex_path = pcfg["path"]["lexicon_path"]
    os.makedirs(os.path.dirname(lex_path), exist_ok=True)
    n = 0
    with open(lex_path, "w", encoding="utf-8") as f:
        for w in tqdm(sorted(words), desc="lexicon"):
            for pron in frontend.word_prons(w):
                if pron.strip():
                    f.write("{}\t{}\n".format(w, pron))
                    n += 1
    print("[lexicon] {} words, {} pronunciations -> {}".format(len(words), n, lex_path))
    return lex_path
