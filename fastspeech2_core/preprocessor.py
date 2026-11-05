"""Feature extraction (adapted from ming024/preprocessor/preprocessor.py).

[MODIFIED] relative to ming024:
  * MFA >= 2 writes silences as EMPTY intervals (""), and `tgt` drops empty
    intervals by default. ming024 therefore lost every inter-word pause and the
    durations no longer summed to the mel length, silently mis-aligning the
    duration / pitch / energy targets. We read empty intervals and map them
    to "sp".
  * The train/val split is NOT random any more: it comes from the data
    selection step (metadata.json), which guarantees no overlap of utterances
    or transcripts and the requested ~5 h / ~1 h sizes.
  * Pitch / energy normalisation statistics are computed on the TRAINING set
    only (no information from validation leaks into the model).
  * Utterances with unknown words (spn), absurd phone durations (alignment
    failures) or more frames than max_seq_len are dropped and counted.
  * Feature extraction runs in a multiprocessing pool (pyworld F0 extraction is
    the bottleneck) and the STFT runs on CPU (ming024 hard-coded .cuda()).
  * F0 search range is configurable (dio defaults 71-800 Hz).
"""
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import librosa
import numpy as np
import pyworld as pw
import tgt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio

SIL_PHONES = ["sil", "sp", "spn", ""]

_worker = {}


def _init_worker(config):
    import torch

    torch.set_num_threads(1)
    _worker["pp"] = Preprocessor(config)


def _process(args):
    speaker, basename = args
    try:
        return basename, _worker["pp"].process_utterance(speaker, basename)
    except Exception as e:  # never let one bad file kill the whole run
        return basename, ("error", repr(e))


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.max_frames = config["preprocessing"].get("max_frames", 1500)
        self.max_phone_frames = config["preprocessing"].get("max_phone_frames", 100)
        self.f0_floor = config["preprocessing"]["pitch"].get("f0_floor", 71.0)
        self.f0_ceil = config["preprocessing"]["pitch"].get("f0_ceil", 800.0)

        assert config["preprocessing"]["pitch"]["feature"] in ["phoneme_level", "frame_level"]
        assert config["preprocessing"]["energy"]["feature"] in ["phoneme_level", "frame_level"]
        self.pitch_phoneme_averaging = config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        self.energy_phoneme_averaging = config["preprocessing"]["energy"]["feature"] == "phoneme_level"

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self, num_workers=4):
        for d in ["mel", "pitch", "energy", "duration"]:
            os.makedirs(os.path.join(self.out_dir, d), exist_ok=True)

        with open(os.path.join(self.out_dir, "metadata.json"), encoding="utf-8") as f:
            metadata = json.load(f)

        jobs = []
        missing_tg = 0
        for basename, m in sorted(metadata.items()):
            tg_path = os.path.join(self.out_dir, "TextGrid", m["speaker"], basename + ".TextGrid")
            if os.path.exists(tg_path):
                jobs.append((m["speaker"], basename))
            else:
                missing_tg += 1
        print("Processing {} utterances ({} have no TextGrid = not aligned by MFA)".format(
            len(jobs), missing_tg))

        results = {}
        if num_workers > 1:
            with Pool(num_workers, initializer=_init_worker, initargs=(self.config,)) as pool:
                for basename, ret in tqdm(pool.imap_unordered(_process, jobs, chunksize=8), total=len(jobs)):
                    results[basename] = ret
        else:
            _worker["pp"] = self
            for job in tqdm(jobs):
                basename, ret = _process(job)
                results[basename] = ret

        drop = defaultdict(int)
        drop["no_textgrid"] = missing_tg
        pitch_scaler, energy_scaler = StandardScaler(), StandardScaler()
        lines = {"train": [], "val": []}
        n_frames = {"train": 0, "val": 0}
        speakers = sorted({m["speaker"] for m in metadata.values()})
        spk_frames = defaultdict(lambda: {"train": 0, "val": 0})
        phone_counts = defaultdict(int)
        for basename in sorted(results):
            ret = results[basename]
            if isinstance(ret, tuple) and len(ret) == 2 and ret[0] == "error":
                drop["error"] += 1
                continue
            if isinstance(ret, str):
                drop[ret] += 1
                continue
            m = metadata[basename]
            split = m["split"]
            info, pitch, energy, n, phones = ret
            if split == "train":
                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))
                for p in phones:
                    phone_counts[p] += 1
            n_frames[split] += n
            spk_frames[m["speaker"]][split] += n
            lines[split].append(info + "|" + m["raw_text"])

        print("Computing statistic quantities ...")
        if self.pitch_normalization:
            pitch_mean, pitch_std = pitch_scaler.mean_[0], pitch_scaler.scale_[0]
        else:
            pitch_mean, pitch_std = 0, 1
        if self.energy_normalization:
            energy_mean, energy_std = energy_scaler.mean_[0], energy_scaler.scale_[0]
        else:
            energy_mean, energy_std = 0, 1

        pitch_min, pitch_max = self.normalize(os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std, results)
        energy_min, energy_max = self.normalize(os.path.join(self.out_dir, "energy"), energy_mean, energy_std, results)

        # speakers that survived (keep a stable, sorted mapping)
        used = sorted({l.split("|")[1] for s in lines for l in lines[s]})
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            json.dump({s: i for i, s in enumerate(used)}, f, indent=1)

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            json.dump({
                "pitch": [float(pitch_min), float(pitch_max), float(pitch_mean), float(pitch_std)],
                "energy": [float(energy_min), float(energy_max), float(energy_mean), float(energy_std)],
            }, f)

        hours = {s: n_frames[s] * self.hop_length / self.sampling_rate / 3600 for s in n_frames}
        spk_hours = {s: {k: round(v * self.hop_length / self.sampling_rate / 3600, 4) for k, v in d.items()}
                     for s, d in spk_frames.items()}
        default_speaker = max(spk_hours, key=lambda s: spk_hours[s]["train"]) if spk_hours else None
        summary = {
            "train_utterances": len(lines["train"]),
            "val_utterances": len(lines["val"]),
            "train_hours_after_trimming": round(hours["train"], 3),
            "val_hours_after_trimming": round(hours["val"], 3),
            "dropped": dict(drop),
            "n_speakers": len(used),
            "default_speaker": default_speaker,
            "per_speaker_hours": spk_hours,
            "phone_counts_train": dict(sorted(phone_counts.items(), key=lambda x: -x[1])),
            "pitch_mean_hz": float(pitch_mean), "pitch_std_hz": float(pitch_std),
            "energy_mean": float(energy_mean), "energy_std": float(energy_std),
        }
        with open(os.path.join(self.out_dir, "preprocess_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, ensure_ascii=False)
        print(json.dumps({k: v for k, v in summary.items() if k not in ("per_speaker_hours", "phone_counts_train")}, indent=1))

        for split in ["train", "val"]:
            with open(os.path.join(self.out_dir, split + ".txt"), "w", encoding="utf-8") as f:
                for line in lines[split]:
                    f.write(line + "\n")
        return summary

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        tg_path = os.path.join(self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename))

        # Get alignments  [MODIFIED] keep empty intervals (= silences in MFA>=2)
        textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
        tier_names = [t.name for t in textgrid.tiers]
        tier = textgrid.get_tier_by_name("phones" if "phones" in tier_names else tier_names[-1])
        phone, duration, start, end = self.get_alignment(tier)
        if start >= end or len(phone) == 0:
            return "empty_alignment"
        if "spn" in phone:
            return "unknown_word_spn"
        if max(d for p, d in zip(phone, duration) if p not in SIL_PHONES) > self.max_phone_frames:
            return "implausible_phone_duration"
        if sum(duration) > self.max_frames:
            return "too_long"
        text = "{" + " ".join(phone) + "}"

        # Read and trim wav files
        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav[int(self.sampling_rate * start): int(self.sampling_rate * end)].astype(np.float32)

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            f0_floor=self.f0_floor,
            f0_ceil=self.f0_ceil,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return "no_voiced_frames"

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]
        if mel_spectrogram.shape[1] < sum(duration):
            return "audio_shorter_than_alignment"

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos: pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos: pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        np.save(os.path.join(self.out_dir, "duration", "{}-duration-{}.npy".format(speaker, basename)), duration)
        np.save(os.path.join(self.out_dir, "pitch", "{}-pitch-{}.npy".format(speaker, basename)), pitch)
        np.save(os.path.join(self.out_dir, "energy", "{}-energy-{}.npy".format(speaker, basename)), energy)
        np.save(os.path.join(self.out_dir, "mel", "{}-mel-{}.npy".format(speaker, basename)), mel_spectrogram.T)

        return (
            "|".join([basename, speaker, text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
            phone,
        )

    def get_alignment(self, tier):
        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text.strip()
            if p == "":
                p = "sp"  # [MODIFIED] MFA>=2 silence

            # Trim leading silences
            if phones == []:
                if p in SIL_PHONES:
                    continue
                else:
                    start_time = s

            if p not in SIL_PHONES:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        # [MODIFIED] merge consecutive silences ("sp" "sil") into one token
        merged_p, merged_d = [], []
        for p, d in zip(phones, durations):
            if merged_p and p in ("sp", "sil", "") and merged_p[-1] in ("sp", "sil", ""):
                merged_d[-1] += d
            else:
                merged_p.append("sp" if p in ("sil", "") else p)
                merged_d.append(d)
        return merged_p, merged_d, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std, results):
        """[MODIFIED] only touch files produced in this run (re-runs are safe)."""
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        kind = os.path.basename(in_dir)
        for basename, ret in results.items():
            if not (isinstance(ret, tuple) and len(ret) == 5):
                continue
            speaker = ret[0].split("|")[1]
            filename = os.path.join(in_dir, "{}-{}-{}.npy".format(speaker, kind, basename))
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
