"""Inference + variance-adaptor analysis (NEW, replaces ming024/synthesize.py).

For every test sentence this writes
  test_generated_audios/<ID>.wav                 22.05 kHz speech (HiFi-GAN universal)
  test_generated_audios/<ID>_overview.png        mel + F0 + energy + durations in one figure
  variance_adaptor_plots/duration/<ID>.png       predicted duration per phone (bar chart)
  variance_adaptor_plots/pitch/<ID>.png          predicted phone-level F0 (Hz) vs F0 measured
                                                 on the synthesised audio, over the mel
  variance_adaptor_plots/energy/<ID>.png         predicted phone-level energy vs measured
  variance_adaptor_plots/<ID>_values.csv         the numbers behind the plots
  frontend_log.json                              text -> normalised text -> phones
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyworld as pw  # noqa: E402
import torch  # noqa: E402
from scipy.io import wavfile  # noqa: E402

import audio as Audio  # noqa: E402
from text import text_to_sequence, _symbol_to_id  # noqa: E402
from text.frontends import get_frontend  # noqa: E402
from utils.model import get_model, get_vocoder, vocoder_infer, latest_step, _torch_load  # noqa: E402

# colours from the validated reference palette (dataviz skill)
C_F0 = "#2a78d6"       # blue
C_ENERGY = "#eb6834"   # orange
C_DUR = "#1baf7a"      # aqua
C_MEASURED = "#52514e"  # secondary ink
C_PAUSE = "#c3c2b7"
INK = "#0b0b0b"
plt.rcParams.update({
    "axes.edgecolor": "#8a8984", "axes.labelcolor": INK, "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
})


class _Args:
    def __init__(self, restore_step):
        self.restore_step = restore_step


def resolve_checkpoint(train_config, which):
    ckpt_dir = train_config["path"]["ckpt_path"]
    if which in (None, "best"):
        bj = os.path.join(ckpt_dir, "best.json")
        if os.path.exists(bj):
            step = json.load(open(bj))["step"]
            if step and os.path.exists(os.path.join(ckpt_dir, "{}.pth.tar".format(step))):
                return int(step)
        which = "latest"
    if which == "latest":
        return latest_step(train_config)
    return int(which)


def load_model(cfg, checkpoint="best", device=None):
    from trainer import resolve_multi_speaker

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre, mod, tr = cfg["preprocess"], cfg["model"], cfg["train"]
    meta_dir = os.path.join(cfg["lang_dir"], "data_meta")
    if not os.path.exists(os.path.join(pre["path"]["preprocessed_path"], "stats.json")) \
            and os.path.exists(os.path.join(meta_dir, "stats.json")):
        # running from the GitHub repo without the 2 GB of features: the model only
        # needs stats.json (pitch/energy bins) and speakers.json, kept in data_meta/
        pre["path"]["preprocessed_path"] = meta_dir
    resolve_multi_speaker(pre, mod)
    if isinstance(checkpoint, str) and checkpoint.endswith((".pth", ".pth.tar", ".pt")):
        from model import FastSpeech2
        model = FastSpeech2(pre, mod).to(device)
        sd = _torch_load(checkpoint, device)["model"]
        model.load_state_dict({k: v.float() for k, v in sd.items()})
        model.eval()
        step = os.path.basename(checkpoint)
    else:
        step = resolve_checkpoint(tr, checkpoint)
        if not step:
            raise FileNotFoundError("No checkpoint found in " + tr["path"]["ckpt_path"])
        model = get_model(_Args(step), (pre, mod, tr), device, train=False)
    vocoder = get_vocoder(mod, device)
    return model, vocoder, step, device


def _stats(pre):
    with open(os.path.join(pre["path"]["preprocessed_path"], "stats.json")) as f:
        return json.load(f)


def _speaker_id(pre, speaker):
    spk_map = json.load(open(os.path.join(pre["path"]["preprocessed_path"], "speakers.json")))
    if speaker in (None, "auto"):
        summ = os.path.join(pre["path"]["preprocessed_path"], "preprocess_summary.json")
        speaker = json.load(open(summ)).get("default_speaker") if os.path.exists(summ) else None
        speaker = speaker if speaker in spk_map else sorted(spk_map)[0]
    if isinstance(speaker, int) or (isinstance(speaker, str) and speaker.isdigit()):
        return int(speaker), [k for k, v in spk_map.items() if v == int(speaker)][0]
    return spk_map[speaker], speaker


def synthesize(model, vocoder, cfg, frontend, text, speaker_id=0, device=None,
               p_control=1.0, e_control=1.0, d_control=1.0):
    pre, mod = cfg["preprocess"], cfg["model"]
    phones, spans = frontend(text, return_words=True)
    unknown = [p for p in phones if "@" + p not in _symbol_to_id]
    if unknown:
        print("WARNING: phones missing from symbol table (dropped):", unknown)
    seq = np.array(text_to_sequence("{" + " ".join(phones) + "}", pre["preprocessing"]["text"]["text_cleaners"]))
    speakers = torch.LongTensor([speaker_id]).to(device)
    texts = torch.from_numpy(seq).long().unsqueeze(0).to(device)
    src_lens = torch.LongTensor([len(seq)]).to(device)
    with torch.no_grad():
        out = model(speakers, texts, src_lens, len(seq),
                    p_control=p_control, e_control=e_control, d_control=d_control)
    mel = out[1][0, : out[9][0].item()].detach()  # postnet mel (T, 80)
    hop = pre["preprocessing"]["stft"]["hop_length"]
    sr = pre["preprocessing"]["audio"]["sampling_rate"]
    wav = vocoder_infer(mel.transpose(0, 1).unsqueeze(0), vocoder, mod, pre,
                        lengths=[mel.shape[0] * hop])[0]
    st = _stats(pre)
    p_mean, p_std = st["pitch"][2], st["pitch"][3]
    e_mean, e_std = st["energy"][2], st["energy"][3]
    n = len(seq)
    res = {
        "text": text,
        "normalised": frontend.normalize(text),
        "phones": phones,
        "spans": spans,
        "durations": out[5][0, :n].cpu().numpy().astype(int),
        "log_dur": out[4][0, :n].cpu().numpy(),
        "pitch_hz": out[2][0, :n].cpu().numpy() * p_std + p_mean,
        "energy": out[3][0, :n].cpu().numpy() * e_std + e_mean,
        "mel": mel.cpu().numpy().T,
        "wav": wav,
        "sr": sr,
        "hop": hop,
        "pitch_level": pre["preprocessing"]["pitch"]["feature"],
    }
    # measure what the vocoded audio actually contains
    x = wav.astype(np.float64) / 32768.0
    min_len = pre["preprocessing"]["stft"]["filter_length"] + 1
    if len(x) < min_len:  # an untrained model can predict (almost) zero durations
        x = np.pad(x, (0, min_len - len(x)))
    f0, t = pw.dio(x, sr, f0_floor=pre["preprocessing"]["pitch"].get("f0_floor", 71.0),
                   f0_ceil=pre["preprocessing"]["pitch"].get("f0_ceil", 800.0),
                   frame_period=hop / sr * 1000)
    res["f0_measured"] = pw.stonemask(x, f0, t, sr)[: res["mel"].shape[1]]
    stft = Audio.stft.TacotronSTFT(
        pre["preprocessing"]["stft"]["filter_length"], hop, pre["preprocessing"]["stft"]["win_length"],
        pre["preprocessing"]["mel"]["n_mel_channels"], sr,
        pre["preprocessing"]["mel"]["mel_fmin"], pre["preprocessing"]["mel"]["mel_fmax"])
    _, energy_meas = Audio.tools.get_mel_from_wav(np.clip(x, -1, 1).astype(np.float32), stft)
    res["energy_measured"] = energy_meas[: res["mel"].shape[1]]
    return res


# ----------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------
def _romanise(label, phones):
    """Readable word label for plots (matplotlib cannot shape Telugu script)."""
    if all(ord(c) < 128 for c in label):
        return label
    return "".join(phones)


def _boundaries(res):
    d = res["durations"]
    edges = np.concatenate([[0], np.cumsum(d)]) * res["hop"] / res["sr"]
    return edges


def _word_axis(ax, res, edges, y):
    for label, a, b in res["spans"]:
        if label == "<pause>":
            continue
        x0, x1 = edges[a], edges[b]
        ax.annotate("", xy=(x0, y), xytext=(x1, y), xycoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color="#8a8984", lw=0.8))
        ax.text((x0 + x1) / 2, y + 0.015, _romanise(label, res["phones"][a:b]),
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color=INK)


def _phone_ticks(ax, res, edges):
    centers = (edges[:-1] + edges[1:]) / 2
    ax.set_xticks(centers)
    ax.set_xticklabels(res["phones"], rotation=90, fontsize=6.5)


def _mel_panel(ax, res, title):
    dur_s = res["mel"].shape[1] * res["hop"] / res["sr"]
    ax.imshow(res["mel"], origin="lower", aspect="auto", cmap="magma",
              extent=[0, dur_s, 0, res["mel"].shape[0]], interpolation="nearest")
    ax.set_ylabel("mel bin")
    ax.set_title(title, loc="left")
    ax.tick_params(labelbottom=False)


def _expand(values, durations):
    return np.repeat(values, np.maximum(durations, 0))


def _fig_width(n_phones):
    return max(10.0, min(30.0, 0.17 * n_phones + 3))


def plot_duration(res, path, title):
    d = res["durations"]
    ms = d * res["hop"] / res["sr"] * 1000
    n = len(d)
    fig, ax = plt.subplots(figsize=(_fig_width(n), 4.2))
    colors = [C_PAUSE if p in ("sp", "sil") else C_DUR for p in res["phones"]]
    ax.bar(np.arange(n), ms, color=colors, width=0.8, edgecolor="white", linewidth=0.5)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(res["phones"], rotation=90, fontsize=6.5)
    ax.set_ylabel("predicted duration (ms)")
    ax.set_xlim(-0.6, n - 0.4)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    top = ms.max() if len(ms) else 1
    ax.set_ylim(0, top * 1.25)
    for label, a, b in res["spans"]:
        if label == "<pause>":
            continue
        ax.plot([a - 0.4, b - 0.6], [top * 1.08] * 2, color="#8a8984", lw=0.8)
        ax.text((a + b - 1) / 2, top * 1.1, _romanise(label, res["phones"][a:b]),
                ha="center", va="bottom", fontsize=7)
    total = d.sum() * res["hop"] / res["sr"]
    ax.set_title("{}  |  {} phones, {:.2f} s total, mean {:.0f} ms/phone".format(
        title, n, total, ms.mean()), loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _contour_plot(res, path, title, pred_phone, measured, ylabel, color, unit_fmt):
    edges = _boundaries(res)
    n_frames = res["mel"].shape[1]
    t = np.arange(n_frames) * res["hop"] / res["sr"]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(_fig_width(len(res["phones"])), 6.4),
                                   sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    _mel_panel(ax0, res, title)
    for e in edges:
        ax1.axvline(e, color="#ecebe6", lw=0.5, zorder=0)
    # predicted: one value per phone -> step function
    pred = _expand(pred_phone, res["durations"])[:n_frames]
    ax1.plot(t[: len(pred)], pred, color=color, lw=2, label="predicted (phone level)")
    meas = measured.astype(float).copy()
    if ylabel.startswith("F0"):
        meas[meas <= 0] = np.nan
    ax1.plot(t[: len(meas)], meas, color=C_MEASURED, lw=1, ls="--", label="measured on synthesised audio")
    _phone_ticks(ax1, res, edges)
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel("phone (time axis in seconds of generated speech)")
    ax1.grid(axis="y", color="#e6e5e0", lw=0.6)
    _word_axis(ax1, res, edges, 1.0)
    ax1.legend(loc="upper right", fontsize=7, frameon=False, bbox_to_anchor=(1.0, 1.18), ncol=2)
    voiced = pred_phone[[p not in ("sp", "sil") for p in res["phones"]]]
    if len(voiced):
        ax1.text(0.0, -0.32, ("predicted range " + unit_fmt + " - " + unit_fmt + ", mean " + unit_fmt).format(
            voiced.min(), voiced.max(), voiced.mean()), transform=ax1.transAxes, fontsize=7.5, color="#52514e")
    ax1.set_xlim(0, t[-1] if len(t) else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pitch(res, path, title):
    _contour_plot(res, path, title, res["pitch_hz"], res["f0_measured"], "F0 (Hz)", C_F0, "{:.0f} Hz")


def plot_energy(res, path, title):
    _contour_plot(res, path, title, res["energy"], res["energy_measured"], "energy (L2 norm of STFT frame)",
                  C_ENERGY, "{:.1f}")


def plot_overview(res, path, title):
    edges = _boundaries(res)
    n_frames = res["mel"].shape[1]
    t = np.arange(n_frames) * res["hop"] / res["sr"]
    fig, axes = plt.subplots(4, 1, figsize=(_fig_width(len(res["phones"])), 9), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1, 1, 0.8]})
    _mel_panel(axes[0], res, title)
    for ax, vals, c, lab in [(axes[1], res["pitch_hz"], C_F0, "F0 (Hz)"),
                             (axes[2], res["energy"], C_ENERGY, "energy")]:
        ax.plot(t, _expand(vals, res["durations"])[:n_frames], color=c, lw=2)
        ax.set_ylabel(lab)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ms = res["durations"] * res["hop"] / res["sr"] * 1000
    axes[3].bar((edges[:-1] + edges[1:]) / 2, ms, width=np.maximum(np.diff(edges) * 0.85, 1e-3),
                color=[C_PAUSE if p in ("sp", "sil") else C_DUR for p in res["phones"]])
    axes[3].set_ylabel("duration (ms)")
    _phone_ticks(axes[3], res, edges)
    _word_axis(axes[1], res, edges, 1.0)
    axes[3].set_xlim(0, t[-1] if len(t) else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_values_csv(res, path):
    edges = np.concatenate([[0], np.cumsum(res["durations"])])
    word_of = {}
    for label, a, b in res["spans"]:
        for i in range(a, b):
            word_of[i] = label
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "phone", "word", "duration_frames", "duration_ms", "f0_pred_hz",
                    "f0_measured_hz_mean_voiced", "energy_pred", "energy_measured_mean"])
        for i, p in enumerate(res["phones"]):
            a, b = edges[i], edges[i + 1]
            seg = res["f0_measured"][a:b]
            seg = seg[seg > 0]
            eseg = res["energy_measured"][a:b]
            w.writerow([i, p, word_of.get(i, ""), int(res["durations"][i]),
                        round(float(res["durations"][i] * res["hop"] / res["sr"] * 1000), 1),
                        round(float(res["pitch_hz"][i]), 1),
                        round(float(seg.mean()), 1) if len(seg) else "",
                        round(float(res["energy"][i]), 3),
                        round(float(eseg.mean()), 3) if len(eseg) else ""])


def read_test_sentences(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            sid, text = line.split("|", 1)
            items.append((sid.strip(), text))
    return items


def run_test_set(cfg, test_file, out_dir, checkpoint="best", speaker="auto",
                 p_control=1.0, e_control=1.0, d_control=1.0):
    pre = cfg["preprocess"]
    language = pre["preprocessing"]["text"]["language"]
    frontend = get_frontend(language)
    model, vocoder, step, device = load_model(cfg, checkpoint)
    spk_id, spk_name = _speaker_id(pre, speaker)
    print("Checkpoint: {} | speaker: {} (id {})".format(step, spk_name, spk_id))

    audio_dir = os.path.join(out_dir, "test_generated_audios")
    plot_root = os.path.join(out_dir, "variance_adaptor_plots")
    for d in [audio_dir] + [os.path.join(plot_root, k) for k in ("duration", "pitch", "energy")]:
        os.makedirs(d, exist_ok=True)

    log = {"checkpoint": str(step), "speaker": spk_name, "speaker_id": spk_id,
           "controls": {"pitch": p_control, "energy": e_control, "duration": d_control}, "sentences": []}
    for sid, text in read_test_sentences(test_file):
        res = synthesize(model, vocoder, cfg, frontend, text, spk_id, device, p_control, e_control, d_control)
        wavfile.write(os.path.join(audio_dir, sid + ".wav"), res["sr"], res["wav"])
        title = "{} - {} (step {})".format(sid, cfg["preprocess"]["dataset"], step)
        plot_duration(res, os.path.join(plot_root, "duration", sid + ".png"), title + " - duration")
        plot_pitch(res, os.path.join(plot_root, "pitch", sid + ".png"), title + " - pitch (F0)")
        plot_energy(res, os.path.join(plot_root, "energy", sid + ".png"), title + " - energy")
        plot_overview(res, os.path.join(audio_dir, sid + "_overview.png"), title)
        write_values_csv(res, os.path.join(plot_root, sid + "_values.csv"))
        speech = res["durations"].sum() * res["hop"] / res["sr"]
        log["sentences"].append({
            "id": sid, "text": text, "normalised": res["normalised"],
            "phones": " ".join(res["phones"]), "n_phones": len(res["phones"]),
            "words": [(w, " ".join(res["phones"][a:b])) for w, a, b in res["spans"]],
            "audio_seconds": round(float(speech), 3),
            "mean_f0_pred_hz": round(float(np.mean(res["pitch_hz"])), 1),
        })
        print("{}: {:.2f} s | {}".format(sid, speech, " ".join(res["phones"])))
    with open(os.path.join(out_dir, "frontend_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    return log
