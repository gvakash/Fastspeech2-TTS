"""Plot training and validation loss curves for one language (NEW).

Reads <lang>/logs/train_losses.csv and val_losses.csv written by the trainer
(falls back to ming024's log.txt files) and writes, into Results/<lang>/:
  training_loss.png         total + each component (mel, postnet, pitch, energy, duration)
  validation_loss.png       same for the validation set
  train_vs_val_loss.png     total and mel loss, train vs val (overfitting check)
  loss_summary.json         numbers quoted in the report

Usage: python tools/plot_losses.py --lang English
"""
import argparse
import csv
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]
LABELS = {"total": "total", "mel": "mel (L1, before postnet)", "mel_postnet": "mel (L1, after postnet)",
          "pitch": "pitch (MSE)", "energy": "energy (MSE)", "duration": "log-duration (MSE)"}
# validated categorical palette (dataviz reference instance), fixed order
COLORS = {"total": "#0b0b0b", "mel": "#2a78d6", "mel_postnet": "#eb6834", "pitch": "#1baf7a",
          "energy": "#eda100", "duration": "#6250d6"}
TRAIN_C, VAL_C = "#2a78d6", "#eb6834"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.titleweight": "bold", "axes.edgecolor": "#8a8984"})


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    data = {"step": np.array([int(r["step"]) for r in rows])}
    for n in NAMES:
        data[n] = np.array([float(r[n]) for r in rows])
    order = np.argsort(data["step"], kind="stable")
    # if a run was resumed, keep the last value written for each step
    _, last = np.unique(data["step"][order][::-1], return_index=True)
    keep = order[::-1][last]
    keep = np.sort(keep)
    return {k: v[keep] for k, v in data.items()}


def read_logtxt(path):
    if not os.path.exists(path):
        return None
    pat = re.compile(r"Step (\d+).*?Total Loss: ([\d.]+), Mel Loss: ([\d.]+), Mel PostNet Loss: ([\d.]+), "
                     r"Pitch Loss: ([\d.]+), Energy Loss: ([\d.]+), Duration Loss: ([\d.]+)")
    rows = [list(map(float, m.groups())) for m in map(pat.search, open(path)) if m]
    if not rows:
        return None
    a = np.array(rows)
    out = {"step": a[:, 0].astype(int)}
    for i, n in enumerate(NAMES):
        out[n] = a[:, i + 1]
    return out


def smooth(y, k):
    if len(y) < k or k <= 1:
        return y
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def plot_components(d, title, path, smooth_k=1):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for ax, n in zip(axes.ravel(), NAMES):
        y = d[n]
        if smooth_k > 1:
            ax.plot(d["step"], y, color=COLORS[n], alpha=0.25, lw=1)
            ax.plot(d["step"], smooth(y, smooth_k), color=COLORS[n], lw=2)
        else:
            ax.plot(d["step"], y, color=COLORS[n], lw=2, marker="o", ms=3)
        ax.set_title(LABELS[n], loc="left", fontsize=10)
        ax.grid(color="#e6e5e0", lw=0.6)
        ax.set_xlabel("training step")
        lo, hi = np.percentile(y, 1), np.percentile(y, 99)
        ax.set_ylim(max(0, lo - 0.1 * (hi - lo)), hi + 0.15 * (hi - lo))
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["English", "Telugu"])
    ap.add_argument("--log_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    log_dir = args.log_dir or os.path.join(ROOT, args.lang, "logs")
    out_dir = args.out_dir or os.path.join(ROOT, "Results", args.lang)
    os.makedirs(out_dir, exist_ok=True)

    tr = read_csv(os.path.join(log_dir, "train_losses.csv")) or read_logtxt(os.path.join(log_dir, "train", "log.txt"))
    va = read_csv(os.path.join(log_dir, "val_losses.csv")) or read_logtxt(os.path.join(log_dir, "val", "log.txt"))
    if tr is None and va is None:
        print("WARNING: no loss logs found in " + log_dir)
        return
    if tr is None or va is None:
        print("WARNING: only {} losses found - plotting what exists".format("validation" if tr is None else "training"))
        if tr is not None:
            plot_components(tr, "{} - training loss".format(args.lang), os.path.join(out_dir, "training_loss.png"))
        if va is not None:
            plot_components(va, "{} - validation loss".format(args.lang), os.path.join(out_dir, "validation_loss.png"))
        return

    k = max(1, len(tr["step"]) // 60)
    plot_components(tr, "{} - training loss (per-batch values, faint; moving average, bold)".format(args.lang),
                    os.path.join(out_dir, "training_loss.png"), smooth_k=k)
    plot_components(va, "{} - validation loss (mean over the whole validation set)".format(args.lang),
                    os.path.join(out_dir, "validation_loss.png"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, n in zip(axes, ["total", "mel_postnet", "duration"]):
        ax.plot(tr["step"], smooth(tr[n], k), color=TRAIN_C, lw=2, label="train (moving avg.)")
        ax.plot(va["step"], va[n], color=VAL_C, lw=2, marker="o", ms=3, label="validation")
        b = int(np.argmin(va[n]))
        ax.annotate("min {:.3f}\n@ {}".format(va[n][b], va["step"][b]), (va["step"][b], va[n][b]),
                    xytext=(0, 18), textcoords="offset points", ha="center", fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color="#8a8984", lw=0.6))
        ax.set_title(LABELS[n], loc="left", fontsize=10)
        ax.set_xlabel("training step")
        ax.grid(color="#e6e5e0", lw=0.6)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("{} - train vs validation".format(args.lang), x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "train_vs_val_loss.png"), dpi=150)
    plt.close(fig)

    summary = {"last_step": int(tr["step"][-1]), "n_val_points": int(len(va["step"]))}
    for n in NAMES:
        b = int(np.argmin(va[n]))
        summary[n] = {
            "train_first": round(float(np.mean(tr[n][:3])), 4),
            "train_last_avg": round(float(np.mean(tr[n][-10:])), 4),
            "val_first": round(float(va[n][0]), 4),
            "val_best": round(float(va[n][b]), 4), "val_best_step": int(va["step"][b]),
            "val_last": round(float(va[n][-1]), 4),
            "gap_last_val_minus_train": round(float(va[n][-1] - np.mean(tr[n][-10:])), 4),
        }
    with open(os.path.join(out_dir, "loss_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
