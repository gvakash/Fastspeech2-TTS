"""Training loop (adapted from ming024/train.py).

[MODIFIED] relative to ming024:
  * one config.yaml instead of three files;
  * `--max_hours`: stop cleanly (and save) before a Kaggle/Colab session limit;
  * `--restore_step latest` resumes from the newest checkpoint;
  * losses are also written to CSV (logs/train_losses.csv, logs/val_losses.csv)
    for the loss-curve plots; learning rate is logged;
  * best checkpoint (lowest validation total loss) is tracked in best.json;
    only the newest `keep_last` periodic checkpoints are kept on disk;
  * optional mixed precision (fp16 autocast + GradScaler) for faster T4 training;
  * DataLoader workers, `quit()` replaced by a clean return;
  * `multi_speaker: auto` turns speaker embeddings on when >1 speaker survives
    preprocessing.
"""
import csv
import json
import os
import shutil
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num, latest_step, ckpt_file
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_NAMES = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def resolve_multi_speaker(preprocess_config, model_config):
    if model_config.get("multi_speaker") == "auto":
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            n = len(json.load(f))
        model_config["multi_speaker"] = n > 1
        print("multi_speaker: auto ->", model_config["multi_speaker"], "({} speakers)".format(n))


def _append_csv(path, row, header):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


class _Args:
    def __init__(self, restore_step):
        self.restore_step = restore_step


def train(configs, restore_step=0, max_hours=None, total_step=None):
    preprocess_config, model_config, train_config = configs
    resolve_multi_speaker(preprocess_config, model_config)
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)

    if restore_step == "latest":
        restore_step = latest_step(train_config)
        print("Resuming from step", restore_step)
    restore_step = int(restore_step or 0)
    args = _Args(restore_step)

    print("Prepare training ...")
    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = train_config["optimizer"].get("group_size", 4)  # sort groups of batches by length
    assert batch_size * group_size < len(dataset), "dataset too small for batch_size*group_size"
    num_workers = train_config.get("num_workers", 2)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    model, optimizer = get_model(args, configs, device, train=True)
    # single GPU: ming024 modules use a global `device`, so multi-GPU DataParallel breaks
    model = nn.DataParallel(model, device_ids=[0] if device.type == "cuda" else None)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)
    print("Training utterances:", len(dataset), "| device:", device)

    use_amp = bool(train_config.get("use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    vocoder = get_vocoder(model_config, device)

    log_path = train_config["path"]["log_path"]
    train_log_path = os.path.join(log_path, "train")
    val_log_path = os.path.join(log_path, "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)
    train_csv = os.path.join(log_path, "train_losses.csv")
    val_csv = os.path.join(log_path, "val_losses.csv")
    best_json = os.path.join(train_config["path"]["ckpt_path"], "best.json")
    best = json.load(open(best_json)) if os.path.exists(best_json) else {"step": None, "val_total": float("inf")}

    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = int(total_step or train_config["step"]["total_step"])
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    keep_last = train_config["step"].get("keep_last", 2)
    t0 = time.time()
    deadline = t0 + max_hours * 3600 if max_hours else None

    saved = {"step": None}

    def save(step, reason=""):
        path = ckpt_file(train_config, step)
        if saved["step"] == step and os.path.exists(path):
            return
        saved["step"] = step
        torch.save(
            {"model": model.module.state_dict(), "optimizer": optimizer._optimizer.state_dict(), "step": step},
            path,
        )
        print("Saved checkpoint {} {}".format(path, reason))
        # prune old periodic checkpoints (never the best one)
        ckpts = sorted(int(f.split(".")[0]) for f in os.listdir(train_config["path"]["ckpt_path"])
                       if f.endswith(".pth.tar") and f.split(".")[0].isdigit())
        for s in ckpts[:-keep_last]:
            if s != best.get("step"):
                os.remove(ckpt_file(train_config, s))

    def validate(step):
        model.eval()
        message, loss_means = evaluate(model, step, configs, val_logger, vocoder, return_losses=True)
        model.train()
        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
            f.write(message + "\n")
        _append_csv(val_csv, [step] + ["{:.5f}".format(l) for l in loss_means], ["step"] + LOSS_NAMES)
        outer_bar.write(message)
        return loss_means

    outer_bar = tqdm(total=total_step, desc="Training", position=0, mininterval=30)
    outer_bar.n = args.restore_step
    outer_bar.update()

    model.train()
    while True:
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    output = model(*(batch[2:]))
                    losses = Loss(batch, output)
                total_loss = losses[0] / grad_acc_step
                if not torch.isfinite(total_loss):
                    print("Non-finite loss at step {} - skipping batch".format(step))
                    optimizer.zero_grad()
                    continue

                if use_amp:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                if step % grad_acc_step == 0:
                    if use_amp:
                        scaler.unscale_(optimizer._optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    if use_amp:
                        optimizer._update_learning_rate()
                        scaler.step(optimizer._optimizer)
                        scaler.update()
                    else:
                        optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses_f = [l.item() for l in losses]
                    lr = optimizer._optimizer.param_groups[0]["lr"]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = ("Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, "
                                "Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}").format(*losses_f)
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")
                    outer_bar.write(message1 + message2 + ", lr: {:.2e}, elapsed: {:.2f} h".format(
                        lr, (time.time() - t0) / 3600))
                    log(train_logger, step, losses=losses_f)
                    train_logger.add_scalar("lr", lr, step)
                    _append_csv(train_csv, [step] + ["{:.5f}".format(l) for l in losses_f] + ["{:.3e}".format(lr)],
                                ["step"] + LOSS_NAMES + ["lr"])

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch, output, vocoder, model_config, preprocess_config)
                    log(train_logger, fig=fig, tag="Training/step_{}_{}".format(step, tag))
                    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
                    log(train_logger, audio=wav_reconstruction, sampling_rate=sr,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag))
                    log(train_logger, audio=wav_prediction, sampling_rate=sr,
                        tag="Training/step_{}_{}_synthesized".format(step, tag))

                if step % val_step == 0:
                    loss_means = validate(step)
                    if loss_means[0] < best["val_total"]:
                        best = {"step": step, "val_total": loss_means[0]}
                        save(step, "(new best val loss {:.4f})".format(loss_means[0]))
                        json.dump(best, open(best_json, "w"))

                if step % save_step == 0:
                    save(step)

                out_of_time = deadline is not None and time.time() > deadline
                if step >= total_step or out_of_time:
                    if step % val_step != 0:
                        loss_means = validate(step)
                        if loss_means[0] < best["val_total"]:
                            best = {"step": step, "val_total": loss_means[0]}
                            json.dump(best, open(best_json, "w"))
                    save(step, "(final{})".format(", time budget reached" if out_of_time else ""))
                    with open(os.path.join(train_config["path"]["ckpt_path"], "last.json"), "w") as f:
                        json.dump({"step": step, "hours": (time.time() - t0) / 3600}, f)
                    outer_bar.close()
                    train_logger.flush()
                    val_logger.flush()
                    print("Training finished at step {} ({:.2f} h). Best val step: {}".format(
                        step, (time.time() - t0) / 3600, best["step"]))
                    return step
                step += 1
                outer_bar.update(1)
        epoch += 1
