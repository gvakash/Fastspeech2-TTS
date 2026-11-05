import os
import json
import zipfile

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim

# [MODIFIED] Paths are resolved relative to this package instead of the current
# working directory, so the scripts can be launched from English/ or Telugu/.
CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIFIGAN_DIR = os.path.join(CORE_DIR, "hifigan")


def _torch_load(path, device):
    # [MODIFIED] torch>=2.6 defaults to weights_only=True, which refuses the
    # pickled optimiser state saved by this codebase.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # very old torch without the weights_only argument
        return torch.load(path, map_location=device)


def ckpt_file(train_config, step):
    return os.path.join(train_config["path"]["ckpt_path"], "{}.pth.tar".format(step))


def latest_step(train_config):
    """Return the largest checkpoint step found in ckpt_path (0 if none)."""
    ckpt_dir = train_config["path"]["ckpt_path"]
    if not os.path.isdir(ckpt_dir):
        return 0
    steps = []
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pth.tar") and f.split(".")[0].isdigit():
            steps.append(int(f.split(".")[0]))
    return max(steps) if steps else 0


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    ckpt = None
    if args.restore_step:
        ckpt = _torch_load(ckpt_file(train_config, args.restore_step), device)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if ckpt is not None and "optimizer" in ckpt:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_(False)  # [MODIFIED] upstream assigned to the attribute by mistake
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def _ensure_unzipped(name):
    pth = os.path.join(HIFIGAN_DIR, name)
    if not os.path.exists(pth):
        with zipfile.ZipFile(pth + ".zip") as z:
            z.extractall(HIFIGAN_DIR)
    return pth


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open(os.path.join(HIFIGAN_DIR, "config.json"), "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = _torch_load(_ensure_unzipped("generator_LJSpeech.pth.tar"), device)
        elif speaker == "universal":
            ckpt = _torch_load(_ensure_unzipped("generator_universal.pth.tar"), device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    # [MODIFIED] clip before int16 conversion to avoid wrap-around clicks
    wavs = (
        wavs.clamp(-1.0, 1.0).cpu().numpy()
        * (preprocess_config["preprocessing"]["audio"]["max_wav_value"] - 1)
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
