"""Single-file configuration loader (NEW).

ming024 uses three YAML files (preprocess.yaml, model.yaml, train.yaml). The
assignment asks for one config.yaml per language, so each language folder has a
single file with the sections `preprocess`, `model`, `train` and `inference`.
This module splits it back into the three dicts the original code expects and
resolves relative paths against the folder that contains the config file.

Placeholders:
  ${LANG_DIR}   folder containing config.yaml (e.g. project/English)
  ${ROOT_DIR}   repository root
  ${DATA_ROOT}  where audio/features go; defaults to ${LANG_DIR}/data but can be
                moved with the FS2_DATA_ROOT environment variable (on Kaggle we
                point it at /tmp so 2 GB of features are not saved as output).
"""
import os
import sys

import yaml

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CORE_DIR)


def setup_path():
    """Make ming024-style top-level imports (`from model import ...`) work."""
    if CORE_DIR not in sys.path:
        sys.path.insert(0, CORE_DIR)


def _resolve(obj, mapping):
    if isinstance(obj, dict):
        return {k: _resolve(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(v, mapping) for v in obj]
    if isinstance(obj, str):
        for k, v in mapping.items():
            obj = obj.replace("${" + k + "}", v)
        return obj
    return obj


def load_config(config_path):
    config_path = os.path.abspath(config_path)
    lang_dir = os.path.dirname(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lang_name = os.path.basename(lang_dir)
    data_root = os.environ.get("FS2_DATA_ROOT")
    data_root = os.path.join(data_root, lang_name) if data_root else os.path.join(lang_dir, "data")
    mapping = {"LANG_DIR": lang_dir, "ROOT_DIR": ROOT_DIR, "DATA_ROOT": data_root}
    cfg = _resolve(cfg, mapping)

    # every remaining relative path is relative to the language folder
    def absolutise(section):
        for k, v in section.items():
            if isinstance(v, str) and v and not os.path.isabs(v) and (v.startswith(".") or "/" in v):
                section[k] = os.path.normpath(os.path.join(lang_dir, v))

    absolutise(cfg["preprocess"]["path"])
    absolutise(cfg["train"]["path"])
    if "inference" in cfg:
        absolutise(cfg["inference"])
    cfg["lang_dir"] = lang_dir
    cfg["data_root"] = data_root
    return cfg


def split_configs(cfg):
    """Return (preprocess_config, model_config, train_config) like ming024."""
    return cfg["preprocess"], cfg["model"], cfg["train"]
