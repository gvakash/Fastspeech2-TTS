# FastSpeech 2 for English (Svarah) and Telugu (IndicTTS)

Two separately trained FastSpeech 2 text-to-speech models for English and Telugu, built on
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (paper: Ren et al., 2020,
[arXiv:2006.04558](https://arxiv.org/abs/2006.04558)).

| | English | Telugu |
|---|---|---|
| Dataset | [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (Indian-accented English, 117 speakers) | [SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) |
| Train / val | ~5 h / ~1 h, no overlap | ~5 h / ~1 h, no overlap |
| Input symbols | ARPAbet phones (stress-marked) | rule-based Telugu phones (49) |
| Front-end | text normaliser + sentence-level POS-aware G2P (g2p_en) | Telugu G2P (inherent vowel, virama, anusvara assimilation) + Telugu number expansion |
| Alignment | MFA, pretrained `english_us_arpa` | MFA acoustic model trained on the Telugu corpus |
| Speakers | multi-speaker (speaker embedding) | single speaker (female) |
| Vocoder | HiFi-GAN universal (pretrained, from ming024) | Vocoder | HiFi-GAN universal (pretrained, from ming024) |

## Repository layout

```
project/
├── English/   train.py  preprocess.py  inference.py  config.yaml  test_sentences.txt  checkpoints/  data_meta/
├── Telugu/    train.py  preprocess.py  inference.py  config.yaml  test_sentences.txt  checkpoints/  data_meta/
├── fastspeech2_core/      shared code adapted from ming024 (model, transformer, audio, hifigan, text, ...)
│   ├── text/english_frontend.py   NEW  English normaliser + G2P
│   ├── text/telugu_frontend.py    NEW  Telugu G2P + number normalisation
│   ├── data_prep.py               NEW  download, filtering, 5 h / 1 h selection
│   ├── align.py                   NEW  Montreal Forced Aligner wrapper
│   ├── preprocessor.py            MODIFIED from ming024
│   ├── trainer.py                 MODIFIED from ming024/train.py
│   └── synthesis.py               NEW  inference + variance-adaptor plots
├── tools/   plot_losses.py  export_checkpoint.py  collect_meta.py
├── notebooks/kaggle_fastspeech2.ipynb    one-click pipeline for a free Kaggle GPU
├── Results/{English,Telugu}/  training_loss.png  validation_loss.png  test_generated_audios/  variance_adaptor_plots/{duration,pitch,energy}/
├── Report/Technical_Report.pdf
├── CHANGES.md        every modification made to the reference implementation
├── README.md
└── requirements.txt
```

`data_meta/` holds the small files that document the data split (`data_selection_report.json`,
`train.txt`, `val.txt`, `metadata.json`) and that inference needs (`stats.json`, `speakers.json`).
The audio and the extracted features (~2 GB per language) are not in the repository; they are
re-created by `preprocess.py`.

## Checkpoints

`English/checkpoints/` and `Telugu/checkpoints/` contain the best checkpoint (lowest validation
loss) as **fp16 model weights** (~68 MB each, `fastspeech2_<lang>_<step>_fp16.pth.tar`), small
enough for GitHub without LFS. Full fp32 checkpoints with optimiser state (~420 MB, needed only
to resume training): see the link in the release notes / below.

<!-- Full checkpoints: https://... (add Kaggle/Drive link here) -->

## Reproducing

### Option A – free Kaggle GPU (what was used)
1. Upload `notebooks/kaggle_fastspeech2.ipynb` to Kaggle (*Create → New Notebook → File → Import*).
2. Session options: **GPU P100** (or T4), **Internet on**. Add-ons → Secrets: `HF_TOKEN`
   (accept the Svarah terms on Hugging Face first).
3. Set `LANG = "English"` or `"Telugu"` and `REPO_URL` in the first cell →
   *Save Version → Save & Run All*. Download `<LANG>_outputs.zip` from the Output tab.

### Option B – any Linux machine with a GPU
```bash
pip install -r requirements.txt
# Montreal Forced Aligner (conda only):
conda create -n mfa -c conda-forge montreal-forced-aligner python=3.11
export MFA_BIN=$(conda run -n mfa which mfa)
export HF_TOKEN=hf_...                 # Svarah is gated
export FS2_DATA_ROOT=/path/with/space  # optional, default <lang>/data

python English/preprocess.py --config English/config.yaml --stage all   # prepare, lexicon, align, features
python English/train.py      --config English/config.yaml --max_hours 7
python tools/plot_losses.py  --lang English
python English/inference.py  --config English/config.yaml --checkpoint best
python tools/export_checkpoint.py --lang English
```
Replace `English` by `Telugu` for the Telugu model. Preprocessing stages can be run one at a time
(`--stage prepare|lexicon|align|features`) and are skipped when their output already exists
(`--overwrite` to redo).

### Inference only (from this repository)
```bash
pip install -r requirements.txt
python English/inference.py --checkpoint English/checkpoints/fastspeech2_english_<step>_fp16.pth.tar
python Telugu/inference.py  --checkpoint Telugu/checkpoints/fastspeech2_telugu_<step>_fp16.pth.tar \
       --text "నమస్కారం, మీరు ఎలా ఉన్నారు?" --id hello
# speed / pitch / energy control
python English/inference.py --duration_control 1.2 --pitch_control 1.1 --text "Slower and higher." --id demo
```

## Test sentences

Kept exactly as given (including the split words "rhyth m" and "అడ విలో"), see
`English/test_sentences.txt` and `Telugu/test_sentences.txt`. For each sentence `inference.py` writes
the waveform, a duration bar chart, pitch and energy plots (prediction vs. values measured on the
generated audio), an overview figure, a CSV with the per-phone numbers and `frontend_log.json`
(text → normalised text → phones).

## Acknowledgements
Model, transformer, variance adaptor, HiFi-GAN code and vocoder weights: ming024/FastSpeech2 (MIT).
Datasets: AI4Bharat Svarah (CC-BY-4.0), IndicTTS Telugu (IIT Madras, via SPRINGLab).
Aligner: Montreal Forced Aligner. English G2P: g2p_en / CMUdict.
