# FastSpeech 2 – English and Telugu TTS

This repo has two text-to-speech models, one for English and one for Telugu, trained separately with
FastSpeech 2. The code is based on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2),
with changes to make it work for Indian English and Telugu.

Paper: [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558) (Ren et al., 2020)

## Overview

| | English | Telugu |
|---|---|---|
| Dataset | [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (Indian-accented English) | [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) |
| Data used | ~5 h train, ~1 h validation | ~5 h train, ~1 h validation |
| Speakers | 114 | 2 (one female, one male) |
| Training | 7 h on a Kaggle T4 | 7 h on a Kaggle T4 |
| Checkpoint | step 31,000 | step 11,000 |

Svarah doesn't come with speaker IDs, so the 114 English speakers are my estimate, grouped from the
speaker details (gender, age, region and so on).

I don't have a GPU, so everything was trained within Kaggle's free 30-hour GPU quota. That gave me
one 7-hour run per language, with no time left for tuning.

## Audio samples

The generated audio for the test sentences is in `Results/`:

- English: [E1](Results/English/test_generated_audios/E1.wav), [E2a](Results/English/test_generated_audios/E2a.wav), [E2b](Results/English/test_generated_audios/E2b.wav), [E3](Results/English/test_generated_audios/E3.wav), [E4](Results/English/test_generated_audios/E4.wav)
- Telugu: [T1](Results/Telugu/test_generated_audios/T1.wav), [T2a](Results/Telugu/test_generated_audios/T2a.wav), [T2b](Results/Telugu/test_generated_audios/T2b.wav), [T3](Results/Telugu/test_generated_audios/T3.wav), [T4](Results/Telugu/test_generated_audios/T4.wav)

The sentences themselves are in `English/test_sentences.txt` and `Telugu/test_sentences.txt`. They are
kept exactly as given, including the split words "rhyth m" and "అడ విలో".

## Quickstart

### Dependencies

```bash
pip install -r requirements.txt
python -c "import nltk; [nltk.download(r) for r in ['cmudict', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'punkt']]"
```

### Inference

The trained checkpoints are already in the repo (about 68 MB each), so you can generate speech right away:

```bash
python English/inference.py --checkpoint English/checkpoints/fastspeech2_english_31000_fp16.pth.tar
python Telugu/inference.py  --checkpoint Telugu/checkpoints/fastspeech2_telugu_11000_fp16.pth.tar
```

This reads every sentence in the language's `test_sentences.txt`. For each one, it saves the audio and
plots of the duration, pitch and energy to `Results/<language>/`. Running it again replaces the files
that are already there; add `--output_dir some_folder` to save them somewhere else.

To say your own sentence:

```bash
python Telugu/inference.py --checkpoint Telugu/checkpoints/fastspeech2_telugu_11000_fp16.pth.tar \
       --text "నమస్కారం, మీరు ఎలా ఉన్నారు?" --id hello
```

### Controlling speed, pitch and energy

```bash
python English/inference.py --checkpoint English/checkpoints/fastspeech2_english_31000_fp16.pth.tar \
       --text "This is a little slower." --id demo --duration_control 1.2 --pitch_control 1.1
```

Duration values above 1 make the speech slower. Pitch and energy values above 1 make the rises and
falls in pitch and loudness bigger, and values below 1 make them flatter.

## Training

The steps are the same for both languages. The commands below are for English; replace `English` with
`Telugu` for the other model.

### What you need

- A GPU (I used a Kaggle T4).
- [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/), which only installs through conda.
- A Hugging Face token. Svarah is a gated dataset, so accept its terms on Hugging Face first.

```bash
conda create -n mfa -c conda-forge montreal-forced-aligner python=3.11
export MFA_BIN=$(conda run -n mfa which mfa)
export HF_TOKEN=hf_...
```

### Preprocessing

```bash
python English/preprocess.py --stage all
```

This runs four stages, which you can also run one at a time with `--stage`:

1. `prepare`: downloads the dataset, removes clips that are too short, too long or too noisy, and picks
   ~5 h for training and ~1 h for validation. The two sets never share a sentence.
2. `lexicon`: builds the word-to-phone dictionary that MFA needs.
3. `align`: runs MFA to find where each phone starts and ends. English uses MFA's pretrained English
   model. MFA has no Telugu model of its own, so for Telugu an aligner is trained on the same 5 h first.
4. `features`: extracts mel spectrograms, pitch, energy and phone durations.

A stage is skipped if its output already exists. Add `--overwrite` to run it again. The audio and
features take about 2 GB per language and go to `English/data/` (or to `FS2_DATA_ROOT` if you set it).

### Training

```bash
python English/train.py --max_hours 7
python tools/plot_losses.py --lang English
```

`--max_hours` stops the run cleanly before the time limit and saves a checkpoint. If the run stops
early, start it again with `--restore_step latest`. The best checkpoint (lowest validation loss) is
tracked automatically. Then generate the test sentences and export a smaller checkpoint:

```bash
python English/inference.py --checkpoint best
python tools/export_checkpoint.py --lang English
```

### On Kaggle (what I used)

1. Import `extras/notebooks/kaggle_fastspeech2.ipynb` into Kaggle.
2. Turn on the T4 GPU and Internet, and add `HF_TOKEN` under Add-ons → Secrets.
3. Set `LANG` and `REPO_URL` in the first cell, then use *Save & Run All*.

The notebook runs everything above and gives you a zip of the results. See
`extras/HOW_TO_RUN_ON_KAGGLE.md` for the full walkthrough.

## What I changed from ming024/FastSpeech2

The model itself (encoder, variance adaptor, decoder, HiFi-GAN vocoder) is unchanged, apart from the
`--energy_control` fix below. Most of the work was around it:

- **English text handling.** Times, dates and numbers are written out in words ("10:45 AM" becomes
  "ten forty five A M"). Words like "conduct" are pronounced according to whether they're a noun or a verb.
- **Telugu text handling.** A rule-based converter turns Telugu script into phones, and Telugu
  numbers, dates and times are written out in words.
- **Telugu aligner.** MFA's own model list has no Telugu acoustic model, so I trained one on the
  Telugu data. AI4Bharat's IndicMFA does offer one, but I didn't use it.
- **Bug fixes.** Pauses in newer MFA output were being dropped, which shifted the timing of everything
  after them. Also, `--energy_control` didn't work, because the code used the pitch value for energy.
- **Training.** A time budget for Kaggle, resuming, CSV loss logs and best-checkpoint tracking.
- **Inference.** It now produces plots of duration, pitch and energy for each sentence.

Every change is listed in `extras/CHANGES.md`.

## Results and known issues

- Both models start overfitting after about 10k steps. After that the validation loss barely changes,
  so most of the 7 hours didn't help.
- The speech is understandable but sounds robotic. Two people listened to the Telugu audio without
  being told what it was. Both found the pacing unnatural (one called it very fast), and one said
  the pronunciation was correct.
- The models only pause at punctuation inside a sentence. In the test sentences, that means just the
  commas in E4 and T4.
- "rhyth m" in E3 comes out as "raitham", and the Telugu word "andaroo" in E4 gets English vowels.
- The Telugu aligner was trained on only 5 h of audio, and some phone timings are off. AI4Bharat's
  IndicMFA would probably do better, but I didn't get to try it.

The loss plots are in `Results/<language>/`, and the full analysis is in [Report/Technical_Report.pdf](Report/Technical_Report.pdf).

## Project structure

```
English/            preprocess.py, train.py, inference.py, config.yaml, test sentences,
                    checkpoint, logs and data-split files
Telugu/             same as English
fastspeech2_core/   shared code: model, text front-ends, MFA wrapper, training, synthesis
tools/              loss plots, checkpoint export
Results/            generated audio, loss plots, duration/pitch/energy plots
Report/             technical report (PDF)
extras/             Kaggle notebook, Kaggle guide, list of changes
requirements.txt
```

## References

- [FastSpeech 2 paper](https://arxiv.org/abs/2006.04558)
- [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (MIT): model code and HiFi-GAN vocoder
- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
- [AI4Bharat Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (CC-BY-4.0)
- [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) (IIT Madras, via SPRINGLab)
- [g2p_en](https://github.com/Kyubyong/g2p) / CMUdict for English pronunciation

## Author

G.V. Akash
