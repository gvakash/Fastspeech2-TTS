# Step-by-step: from zero to two trained models (free GPU, no local GPU needed)

Total hands-on time ≈ 45 min. Then the GPU works for ~9 h per language in the background.

## 0. What is going on (plain English)

* **FastSpeech 2** turns a sequence of phones (speech sounds) into a mel-spectrogram (a picture of the
  sound: time × frequency). A separate, already-trained **vocoder** (HiFi-GAN) turns that picture
  into a waveform. We only train FastSpeech 2; the vocoder is downloaded ready-made.
* Inside FastSpeech 2 the **variance adaptor** predicts, for every phone,
  * **duration** – how many 11.6 ms frames the phone lasts,
  * **pitch (F0)** – how high the voice is (Hz),
  * **energy** – how loud it is.
  During training it learns these from the real recordings; at test time it predicts them. The
  assignment's plots show exactly these predictions.
* To learn durations, the model needs to know where each phone starts and ends in every training
  recording. The **Montreal Forced Aligner (MFA)** finds this automatically ("forced alignment").
* **G2P** (grapheme-to-phoneme) converts spelling into phones: "conduct" → K AA1 N D AH0 K T,
  "పట్టు" → p a tt tt u.
* **Loss** = how wrong the model still is. It should go down. *Training loss* is measured on data the
  model learns from, *validation loss* on the 1 h it never learns from. If training loss keeps going
  down while validation loss goes up, the model is memorising (overfitting).

## 1. Put the code on GitHub (10 min)
1. Create a **public** repository on github.com, e.g. `fastspeech2-english-telugu` (no README).
2. In this folder:
   ```bash
   git init && git add . && git commit -m "FastSpeech2 English + Telugu pipeline"
   git branch -M main
   git remote add origin https://github.com/<you>/fastspeech2-english-telugu.git
   git push -u origin main
   ```
   (Or ask Claude Code in VS Code: "push this folder to a new public GitHub repo".)

## 2. Hugging Face access (5 min)
1. Sign up at huggingface.co.
2. Open https://huggingface.co/datasets/ai4bharat/Svarah and click **Agree and access repository**.
3. Settings → Access Tokens → **New token** (type *Read*). Copy it (`hf_...`).

## 3. Kaggle (15 min)
1. Sign up at kaggle.com and **verify your phone number** (Settings → Phone verification).
   Without this you get no GPU and no internet.
2. *Create → New Notebook*. Menu *File → Import Notebook* → upload `notebooks/kaggle_fastspeech2.ipynb`.
3. Right-hand panel → *Session options*: **Accelerator = GPU P100** (or GPU T4 x2),
   **Internet = On**.
4. *Add-ons → Secrets → Add a new secret*: label `HF_TOKEN`, value = your token. Tick the checkbox so
   this notebook can use it.
5. In the first code cell set
   ```python
   LANG = "English"
   REPO_URL = "https://github.com/<you>/fastspeech2-english-telugu.git"
   ```
6. Top right: **Save Version → Save & Run All (Commit) → Save**. It now runs in the background;
   you can close the browser.
7. Repeat 2–6 with a second notebook and `LANG = "Telugu"`. (Kaggle allows a few sessions at once; if it
   refuses, start Telugu when English finishes.)

**Check after ~1 hour**: open the notebook → *View versions* → the running version → *Logs*. You should
see `smoke test passed`. If something failed, the log ends with `command failed` – copy the last
~40 lines and ask Claude.

## 4. Collect the results (~10 h later)
1. Open the finished version → **Output** tab → download `English_outputs.zip` / `Telugu_outputs.zip`.
2. Unzip both *into the repository folder* (they contain `Results/<LANG>/…`, `<LANG>/data_meta/`,
   `<LANG>/logs/`, `<LANG>/checkpoints/*_fp16.pth.tar`).
3. `git add . && git commit -m "results" && git push`.

## 5. Report
Share `Results/*/loss_summary.json`, the plots and `Results/*/frontend_log.json` with Claude to write the
analysis sections; listen to every wav yourself and note pronunciation errors – that part needs ears.

## If something goes wrong
| Symptom | Fix |
|---|---|
| `401` / `GatedRepoError` for Svarah | accept the terms on the dataset page; check the `HF_TOKEN` secret is attached |
| `No GPU` in the nvidia-smi line | Session options → Accelerator, then commit again |
| `CUDA out of memory` | in `config.yaml` set `batch_size: 8` |
| Worried about the 12 h limit | the notebook caps training so the whole run ends by ~11.5 h and still writes all results |
| MFA install fails | re-run; it downloads ~300 MB from conda-forge |
