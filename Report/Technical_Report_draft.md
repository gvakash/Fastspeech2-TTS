# FastSpeech 2–Based Text-to-Speech for English and Telugu — Technical Report (DRAFT)

> Sections marked **[RESULTS]** are filled in after training (numbers come from
> `Results/<lang>/loss_summary.json`, `data_meta/*.json` and `frontend_log.json`).

## 1. Implementation approach

### 1.1 Architecture (from the reference implementation)
FastSpeech 2 (Ren et al., 2020) is a non-autoregressive acoustic model. All components below are
used unchanged from ming024/FastSpeech2:

* **Phone embedding + encoder** – 4 feed-forward Transformer blocks (hidden 256, 2 heads,
  1D-conv FFN 1024, kernel 9).
* **Variance adaptor** – three predictors (2 × Conv1D-ReLU-LayerNorm-Dropout + linear):
  *duration* (log-domain, MSE), *pitch* and *energy* (phone-level, z-normalised, MSE). Pitch and energy
  are quantised into 256 bins and added back to the hidden sequence as embeddings; the length
  regulator expands every phone by its duration. During training ground-truth values are used
  (teacher forcing); at inference the predictions are used.
* **Mel decoder** – 6 FFT blocks + linear projection to 80 mel bins, followed by a 5-layer Conv1D
  **PostNet** (L1 loss before and after the PostNet).
* **Speaker embedding** (multi-speaker English model) added to the encoder output.
* **Vocoder** – pretrained HiFi-GAN *universal* generator (22.05 kHz, hop 256, 80 mels, fmax 8 kHz).
  Not fine-tuned.

Total loss = L1(mel) + L1(mel_postnet) + MSE(log-duration) + MSE(pitch) + MSE(energy).
~35.4 M parameters per model.

### 1.2 Modifications (summary – full list in CHANGES.md)
* Compatibility fixes: CPU-safe STFT, librosa ≥ 0.10 API, torch ≥ 2.6 checkpoint loading.
* Bug fixes: energy control used the pitch control value; MFA ≥ 2 empty silence intervals were
  dropped by `tgt`, which shifted every duration/pitch/energy target relative to the mel frames.
* New English front-end: normalisation of times/dates/numbers/acronyms and **sentence-level** POS-aware
  G2P so heteronyms are resolved (ming024 ran G2P word by word).
* **Telugu support**: rule-based Telugu G2P (inherent vowel, virama/conjuncts, anusvara place
  assimilation, visarga, nukta), Telugu number normalisation (Indian numbering, -వ ordinals, times),
  Telugu phone set added to the symbol table, MFA acoustic model trained on the Telugu corpus.
* Data pipeline: Hugging Face download, filtering, ~5 h / ~1 h selection with no overlap.
* Training: single config per language, time-budgeted training for free GPUs, CSV loss logs,
  best-checkpoint selection on validation loss.
* Inference: per-phone duration / F0 / energy plots, with F0 and energy re-measured on the
  synthesised audio for comparison.

## 2. Data

### 2.1 Selection and sub-sampling
| | English – Svarah | Telugu – IndicTTS |
|---|---|---|
| Source | 9.6 h, 117 speakers, read + spontaneous Indian-accented English | IndicTTS Telugu (female + male studio voices) |
| Clip length filter | 1–15 s | 1–15 s |
| Text filter | no annotation tags, ASCII only, ≥ 2 words | no Latin letters / unexpected characters |
| Noise filter | lowest 10 % SNR-proxy removed | none (studio) |
| Speakers | all; pseudo speaker id = hash of the speaker-metadata tuple | female voice only (single speaker) |
| Selection | speaker-balanced round-robin over shuffled sentence groups (seed 1234): validation first (~1 h), then training (~5 h) | same procedure |
| Train | **[RESULTS]** utterances / h | **[RESULTS]** |
| Validation | **[RESULTS]** utterances / h | **[RESULTS]** |

*No overlap*: clips are grouped by normalised transcript before splitting, so no recording and no
sentence appears in both sets (asserted in code; ids and texts are in `data_meta/train.txt`,
`val.txt`).

### 2.2 Preprocessing
1. Audio decoded from the parquet shards, converted to mono, resampled to 22.05 kHz, peak-normalised
   to 0.95, 16-bit PCM.
2. Transcripts normalised by the language front-end; the aligner sees the normalised words.
3. Pronunciation lexicon: English – all CMUdict variants + g2p_en predictions for OOVs;
   Telugu – rule-based G2P (one pronunciation per word, no OOVs possible).
4. Forced alignment with MFA 3.4: English with the pretrained `english_us_arpa` model; Telugu with
   an acoustic model trained on the selected data (monophone → triphone → LDA → SAT).
5. Features (ming024 recipe): 80-bin log-mel (1024 FFT, hop 256), F0 with WORLD DIO + StoneMask
   (65–600 Hz), frame energy = L2 norm of the STFT frame; F0 linearly interpolated over unvoiced
   frames and averaged per phone; energy averaged per phone; leading/trailing silence trimmed using
   the alignment; z-normalisation with statistics from the training set only.
6. Utterances with unknown words, phones longer than 1.16 s (alignment failure) or no voiced frames
   are removed (**[RESULTS]** counts from `preprocess_summary.json`).

### 2.3 Training configuration
| | both models |
|---|---|
| Optimiser | Adam (β = 0.9, 0.98, ε = 1e-9), Noam schedule, 4000 warm-up steps, ×0.3 at 50k/70k/90k |
| Batch | 16 utterances, length-sorted groups of 4 batches |
| Gradient clipping | 1.0 |
| Hardware | Kaggle **[RESULTS: P100/T4]**, single GPU, fp32 |
| Budget | **[RESULTS]** h, **[RESULTS]** steps; validation every 1000 steps on the full validation set |
| Checkpoint used for synthesis | lowest total validation loss (**[RESULTS]** step) |

## 3. Loss curves **[RESULTS]**
Figures: `Results/<lang>/training_loss.png`, `validation_loss.png`, `train_vs_val_loss.png`.
Discuss: speed of convergence per component (duration and pitch converge first, mel losses
slowly), the train–validation gap, the step where validation mel loss stops improving
(overfitting on 5 h), noisiness of English batches (many speakers/channels) vs. Telugu.

## 4. Generated speech **[RESULTS]**
Per sentence: intelligibility, pronunciation errors, prosody, artefacts.
Points to check (prepared from the front-end output, `frontend_log.json`):

* **E1** – *antidisestablishmentarianism* is in CMUdict (28 phones); check whether the long word is
  rushed (duration predictor) and whether stress (EH1) gets a pitch accent.
* **E2a/E2b** – heteronym: noun *conduct* → K AA1 N D AH0 K T (stress on 1st syllable), verb →
  K AA0 N D AH1 K T (2nd). Compare the predicted durations/F0 of the two vowels.
* **E3** – "rhyth m": the split word is pronounced as two words: *rhyth* (OOV, neural G2P →
  R AY1 DH) + letter *m* (EH1 M); not "rhythm" (R IH1 DH AH0 M). Text preserved as given.
* **E4** – normalised to "ten forty five A M on twelfth November twenty twenty six, so andaroo…";
  the comma yields a pause (sp); *andaroo* (Telugu అందరూ "everyone") is OOV → neural G2P
  AE0 N D AA1 R OW0 (final vowel /oʊ/ instead of /uː/).
* **T1** – long transliterated English technical word (21 phones); ఫ is read as aspirated /pʰ/ by the
  rule, whereas speakers say /f/ in this loanword.
* **T2a/T2b** – పట్టు (grip / silk) has identical spelling and pronunciation (p a tt tt u); any
  difference comes only from context (sentence position, neighbours). Geminate tt duration.
* **T3** – "అడ విలో": because Telugu G2P is syllable-based with an inherent vowel, అడ + విలో gives
  exactly the same phones as అడవిలో (a dd a v i l oo) and the model has no word-boundary symbol, so
  the split is neutralised – unlike English E3.
* **T4** – digits expanded: 2026 → రెండు వేల ఇరవై ఆరు, 12వ → పన్నెండవ, 10:45 → పది నలభై ఐదు;
  comma → pause.

## 5. Variance-adaptor analysis **[RESULTS]**
Figures in `Results/<lang>/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png`
(numbers in `<ID>_values.csv`).

## 6. English vs Telugu – comparison, limitations, improvements **[RESULTS]**
Prepared points: multi-speaker noisy data vs single-speaker studio data; pretrained vs self-trained
aligner; dictionary + neural G2P vs deterministic G2P; 5 h is ~20 % of LJSpeech; vocoder not
fine-tuned; no word-boundary/stress information for Telugu.
Improvements: more data or fine-tuning from an LJSpeech checkpoint (English); vocoder fine-tuning on
ground-truth-aligned mels; word-boundary tokens; loanword lexicon for Telugu; frame-level pitch;
speaker/channel-aware filtering for Svarah; longer training with early stopping on validation mel loss.

## 7. Reproducibility
Software: Python 3.11, PyTorch **[RESULTS]**, librosa ≥ 0.10, pyworld, g2p_en 2.1.0, MFA 3.4.3
(conda-forge); full list in `requirements.txt`. Configs: `English/config.yaml`, `Telugu/config.yaml`.
Commands: see README (preprocess → train → plot_losses → inference → export_checkpoint) or run
`notebooks/kaggle_fastspeech2.ipynb`. Seeds: data selection seed 1234.
