# Changes relative to ming024/FastSpeech2

Reference: https://github.com/ming024/FastSpeech2 (commit on `master`, 2021). Modified lines in the
copied files are marked with `[MODIFIED]` comments.

## Shared (both languages)

| File | Change | Why |
|---|---|---|
| `audio/stft.py` | removed hard-coded `.cuda()` in the STFT; librosa calls use keyword arguments | crashed on CPU / multiprocessing; librosa ≥ 0.10 API |
| `audio/audio_processing.py` | `pad_center(..., size=)` | librosa ≥ 0.10 |
| `model/modules.py` | energy predictor used `p_control` instead of `e_control` | upstream bug: `--energy_control` had no effect and pitch control also scaled energy |
| `utils/model.py` | `torch.load(..., weights_only=False, map_location=...)`; vocoder paths relative to the package; auto-unzip HiFi-GAN weights; `requires_grad_(False)` (was an attribute assignment); clip before int16 | torch ≥ 2.6; runs from any working directory; no wrap-around clicks |
| `text/symbols.py` | removed pinyin, added the Telugu phone set | Telugu support |
| `preprocessor.py` | read MFA empty intervals as `sp`; split from data-selection step instead of random; normalisation stats from training set only; drop `spn`/implausible/too-long utterances; merge consecutive silences; multiprocessing; configurable F0 range | MFA ≥ 2 writes silences as empty intervals, which `tgt` drops by default → durations no longer summed to the mel length and all targets were shifted; required 5 h / 1 h non-overlapping split; no leakage of validation statistics; speed |
| `train.py` → `trainer.py` | single config; `--max_hours` time budget; resume from latest; CSV loss logs; best-checkpoint tracking; checkpoint pruning; optional AMP; DataLoader workers; single-GPU DataParallel | free GPU sessions have time limits; loss plots; disk space; ming024's global `device` breaks multi-GPU DataParallel |
| `evaluate.py` | returns the loss values | used for best-checkpoint selection and CSV logging |
| `synthesize.py` → `synthesis.py` | rewritten: test-set synthesis, per-phone duration/F0/energy plots, F0 and energy measured on the generated audio, CSV export | assignment deliverables |
| 3 × yaml → `config.yaml` | one config per language with `preprocess`, `model`, `train`, `inference` sections; `multi_speaker: auto` | assignment structure |

## English-specific

* **New text front-end** (`text/english_frontend.py`) replacing `preprocess_english`:
  * normalises times (`10:45 AM` → "ten forty five A M"), dates (`12 November 2026` →
    "twelfth November twenty twenty six"), years, decades, ordinals, decimals, currency (₹/Rs/$),
    percentages, abbreviations, hyphenated compounds; spells out capitalised acronyms letter by letter.
    ming024/g2p_en produced "tenforty-five" and read "AM" as the verb *am*.
  * runs the POS tagger over the **whole sentence** so g2p_en's heteronym table works
    (`conduct` noun = K AA1 N D AH0 K T, verb = K AA0 N D AH1 K T). ming024 called g2p_en
    word by word, so every heteronym got the same reading.
* **Pronunciation dictionary for MFA** built from the same front-end: every CMUdict variant of every
  corpus word (+ g2p_en neural predictions for OOV words, + letter names), so MFA picks the
  variant actually spoken.
* **Alignment** with the pretrained MFA `english_us_arpa` model (ARPAbet with stress, same phone set
  as g2p_en).
* **Speakers**: Svarah has no speaker-id column; the tuple of speaker metadata (gender, age group,
  L1, state, district, education, job, domain) is hashed into a pseudo speaker id; speaker embedding
  enabled; inference uses the speaker with the most training data.
* **Data selection**: 1–15 s clips, no annotation tags / non-ASCII transcripts, noisiest 10 % (SNR
  proxy) removed, speaker-balanced round-robin selection, sentence-disjoint splits.

## Telugu-specific (Telugu support)

* **Telugu grapheme-to-phoneme converter** (`text/telugu_frontend.py`), since ming024 has no Telugu
  support and no Telugu G2P/lexicon exists in the codebase:
  * 36 consonants, 12 vowels (+ vocalic r/l), vowel signs, virama, conjuncts, nukta (ఫ఼ → f, జ఼ → z);
  * inherent /a/ after a consonant unless followed by a vowel sign or virama;
  * anusvara ం assimilated to the place of the next consonant (ng/ny/nn/n/m) and /m/ word-finally;
  * visarga ః → /h/, arasunna ఁ silent, ZWJ/ZWNJ ignored, ై written as ె+ౖ handled.
  * phones are lowercase ASCII so they never collide with ARPAbet in the shared symbol table.
* **Telugu text normalisation**: Telugu digits → ASCII; cardinals in the Indian system
  (2026 → రెండు వేల ఇరవై ఆరు), ordinals with -వ (12వ → పన్నెండవ), clock times
  (10:45 → పది నలభై ఐదు), decimals, percentages.
* **Alignment**: no pretrained Telugu MFA model exists, so an MFA GMM-HMM acoustic model is trained
  on the selected Telugu data (`mfa train`, single-speaker mode) with the rule-based lexicon.
* **Data selection**: single female speaker when she has ≥ 6 h of 1–15 s clips (automatic fallback to
  both speakers with speaker embedding otherwise); transcripts with Latin letters or unexpected
  characters removed.
* **Plots** label Telugu words in romanised form because matplotlib cannot shape Telugu conjuncts;
  the CSV/JSON outputs keep the Telugu script.
