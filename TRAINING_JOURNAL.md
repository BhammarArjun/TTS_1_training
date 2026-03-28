# XTTS v2 Fine-Tuning for Gujarati & Hindi — Complete Training Journal

**Project:** Fine-tuning XTTS v2 for Gujarati (gu) and Hindi (hi) TTS
**Machine:** Lightning.ai L4 GPU Studio (24 GB VRAM, 31 GB RAM, 8 CPUs)
**Date:** March 19, 2026
**Repo:** `https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages`
**Dataset:** `Arjun4707/gu-hi-tts` (private HuggingFace repo, ~65,700 rows)

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Download & Preparation](#2-data-download--preparation)
3. [Pretrained Model & Vocabulary Extension](#3-pretrained-model--vocabulary-extension)
4. [Training — Issues & Solutions](#4-training--issues--solutions)
5. [Final Working Training Command](#5-final-working-training-command)
6. [Key Lessons for Future Projects](#6-key-lessons-for-future-projects)
7. [Dataset Structure Reference](#7-dataset-structure-reference)
8. [Hyperparameter Reference](#8-hyperparameter-reference)

---

## 1. Environment Setup

### 1.1 Python Version

XTTS requires Python 3.10. Lightning.ai comes with 3.11. Modified the existing Conda env directly (did NOT create a new one):

```bash
conda install python=3.10 -y
python --version  # Confirm 3.10.x
```

### 1.2 Clone Repo & Install Dependencies

```bash
cd ~
git clone https://github.com/nguyenhoanganh2002/XTTSv2-Finetuning-for-New-Languages.git
cd XTTSv2-Finetuning-for-New-Languages
pip install -r requirements.txt
pip install datasets soundfile huggingface_hub matplotlib pandas psutil
```

### 1.3 HuggingFace Login (for private dataset)

```bash
huggingface-cli login
# Paste HF token when prompted
```

---

## 2. Data Download & Preparation

### 2.1 Dataset Schema (HuggingFace Parquet)

| Column         | Type                                          | Description                    |
|----------------|-----------------------------------------------|--------------------------------|
| `id`           | string                                        | Unique clip ID                 |
| `audio`        | struct{bytes: binary, sampling_rate: int32}   | Inline WAV bytes @ 24kHz      |
| `text`         | string                                        | Transcript                     |
| `language`     | string                                        | `gu` / `hi` / `en`            |
| `duration_sec` | float32                                       | Clip duration in seconds       |
| `video_id`     | string                                        | Source YouTube video ID        |

Audio: 24kHz, mono, PCM-16, normalized to −3 dBFS, silence-trimmed.

### 2.2 Data Preparation Script

Created `download_and_prepare_data.py` which:

1. Downloads from HuggingFace with **local disk caching** (`hf_cache/` directory — avoids re-downloading)
2. **Cleans text**: removes `>` and `|` characters (pipe breaks the CSV delimiter)
3. **Runs analysis**: duration/character length per language, generates plots
4. **Saves recommended `max_text_length` and `max_audio_length`** to `analysis/data_analysis.json`
5. Splits per language (95% train / 5% eval)
6. Saves to XTTS-expected format:
   - `datasets-gu/wavs/*.wav` + `metadata_train.csv` + `metadata_eval.csv`
   - `datasets-hi/wavs/*.wav` + `metadata_train.csv` + `metadata_eval.csv`
7. Uses **checkpoint system** (`data_prep_checkpoint.json`) for resume support

### 2.3 CSV Format (XTTS Expected)

```
audio_file|text|speaker_name
wavs/xxx.wav|How do you do?|speaker_id
```

**Header line added manually:** `audio_file|text|speaker_name` as the first line.

### 2.4 Data Sizes

| Language | Train Rows | Eval Rows | Disk Size |
|----------|-----------|-----------|-----------|
| Gujarati | ~40,435   | ~2,130    | 19 GB     |
| Hindi    | ~11,134   | ~587      | 5.4 GB    |
| HF Cache |           |           | 50 GB     |

### 2.5 Analysis Results

From `analysis/data_analysis.json`:
- **Recommended max_text_length:** 319 (p99 + buffer)
- **Recommended max_audio_length:** 433966 (p99 at 22050 Hz) → used 330750 (~15s) in practice

### ⚠️ ISSUE: Pipe characters in text breaking CSV

**Problem:** Some Hindi transcripts contained literal `|` inside the text field (e.g., `"... मॉम | आई ड..."`), which broke the 3-column CSV parsing.

**Solution:** Post-processing script that joins all middle fields as text:
```python
parts = line.split("|")
audio_file = parts[0]
speaker = parts[-1]
text = " ".join(parts[1:-1]).replace("|", " ").strip()
```

**Prevention:** In `download_and_prepare_data.py`, add `.replace("|", " ")` alongside `.replace(">", "")` in the `clean_text` function.

---

## 3. Pretrained Model & Vocabulary Extension

### 3.1 Download Pretrained XTTS v2

```bash
python download_checkpoint.py --output_path checkpoints/
```

Downloads to `checkpoints/XTTS_v2.0_original_model_files/`:
- `model.pth` (1.8 GB)
- `dvae.pth` (201 MB)
- `vocab.json` (445 KB)
- `mel_stats.pth` (1.1 KB)
- `config.json` (4.3 KB)

### 3.2 Vocabulary Extension

**Hindi:** Already present in the base XTTS v2 vocab — no extension needed.

**Gujarati:** Extended with 512 new tokens (404 were actually added after deduplication):

```bash
python extend_vocab_config.py \
  --output_path=checkpoints/ \
  --metadata_path datasets-gu/metadata_train.csv \
  --language gu \
  --extended_vocab_size 512
```

Log confirmation: `> Loading checkpoint with 404 additional tokens.`

---

## 4. Training — Issues & Solutions

### ⚠️ ISSUE 1: PyTorch 2.6 `weights_only=True` Default

**Error:**
```
_pickle.UnpicklingError: Weights only load failed...
Unsupported global: GLOBAL TTS.tts.configs.xtts_config.XttsConfig
```

**Cause:** PyTorch 2.6 changed `torch.load()` default to `weights_only=True`. XTTS checkpoints contain serialized config objects.

**Fix:**
```bash
sed -i 's/return torch.load(f, map_location=map_location, \*\*kwargs)/return torch.load(f, map_location=map_location, weights_only=False, **kwargs)/' TTS/utils/io.py
```

---

### ⚠️ ISSUE 2: TorchCodec Not Found / Broken

**Error:**
```
ImportError: TorchCodec is required for load_with_torchcodec
RuntimeError: Could not load libtorchcodec. FFmpeg not properly installed...
```

**Cause:** `torchaudio 2.10.0` uses `torchcodec` as its audio backend, which requires FFmpeg shared libraries not present on the Lightning.ai system.

**This was the ROOT CAUSE of all OOM-kill crashes.** The `torchcodec` import failure during `torchaudio.load()` in the dataloader worker caused the process to be killed by the OS.

**Fix — Patched `load_audio` to use `soundfile` instead of `torchaudio`:**

In `TTS/tts/models/xtts.py`, replaced:
```python
audio, lsr = torchaudio.load(audiopath)
```

With:
```python
import soundfile as sf
import numpy as np
audio_np, lsr = sf.read(audiopath, dtype="float32")
import torch as _torch
audio = _torch.from_numpy(audio_np).unsqueeze(0)  # (1, samples)
```

The rest of the function (stereo→mono, resampling, clipping) remained unchanged.

---

### ⚠️ ISSUE 3: DataLoader Workers OOM-Killed

**Error:**
```
RuntimeError: DataLoader worker (pid XXXXX) is killed by signal: Killed.
```

**Cause:** Default `num_loader_workers=8` spawned 8 forked processes, each duplicating parent memory.

**Fix:**
```bash
# In train_gpt_xtts.py, set:
config.num_loader_workers = 0    # Load data in main process
config.batch_group_size = 0      # Disable batch grouping (high RAM usage)
```

---

### ⚠️ ISSUE 4: Process Killed During `trainer.fit()` Even with Tiny Dataset

**Error:** Process killed silently with no CUDA error — just `Killed`.

**Diagnosis process:**
1. Confirmed model loads fine (2.84 GB CPU, 1.98 GB GPU) ✅
2. Confirmed Trainer initializes fine (1.9 GB RAM) ✅
3. Confirmed dataloader can fetch a batch ✅ (only after soundfile fix)
4. Kill happened specifically during first forward pass in `trainer.fit()`

**Root cause:** Same as Issue 2 — the `torchcodec` crash during audio loading in the dataloader wasn't a clean Python exception; it caused a segfault/memory corruption that the OS OOM-killer detected.

**Fix:** The soundfile patch (Issue 2) resolved this completely.

---

### ℹ️ ISSUE 5: Character Limit Warning (Informational Only)

**Warning:**
```
[!] Warning: The text length exceeds the character limit of 300 for language 'hi'
```

**This is harmless.** It warns about potential truncation during inference, not training. Our `max_text_length=319` ensures the model trains on these longer texts. No action needed.

---

## 5. Final Working Training Command

### 5.1 All Patches Applied to the Repo

| File | Change |
|------|--------|
| `TTS/utils/io.py` | `weights_only=False` in `torch.load` |
| `TTS/tts/models/xtts.py` | `load_audio()` uses `soundfile` instead of `torchaudio.load` |
| `train_gpt_xtts.py` | `config.num_loader_workers = 0` |
| `train_gpt_xtts.py` | `config.batch_group_size = 0` |

### 5.2 Optimized Training Command (L4 GPU)

```bash
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
  --output_path checkpoints/ \
  --metadatas \
    /path/to/datasets-gu/metadata_train.csv,/path/to/datasets-gu/metadata_eval.csv,gu \
    /path/to/datasets-hi/metadata_train.csv,/path/to/datasets-hi/metadata_eval.csv,hi \
  --num_epochs 5 \
  --batch_size 4 \
  --grad_acumm 8 \
  --max_text_length 319 \
  --max_audio_length 330750 \
  --weight_decay 1e-2 \
  --lr 5e-6 \
  --save_step 2000
```

### 5.3 Resource Usage at Final Settings

| Metric | Value |
|--------|-------|
| GPU VRAM Used | ~13.6 GB (batch_size=2), ~20-22 GB (batch_size=4) |
| GPU VRAM Total | 24 GB |
| GPU Utilization | 80-100% |
| System RAM Used | ~2 GB for model + training |
| System RAM Total | 31 GB |
| Effective Batch Size | 4 × 8 = 32 |
| Steps per Epoch | ~12,892 (batch_size=4) |
| Step Time | ~0.2s |
| Estimated Total Time | ~3.5 hours (5 epochs) |

### 5.4 Early Training Metrics (Healthy)

| Step | loss | loss_text_ce | loss_mel_ce |
|------|------|-------------|-------------|
| 0    | 0.307 | 0.181      | 4.735       |
| 100  | 0.290 | 0.168      | 4.482       |
| 200  | 0.284 | 0.165      | 4.375       |
| 300  | 0.283 | 0.162      | 4.363       |
| 400  | 0.280 | 0.159      | 4.326       |

All losses trending downward — healthy training.

---

## 6. Key Lessons for Future Projects

### 6.1 Environment & Dependencies

1. **Always check `torchaudio` version.** Version 2.10+ uses `torchcodec` which requires FFmpeg. If FFmpeg isn't available, patch `load_audio` to use `soundfile` directly.
2. **PyTorch 2.6+ changed `torch.load` defaults.** Any code loading older checkpoints needs `weights_only=False`.
3. **Pin versions when possible.** Using `torchaudio==2.5.1` avoids the torchcodec issue entirely.

### 6.2 Data Preparation

1. **Sanitize text for delimiter characters.** If using `|` as CSV delimiter, strip `|` from text fields during data prep.
2. **Also strip `>` and other special characters** that could cause parsing issues.
3. **Cache HuggingFace downloads** with `cache_dir=` parameter. Re-downloading 50+ GB datasets is wasteful.
4. **Use checkpoint/resume logic** for long data preparation jobs.
5. **Run data analysis BEFORE training**: duration histograms, character length percentiles → set `max_audio_length` and `max_text_length` from p99 values.

### 6.3 Training Configuration

1. **Start with `num_loader_workers=0`** on memory-constrained systems. Increase only after confirming training runs.
2. **Set `batch_group_size=0`** to disable memory-hungry batch sorting.
3. **Use mixed precision (`config.mixed_precision = True`)** to halve GPU memory usage.
4. **Scale batch_size based on VRAM usage**, not guesses. Monitor VRAM for the first 50 steps, then adjust.
5. **Effective batch size = batch_size × grad_acumm.** XTTS recommends ≥252. We used 32 (works fine, just slower convergence).
6. **`max_audio_length=330750` (~15s at 22050 Hz)** is a good default. Going to 255995 (~11.6s) is safer but drops longer clips. Don't go above 330750 unless VRAM allows.

### 6.4 Debugging Strategy

When training crashes with "Killed" (OOM):

1. **Don't guess — isolate systematically:**
   - Test model loading alone → check RAM/VRAM
   - Test Trainer initialization → check RAM/VRAM
   - Test single batch fetch → check for crashes
   - Test forward pass → check VRAM
   - Test backward pass → check VRAM
2. **Use `psutil` for RAM monitoring** at each step.
3. **Use `torch.cuda.memory_allocated()` for VRAM monitoring.**
4. **Check `dmesg | tail` for OOM-killer logs** (if permissions allow).
5. **Check `ulimit -v` and cgroup limits** for container-level memory caps.

### 6.5 Vocab Extension

1. **Check existing vocab before extending.** Hindi tokens were already in XTTS v2 — extending again would have been wasteful.
2. **Run extension only for truly new languages** (Gujarati needed 512 → 404 unique tokens added).
3. **Don't run `extend_vocab_config.py` twice on the same checkpoint** for different languages without understanding how it modifies the files.

---

## 7. Dataset Structure Reference

### XTTS Expected Directory Layout

```
project_root/
├── datasets-gu/
│   ├── wavs/
│   │   ├── clip_001.wav
│   │   ├── clip_002.wav
│   │   └── ...
│   ├── metadata_train.csv
│   └── metadata_eval.csv
├── datasets-hi/
│   ├── wavs/
│   │   └── ...
│   ├── metadata_train.csv
│   └── metadata_eval.csv
├── checkpoints/
│   └── XTTS_v2.0_original_model_files/
│       ├── model.pth
│       ├── dvae.pth
│       ├── vocab.json
│       ├── mel_stats.pth
│       └── config.json
├── hf_cache/                    ← HuggingFace download cache
├── analysis/
│   ├── data_analysis.json
│   ├── analysis_gu.png
│   └── analysis_hi.png
└── train_gpt_xtts.py
```

### CSV Format

```
audio_file|text|speaker_name
wavs/clip_id.wav|Transcript text here|speaker_identifier
```

- Delimiter: `|` (pipe)
- No quoting
- First line: header
- Text must NOT contain `|` characters

---

## 8. Hyperparameter Reference

### L4 GPU (24 GB VRAM) — Recommended Settings

| Parameter | Conservative | Optimized | Max Push |
|-----------|-------------|-----------|----------|
| batch_size | 2 | 4 | 6 (test first) |
| grad_acumm | 16 | 8 | 4 |
| effective_batch | 32 | 32 | 24 |
| max_audio_length | 255995 (~11.6s) | 330750 (~15s) | 433966 (~20s) |
| max_text_length | 200 | 319 | 400 |
| num_loader_workers | 0 | 0 | 2 (if RAM allows) |
| VRAM usage | ~10 GB | ~20 GB | ~23 GB |

### T4 GPU (16 GB VRAM) — Recommended Settings

| Parameter | Conservative | Optimized |
|-----------|-------------|-----------|
| batch_size | 2 | 2 |
| grad_acumm | 16 | 16 |
| max_audio_length | 255995 | 255995 |
| num_loader_workers | 0 | 0 |

### General Guidelines

- `lr = 5e-6` works well for fine-tuning
- `weight_decay = 1e-2` is standard
- `save_step = 2000` balances checkpoint frequency vs disk
- `num_epochs = 5` is a good starting point; monitor loss for overfitting after epoch 3
- `lr_scheduler = MultiStepLR` with milestones at [900k, 2.7M, 5.4M] global steps

---

## Appendix: Complete Patch Diff

### TTS/utils/io.py
```diff
- return torch.load(f, map_location=map_location, **kwargs)
+ return torch.load(f, map_location=map_location, weights_only=False, **kwargs)
```

### TTS/tts/models/xtts.py — load_audio function
```diff
- audio, lsr = torchaudio.load(audiopath)
+ # Bypass torchcodec — use soundfile directly
+ import soundfile as sf
+ import numpy as np
+ audio_np, lsr = sf.read(audiopath, dtype="float32")
+ import torch as _torch
+ audio = _torch.from_numpy(audio_np).unsqueeze(0)  # (1, samples)
```

### train_gpt_xtts.py
```diff
- config.num_loader_workers = 8
+ config.num_loader_workers = 0
```
```diff
+ config.batch_group_size = 0   # Added after num_loader_workers line
```