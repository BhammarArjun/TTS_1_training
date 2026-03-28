"""
Downloads data from HF, cleans text (removes '>'), saves wavs + metadata CSVs
in the format expected by XTTSv2-Finetuning-for-New-Languages repo.

Expected output structure:
  datasets-gu/wavs/*.wav    + metadata_train.csv + metadata_eval.csv
  datasets-hi/wavs/*.wav    + metadata_train.csv + metadata_eval.csv

Metadata format: audio_file|text|speaker_name
"""

import os
import io
import json
import soundfile as sf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter

# ─── CONFIG ───────────────────────────────────────────────────────────────────
HF_REPO       = "Arjun4707/gu-hi-tts"
EVAL_RATIO     = 0.05
SEED           = 42
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR      = os.path.join(BASE_DIR, "hf_cache")     # persistent cache
ANALYSIS_DIR   = os.path.join(BASE_DIR, "analysis")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "data_prep_checkpoint.json")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ─── LOAD (uses cache on subsequent runs) ─────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading dataset from HuggingFace (cached if available)")
print("=" * 60)
ds = load_dataset(HF_REPO, split="train", cache_dir=CACHE_DIR)
print(f"Total rows loaded: {len(ds)}")

# ─── CLEAN TEXT: remove '>' symbol ─────────────────────────────────────────────
print("\nSTEP 2: Cleaning text (removing '>' characters)")
def clean_text(example):
    example["text"] = example["text"].replace(">", "").replace("|", " ")
    return example

ds = ds.map(clean_text)

# ─── ANALYSIS BEFORE SPLITTING ────────────────────────────────────────────────
print("\nSTEP 3: Running analysis by language")

# Convert relevant columns to a DataFrame for analysis
records = []
for i, row in enumerate(ds):
    records.append({
        "id": row["id"],
        "language": row["language"],
        "duration_sec": row["duration_sec"],
        "text_len": len(row["text"]),
        "text": row["text"],
    })
df = pd.DataFrame(records)

# ── 3a. Summary: total audio hours per language ──
print("\n" + "─" * 50)
print("AUDIO HOURS SUMMARY BY LANGUAGE")
print("─" * 50)
lang_summary = df.groupby("language").agg(
    count=("duration_sec", "count"),
    total_hours=("duration_sec", lambda x: x.sum() / 3600),
    mean_dur=("duration_sec", "mean"),
    median_dur=("duration_sec", "median"),
    min_dur=("duration_sec", "min"),
    max_dur=("duration_sec", "max"),
    std_dur=("duration_sec", "std"),
).round(3)
print(lang_summary.to_string())

total_hours = df["duration_sec"].sum() / 3600
print(f"\nTotal across all languages: {total_hours:.2f} hours  ({len(df)} clips)")

# ── 3b. Character length analysis per language ──
print("\n" + "─" * 50)
print("CHARACTER LENGTH ANALYSIS BY LANGUAGE")
print("─" * 50)
char_summary = df.groupby("language").agg(
    mean_chars=("text_len", "mean"),
    median_chars=("text_len", "median"),
    p95_chars=("text_len", lambda x: np.percentile(x, 95)),
    p99_chars=("text_len", lambda x: np.percentile(x, 99)),
    max_chars=("text_len", "max"),
).round(1)
print(char_summary.to_string())

# ── 3c. Duration analysis per language ──
print("\n" + "─" * 50)
print("DURATION ANALYSIS BY LANGUAGE (seconds)")
print("─" * 50)
dur_summary = df.groupby("language").agg(
    mean_dur=("duration_sec", "mean"),
    median_dur=("duration_sec", "median"),
    p95_dur=("duration_sec", lambda x: np.percentile(x, 95)),
    p99_dur=("duration_sec", lambda x: np.percentile(x, 99)),
    max_dur=("duration_sec", "max"),
).round(3)
print(dur_summary.to_string())

# ── 3d. Compute recommended max_text_length and max_audio_length ──
# Use p99 to avoid extreme outliers
recommended_max_text = int(np.percentile(df["text_len"], 99)) + 10  # small buffer
recommended_max_audio_sec = float(np.percentile(df["duration_sec"], 99))
recommended_max_audio_samples = int(recommended_max_audio_sec * 22050)  # XTTS uses 22050 Hz

print("\n" + "─" * 50)
print("RECOMMENDED TRAINING PARAMETERS")
print("─" * 50)
print(f"max_text_length  : {recommended_max_text}  (p99 chars + buffer)")
print(f"max_audio_length : {recommended_max_audio_samples}  (p99 duration = {recommended_max_audio_sec:.2f}s @ 22050 Hz)")

# ── 3e. Save plots ──
for lang in df["language"].unique():
    sub = df[df["language"] == lang]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Language: {lang}  (n={len(sub)})", fontsize=14)

    axes[0].hist(sub["duration_sec"], bins=60, edgecolor="black", alpha=0.7)
    axes[0].set_title("Duration Distribution (seconds)")
    axes[0].set_xlabel("Duration (s)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(sub["duration_sec"].median(), color="red", linestyle="--", label=f"median={sub['duration_sec'].median():.1f}s")
    axes[0].legend()

    axes[1].hist(sub["text_len"], bins=60, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_title("Character Length Distribution")
    axes[1].set_xlabel("Characters")
    axes[1].set_ylabel("Count")
    axes[1].axvline(sub["text_len"].median(), color="red", linestyle="--", label=f"median={sub['text_len'].median():.0f}")
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(ANALYSIS_DIR, f"analysis_{lang}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")

# Save analysis JSON for use by training scripts
analysis = {
    "recommended_max_text_length": recommended_max_text,
    "recommended_max_audio_length": recommended_max_audio_samples,
    "recommended_max_audio_sec": round(recommended_max_audio_sec, 3),
    "total_hours": round(total_hours, 2),
    "per_language": {},
}
for lang in df["language"].unique():
    sub = df[df["language"] == lang]
    analysis["per_language"][lang] = {
        "count": int(len(sub)),
        "hours": round(sub["duration_sec"].sum() / 3600, 2),
        "p99_chars": int(np.percentile(sub["text_len"], 99)),
        "p99_dur_sec": round(float(np.percentile(sub["duration_sec"], 99)), 3),
    }

analysis_path = os.path.join(ANALYSIS_DIR, "data_analysis.json")
with open(analysis_path, "w") as f:
    json.dump(analysis, f, indent=2)
print(f"\nSaved analysis JSON: {analysis_path}")

# ─── SPLIT AND SAVE PER-LANGUAGE DATASETS ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Splitting and saving per-language datasets")
print("=" * 60)

# Load checkpoint if exists
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f:
        checkpoint = json.load(f)
    print(f"Resuming from checkpoint: {checkpoint.get('last_completed', 'none')}")
else:
    checkpoint = {"completed_languages": [], "last_completed": None}

for lang in ["gu", "hi"]:
    if lang in checkpoint["completed_languages"]:
        print(f"\n[SKIP] {lang} already completed (from checkpoint)")
        continue

    print(f"\n── Processing language: {lang} ──")

    # Filter dataset by language
    lang_ds = ds.filter(lambda x: x["language"] == lang)
    print(f"  Rows for {lang}: {len(lang_ds)}")

    if len(lang_ds) == 0:
        print(f"  WARNING: No data for language '{lang}', skipping.")
        continue

    # Train/eval split
    split = lang_ds.train_test_split(test_size=EVAL_RATIO, seed=SEED)
    train_ds = split["train"]
    eval_ds  = split["test"]
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Create directory structure
    dataset_dir = os.path.join(BASE_DIR, f"datasets-{lang}")
    wavs_dir    = os.path.join(dataset_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    # Helper to save split
    def save_split(split_ds, csv_name):
        rows = []
        for i, row in enumerate(split_ds):
            clip_id  = row["id"]
            text     = row["text"].strip()
            if not text:
                continue

            # Decode audio bytes → wav file
            audio_bytes = row["audio"]["bytes"]
            sr          = row["audio"]["sampling_rate"]
            audio_array, _ = sf.read(io.BytesIO(audio_bytes), dtype="float32")

            wav_filename = f"{clip_id}.wav"
            wav_path     = os.path.join(wavs_dir, wav_filename)

            if not os.path.exists(wav_path):
                sf.write(wav_path, audio_array, sr, subtype="PCM_16")

            # speaker_name = video_id or generic
            speaker = row.get("video_id", f"speaker_{lang}")
            rows.append(f"wavs/{wav_filename}|{text}|{speaker}")

            if (i + 1) % 1000 == 0:
                print(f"    [{csv_name}] Processed {i+1}/{len(split_ds)}")

        csv_path = os.path.join(dataset_dir, csv_name)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        print(f"    Saved {csv_path} ({len(rows)} rows)")
        return len(rows)

    n_train = save_split(train_ds, "metadata_train.csv")
    n_eval  = save_split(eval_ds,  "metadata_eval.csv")

    # Update checkpoint
    checkpoint["completed_languages"].append(lang)
    checkpoint["last_completed"] = lang
    checkpoint[f"{lang}_train"] = n_train
    checkpoint[f"{lang}_eval"]  = n_eval
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"  Checkpoint saved for '{lang}'")

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)
print(f"  datasets-gu/  →  wavs/ + metadata_train.csv + metadata_eval.csv")
print(f"  datasets-hi/  →  wavs/ + metadata_train.csv + metadata_eval.csv")
print(f"  analysis/     →  plots + data_analysis.json")
