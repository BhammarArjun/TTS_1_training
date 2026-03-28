# XTTS v2 Fine-Tuning for Gujarati & Hindi

Fine-tuning [XTTS v2](https://huggingface.co/coqui/XTTS-v2) for Gujarati (gu) and Hindi (hi) text-to-speech using the [XTTSv2-Finetuning-for-New-Languages](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages) repo.

## Model

Trained model available at: [Arjun4707/xtts-v2-gujarati-hindi](https://huggingface.co/Arjun4707/xtts-v2-gujarati-hindi)


## Dataset

Private dataset: `Arjun4707/gu-hi-tts` (~65,700 clips)
- Gujarati: ~40,435 train / ~2,130 eval (19 GB)
- Hindi: ~11,134 train / ~587 eval (5.4 GB)
- Audio: 24kHz mono PCM-16, silence-trimmed, normalized to -3 dBFS

## Quick Start

### 1. Clone the XTTS training repo
```bash
git clone https://github.com/nguyenhoanganh2002/XTTSv2-Finetuning-for-New-Languages.git
cd XTTSv2-Finetuning-for-New-Languages
pip install -r requirements.txt
pip install datasets soundfile huggingface_hub matplotlib pandas psutil
```

### 2. Apply patches (critical!)
```bash
# Copy apply_patches.sh into the cloned repo, then:
bash apply_patches.sh
```

### 3. Login to HuggingFace
```bash
huggingface-cli login
```

### 4. Download & prepare data
```bash
python download_and_prepare_data.py
```

### 5. Download pretrained checkpoint & extend vocab
```bash
python download_checkpoint.py --output_path checkpoints/

# Gujarati vocab extension (Hindi already in base vocab)
python extend_vocab_config.py \
  --output_path=checkpoints/ \
  --metadata_path datasets-gu/metadata_train.csv \
  --language gu \
  --extended_vocab_size 512
```

### 6. Train
```bash
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
  --output_path checkpoints/ \
  --metadatas \
    datasets-gu/metadata_train.csv,datasets-gu/metadata_eval.csv,gu \
    datasets-hi/metadata_train.csv,datasets-hi/metadata_eval.csv,hi \
  --num_epochs 5 \
  --batch_size 4 \
  --grad_acumm 8 \
  --max_text_length 319 \
  --max_audio_length 330750 \
  --weight_decay 1e-2 \
  --lr 5e-6 \
  --save_step 2000
```

### 7. Inference
```bash
python inference_test.py
```

## Training Results (L4 GPU, 24 GB VRAM)

| Epoch | loss_text_ce | loss_mel_ce | loss |
|-------|-------------|-------------|------|
| 1     | 0.055       | 3.943       | 0.500 |
| 2     | 0.039       | 3.585       | 0.453 |
| 3     | 0.037       | 3.488       | 0.441 |

All losses trending down with no overfitting through 5 epochs.

## Issues & Fixes

See [TRAINING_JOURNAL.md](TRAINING_JOURNAL.md) for the complete troubleshooting guide covering:
- PyTorch 2.6 `weights_only` breaking checkpoint loading
- `torchcodec`/FFmpeg causing silent OOM kills (the big one!)
- DataLoader worker OOM from forked processes
- Pipe `|` characters in Hindi text breaking CSV parsing
- Hyperparameter tuning for L4 vs T4 GPUs

## Project Structure
```
.
├── README.md                      ← This file
├── TRAINING_JOURNAL.md            ← Complete training journal with all issues & fixes
├── apply_patches.sh               ← One-click patch script for the XTTS repo
├── download_and_prepare_data.py   ← HF dataset → wav + CSV (with caching & resume)
├── train_gpt_xtts.py              ← Modified training script
├── inference_test.py              ← Production inference with auto-splitting
├── requirements.txt               ← Python dependencies
├── analysis/                      ← Duration/character analysis plots & JSON
├── patches/                       ← Documentation of all repo patches
└── .gitignore
```

## Hardware Reference

| GPU | batch_size | grad_acumm | max_audio_length | VRAM Used |
|-----|-----------|------------|------------------|-----------|
| L4 (24 GB) | 4 | 8 | 330750 (~15s) | ~20 GB |
| T4 (16 GB) | 2 | 16 | 255995 (~11.6s) | ~13 GB |



## Data provenance

The training dataset (`Arjun4707/gu-hi-tts`) was constructed by scraping audio from publicly available YouTube videos, followed by automatic transcription and audio preprocessing (silence trimming, peak normalization, mono conversion to 24kHz PCM-16).

**This means:**
- The trained model weights and generated audio are for **non-commercial use only**
- Reference audio clips from the training data should not be redistributed
- If using this model, please be transparent that the voice was synthesized

## License

- **Training scripts** in this repo: MIT
- **Trained model weights** ([Arjun4707/xtts-v2-gujarati-hindi](https://huggingface.co/Arjun4707/xtts-v2-gujarati-hindi)): CC-BY-NC-4.0 (non-commercial only)
  - Inherited from [Coqui Public Model License](https://coqui.ai/cpml) (base XTTS v2)
  - YouTube-sourced training data adds further non-commercial restriction
- Training framework: [XTTSv2-Finetuning-for-New-Languages](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages) by anhnh2002

## Author

**Arjun Bhammar** — [HuggingFace](https://huggingface.co/Arjun4707) | [GitHub](https://github.com/BhammarArjun)

