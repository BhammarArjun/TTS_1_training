# Patches Applied to the XTTS Repo

These patches are applied to the original repo:
`https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages`

Apply them after cloning the repo and installing requirements.

## 1. TTS/utils/io.py — Fix PyTorch 2.6+ weights_only default
```bash
sed -i 's/return torch.load(f, map_location=map_location, \*\*kwargs)/return torch.load(f, map_location=map_location, weights_only=False, **kwargs)/' TTS/utils/io.py
```

## 2. TTS/tts/models/xtts.py — Replace torchaudio.load with soundfile

torchaudio 2.10+ uses torchcodec which requires FFmpeg shared libraries.
On systems without FFmpeg (e.g., Lightning.ai), this causes silent OOM kills.

Replace line 73:
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

## 3. train_gpt_xtts.py — Reduce dataloader workers and batch grouping
```bash
sed -i 's/config.num_loader_workers = 8/config.num_loader_workers = 0/' train_gpt_xtts.py
sed -i '/config.num_loader_workers/a\    config.batch_group_size = 0' train_gpt_xtts.py
```
