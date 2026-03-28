#!/bin/bash
# Apply all necessary patches to the cloned XTTS repo
# Run from inside the XTTSv2-Finetuning-for-New-Languages directory

echo "Applying patches..."

# 1. Fix torch.load weights_only for PyTorch 2.6+
sed -i 's/return torch.load(f, map_location=map_location, \*\*kwargs)/return torch.load(f, map_location=map_location, weights_only=False, **kwargs)/' TTS/utils/io.py
echo "  ✅ TTS/utils/io.py — weights_only=False"

# 2. Replace torchaudio.load with soundfile (bypass torchcodec)
python << 'PATCH'
with open("TTS/tts/models/xtts.py", "r") as f:
    content = f.read()

old = "    audio, lsr = torchaudio.load(audiopath)"
new = """    # Bypass torchcodec — use soundfile directly
    import soundfile as sf
    import numpy as np
    audio_np, lsr = sf.read(audiopath, dtype="float32")
    import torch as _torch
    audio = _torch.from_numpy(audio_np).unsqueeze(0)  # (1, samples)"""

if old in content:
    content = content.replace(old, new)
    with open("TTS/tts/models/xtts.py", "w") as f:
        f.write(content)
    print("  ✅ TTS/tts/models/xtts.py — soundfile patch applied")
else:
    print("  ⚠️  TTS/tts/models/xtts.py — already patched or line not found")
PATCH

# 3. Set num_loader_workers=0 and batch_group_size=0
sed -i 's/config.num_loader_workers = 8/config.num_loader_workers = 0/' train_gpt_xtts.py
# Add batch_group_size=0 if not already present
grep -q "batch_group_size" train_gpt_xtts.py || sed -i '/config.num_loader_workers/a\    config.batch_group_size = 0' train_gpt_xtts.py
echo "  ✅ train_gpt_xtts.py — workers=0, batch_group_size=0"

echo ""
echo "All patches applied! You can now run training."
