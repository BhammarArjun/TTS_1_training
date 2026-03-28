"""
XTTS v2 Production Inference Script
- Auto-splits long text into sentence chunks
- Smart generation params based on text length
- Concatenates chunks seamlessly
- Duration analysis and quality checks
"""

import torch
import soundfile as sf
import os
import shutil
import re
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ─── PATHS ────────────────────────────────────────────────────────────────────
xtts_checkpoint = "checkpoints/GPT_XTTS_FT-March-19-2026_09+45AM-8e59ec3/best_model_64460.pth"
xtts_config = "checkpoints/GPT_XTTS_FT-March-19-2026_09+45AM-8e59ec3/config.json"
xtts_vocab = "checkpoints/XTTS_v2.0_original_model_files/vocab.json"

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MAX_CHARS_PER_CHUNK = 250       # max characters before splitting
OVERLAP_SILENCE_SEC = 0.15      # small silence between chunks (seconds)
OUTPUT_SAMPLE_RATE = 24000      # XTTS output sample rate
CHARS_PER_SEC_GU = 13           # approximate Gujarati speech rate
CHARS_PER_SEC_HI = 14           # approximate Hindi speech rate


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
print("Loading model...")
config = XttsConfig()
config.load_json(xtts_config)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
model.to(device)
print("Model loaded!\n")


# ─── TEXT SPLITTING ───────────────────────────────────────────────────────────

def split_text_gujarati(text, max_chars=MAX_CHARS_PER_CHUNK):
    """
    Splits Gujarati/Hindi text into chunks that fit within max_chars.
    
    Priority order for split points:
      1. Sentence endings:  .  !  ?  ।  (Devanagari danda)
      2. Comma and semicolons:  ,  ;  ،
      3. Mid-sentence natural breaks (conjunctions, postpositions)
      4. Hard split at max_chars if nothing else works
    """
    text = text.strip()
    
    # If text fits, return as-is
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    remaining = text
    
    while len(remaining) > max_chars:
        # Search window: look for split point within max_chars
        window = remaining[:max_chars]
        
        # Priority 1: sentence endings
        split_pos = -1
        for sep in ['. ', '! ', '? ', '।', '.\n', '!\n', '?\n']:
            pos = window.rfind(sep)
            if pos > 0:
                split_pos = max(split_pos, pos + len(sep))
        
        # Priority 2: commas and semicolons
        if split_pos < 0:
            for sep in [', ', '; ', '،']:
                pos = window.rfind(sep)
                if pos > 0:
                    split_pos = max(split_pos, pos + len(sep))
        
        # Priority 3: Gujarati/Hindi conjunction words as split points
        if split_pos < 0:
            # Look for common conjunctions: અને (and), પણ (but/also), કે (that),
            # તો (then), પરંતુ (but), જેથી (so that), और (and-Hindi), लेकिन (but-Hindi)
            for conj in [' અને ', ' પણ ', ' કે ', ' તો ', ' પરંતુ ', ' જેથી ',
                         ' और ', ' लेकिन ', ' तो ', ' क्योंकि ', ' जो ']:
                pos = window.rfind(conj)
                if pos > 0:
                    split_pos = max(split_pos, pos + 1)  # split before conjunction
        
        # Priority 4: any space
        if split_pos < 0:
            pos = window.rfind(' ')
            if pos > 0:
                split_pos = pos + 1
        
        # Priority 5: hard cut (last resort)
        if split_pos < 0:
            split_pos = max_chars
        
        chunk = remaining[:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos:].strip()
    
    # Add the last piece
    if remaining:
        chunks.append(remaining)
    
    return chunks


# ─── GENERATION PARAMS BY LENGTH ──────────────────────────────────────────────

def get_generation_params(text):
    """Returns inference params tuned to text length."""
    char_count = len(text.strip())
    
    if char_count < 15:
        return {
            "temperature": 0.1,
            "repetition_penalty": 10.0,
            "top_k": 10,
            "top_p": 0.7,
            "length_penalty": 1.0,
            "do_sample": True,
        }
    elif char_count < 50:
        return {
            "temperature": 0.2,
            "repetition_penalty": 8.0,
            "top_k": 20,
            "top_p": 0.8,
            "length_penalty": 1.0,
            "do_sample": True,
        }
    elif char_count < 150:
        return {
            "temperature": 0.4,
            "repetition_penalty": 5.0,
            "top_k": 30,
            "top_p": 0.85,
            "length_penalty": 1.0,
            "do_sample": True,
        }
    elif char_count < 300:
        return {
            "temperature": 0.65,
            "repetition_penalty": 3.0,
            "top_k": 50,
            "top_p": 0.9,
            "length_penalty": 1.0,
            "do_sample": True,
        }
    else:
        return {
            "temperature": 0.7,
            "repetition_penalty": 2.0,
            "top_k": 50,
            "top_p": 0.95,
            "length_penalty": 1.0,
            "do_sample": True,
        }


# ─── MAIN GENERATION FUNCTION ────────────────────────────────────────────────

def generate_speech(
    model,
    text,
    language,
    ref_audio_path,
    output_path,
    gpt_cond_len=6,
    max_ref_length=10,
    verbose=True,
):
    """
    Generate speech from text. Automatically splits long text into chunks,
    generates each chunk, and concatenates into a single wav file.
    
    Args:
        model: loaded XTTS model
        text: input text (any length)
        language: "gu" or "hi"
        ref_audio_path: path to reference speaker audio
        output_path: where to save the output wav
        gpt_cond_len: conditioning length in seconds
        max_ref_length: max reference audio length in seconds
        verbose: print progress
    
    Returns:
        dict with metadata about the generation
    """
    # Get speaker conditioning (compute once, reuse for all chunks)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=ref_audio_path,
        gpt_cond_len=gpt_cond_len,
        max_ref_length=max_ref_length,
    )
    
    # Split text into chunks
    chunks = split_text_gujarati(text, max_chars=MAX_CHARS_PER_CHUNK)
    
    if verbose:
        if len(chunks) > 1:
            print(f"  Text split into {len(chunks)} chunks:")
            for j, chunk in enumerate(chunks):
                print(f"    Chunk {j+1} ({len(chunk)} chars): {chunk[:60]}{'...' if len(chunk) > 60 else ''}")
        else:
            print(f"  Single chunk ({len(text)} chars)")
    
    # Generate each chunk
    all_audio = []
    silence = np.zeros(int(OVERLAP_SILENCE_SEC * OUTPUT_SAMPLE_RATE), dtype=np.float32)
    
    for j, chunk in enumerate(chunks):
        params = get_generation_params(chunk)
        
        out = model.inference(
            text=chunk,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            **params,
        )
        
        wav_data = out["wav"]
        if isinstance(wav_data, torch.Tensor):
            wav_data = wav_data.cpu().numpy()
        
        all_audio.append(wav_data)
        
        # Add small silence between chunks (not after last)
        if j < len(chunks) - 1:
            all_audio.append(silence)
        
        if verbose:
            chunk_dur = len(wav_data) / OUTPUT_SAMPLE_RATE
            print(f"    Chunk {j+1} generated: {chunk_dur:.2f}s")
    
    # Concatenate all chunks
    final_audio = np.concatenate(all_audio)
    total_duration = len(final_audio) / OUTPUT_SAMPLE_RATE
    
    # Save
    sf.write(output_path, final_audio, OUTPUT_SAMPLE_RATE, subtype="PCM_16")
    
    # Compute expected duration
    chars_per_sec = CHARS_PER_SEC_GU if language == "gu" else CHARS_PER_SEC_HI
    expected_duration = len(text) / chars_per_sec
    
    result = {
        "text": text,
        "language": language,
        "chars": len(text),
        "chunks": len(chunks),
        "expected_sec": round(expected_duration, 2),
        "actual_sec": round(total_duration, 2),
        "ratio": round(total_duration / max(expected_duration, 0.1), 2),
        "output_path": output_path,
    }
    
    if verbose:
        print(f"  Total: {total_duration:.2f}s (expected ~{expected_duration:.1f}s) → {output_path}")
    
    return result


# ─── TEST SAMPLES ─────────────────────────────────────────────────────────────
reference_key = 9
os.makedirs(f"generated_samples_{reference_key}", exist_ok=True)

gu_ref = os.path.join("datasets-gu/wavs", sorted(os.listdir("datasets-gu/wavs"))[3567])
# gu_ref = "alice2.wav"
shutil.copy(gu_ref, f"generated_samples_{reference_key}/reference_audio.wav")
print(f"Reference audio: {gu_ref}\n")

samples = [
    ("gu", gu_ref, "આજે થોડું મોડું ઊઠ્યો પરંતુ દિવસની શરૂઆત શાંતિથી થઈ અને મન સ્થિર લાગ્યું."),
    ("gu", gu_ref, "બહારની ઠંડી હવામાં ચાલવા જવાથી તાજગી અનુભવી અને ઊર્જા વધી ગઈ."),
    
    ("gu", gu_ref, "કામ વચ્ચે થોડો વિરામ લઈને ચા પીવી એ દિવસને વધુ સરળ બનાવી દે છે."),
    ("gu", gu_ref, "આજે રસ્તા પર ઓછો ટ્રાફિક હતો એટલે સમયસર પહોંચવું સરળ બન્યું."),
    
    ("gu", gu_ref, "મને લાગે છે કે હવે નવી સ્કિલ શીખવાનો સમય આવી ગયો છે અને શરૂઆત પણ કરવી જોઈએ."),
    ("gu", gu_ref, "ઘરે બનાવેલું ખાવાનું હંમેશા બહારના ખાવાથી વધુ સ્વાદિષ્ટ અને સ્વસ્થ લાગે છે."),
    
    ("gu", gu_ref, "આજે થોડો સમય જૂના ફોટા જોઈને યાદો તાજી કરી અને મન ખુશ થઈ ગયું."),
    ("gu", gu_ref, "જ્યારે કામ પૂરું થાય ત્યારે એક પ્રકારની સંતોષની લાગણી મળે છે જે ખૂબ મહત્વની છે."),
    
    ("gu", gu_ref, "હવે ધીમે ધીમે આદતો સુધારવાની જરૂર છે જેથી જીવન વધુ વ્યવસ્થિત બની શકે."),
    ("gu", gu_ref, "સાંજના સમયે થોડી વોક પર જવું શરીર અને મન બંને માટે ખૂબ ફાયદાકારક છે.")
]
# ─── GENERATE ALL ─────────────────────────────────────────────────────────────
print("=" * 80)
print("GENERATING SAMPLES")
print("=" * 80)

results = []
for i, (lang, ref_audio, text) in enumerate(samples):
    print(f"\n[{i+1}/{len(samples)}] ({len(text)} chars)")
    
    out_path = f"generated_samples_{reference_key}/sample_{i+1:02d}.wav"
    result = generate_speech(model, text, lang, ref_audio, out_path)
    results.append(result)


# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"{'#':<4} {'Chars':<7} {'Chunks':<8} {'Expected':<10} {'Actual':<10} {'Ratio':<8} {'Text'}")
print("=" * 80)

for r in results:
    text_preview = r["text"][:50] + "..." if len(r["text"]) > 50 else r["text"]
    status = "⚠️" if r["ratio"] > 2.5 or r["ratio"] < 0.3 else "✅"
    print(f"{r['chars']:<7} {r['chunks']:<8} {r['expected_sec']:<10} {r['actual_sec']:<10} {r['ratio']:<8} {status} {text_preview}")

multi_chunk = [r for r in results if r["chunks"] > 1]
if multi_chunk:
    print(f"\n{len(multi_chunk)} samples were auto-split into multiple chunks.")

print(f"\nAll {len(results)} samples saved to generated_samples/")