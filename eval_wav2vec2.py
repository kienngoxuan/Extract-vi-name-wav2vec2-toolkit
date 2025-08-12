# file: eval_wav2vec2.py
"""
Evaluation script (standalone).
Unzips a folder with .wav files, loads safetensors (optional) into model architecture from the hub, performs preprocessing and transcribes files.
Usage example:
python eval_wav2vec2.py --zip_path /content/person_name_500.zip --extract_dir /content/person_name_500 --local_weights /content/model.safetensors --model_id nguyenvulebinh/wav2vec2-large-vi-vlsp2020 --out_save_dir /content/my_wav2vec2_large
"""

import os
import re
import uuid
import zipfile
import glob
import logging
import argparse
from importlib.machinery import SourceFileLoader

import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2Processor

# optional libs
try:
    from safetensors.torch import load_file as load_safetensors
except Exception:
    load_safetensors = None

try:
    import noisereduce as nr
except Exception:
    nr = None

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

try:
    from scipy.io import wavfile
except Exception:
    wavfile = None

try:
    import librosa
except Exception:
    librosa = None

try:
    from unidecode import unidecode
except Exception:
    unidecode = None

try:
    from jiwer import wer
except Exception:
    wer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Defaults
LOCAL_WEIGHTS_DEFAULT = "/content/model.safetensors"
ZIP_PATH_DEFAULT = "/content/person_name_500.zip"
EXTRACT_DIR_DEFAULT = "/content/person_name_500"
OUT_SAVE_DIR_DEFAULT = "/content/my_wav2vec2_large"
DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"

# Audio helpers

def normalize_audio_pydub(input_file, output_file, target_level=-24):
    audio = AudioSegment.silent(duration=500) + AudioSegment.from_file(input_file) + AudioSegment.silent(duration=500)
    normalized = audio.apply_gain(target_level - audio.dBFS)
    normalized.export(output_file, format=output_file.split('.')[-1])


def remove_noise(input_file, output_file):
    rate, data = wavfile.read(input_file)
    if data.dtype != 'float32':
        data = data.astype('float32') / 32768.0
    reduced = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(output_file, rate, (reduced * 32768).astype('int16'))


def vietnamese_number_converter(text):
    number_mapping = {
        'không':'0','hông':'0','một':'1','mốt':'1','hai':'2','ba':'3','bốn':'4',
        'năm':'5','sáu':'6','bảy':'7','tám':'8','chín':'9'
    }
    words, result, i = text.split(), [], 0
    while i < len(words):
        w = ''.join(c for c in words[i].lower() if c.isalpha())
        if w in number_mapping:
            seq, punct, j = [], '', i
            while j < len(words) and ''.join(c for c in words[j].lower() if c.isalpha()) in number_mapping:
                punct_tmp = ''.join(c for c in words[j] if not c.isalpha())
                punct = punct_tmp or punct
                seq.append(''.join(c for c in words[j].lower() if c.isalpha()))
                j += 1
            if len(seq) > 2:
                result.append(''.join(number_mapping[x] for x in seq) + punct)
                i = j
            else:
                result.append(words[i]); i +=1
        else:
            result.append(words[i]); i +=1
    return ' '.join(result)


def transcribe_wav2vec(audio_path, processor_ref, model_ref, device=DEVICE_DEFAULT):
    if librosa is None:
        raise RuntimeError('librosa is required for audio loading')
    audio_arr, sr = librosa.load(audio_path, sr=16000)
    inputs = processor_ref(audio_arr, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model_ref(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor_ref.decode(pred_ids[0])
    return transcription.strip()


def evaluate_folder(zip_path=ZIP_PATH_DEFAULT, extract_dir=EXTRACT_DIR_DEFAULT, run_postprocess=False, model_id: str = None, local_weights: str = LOCAL_WEIGHTS_DEFAULT, out_save_dir: str = OUT_SAVE_DIR_DEFAULT, device: str = DEVICE_DEFAULT):
    if not os.path.exists(zip_path):
        logger.error("ZIP not found: %s", zip_path)
        return
    logger.info("Extracting %s -> %s", zip_path, extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)

    if model_id is None:
        raise ValueError('model_id must be provided')

    logger.info("Downloading model_handling.py from %s", model_id)
    try:
        model_script = hf_hub_download(repo_id=model_id, filename="model_handling.py")
        model_loader = SourceFileLoader("model_handling", model_script).load_module()
    except Exception as e:
        logger.warning("Could not download model_handling.py: %s — will try hub architecture.", e)
        model_loader = None

    logger.info("Loading processor from hub repo: %s", model_id)
    processor_ref = Wav2Vec2Processor.from_pretrained(model_id)

    logger.info("Instantiating model architecture (from hub)...")
    if model_loader is not None and hasattr(model_loader, 'Wav2Vec2ForCTC'):
        model_ref = model_loader.Wav2Vec2ForCTC.from_pretrained(model_id, trust_remote_code=True)
    else:
        # fallback to AutoModelForCTC
        from transformers import AutoModelForCTC
        model_ref = AutoModelForCTC.from_pretrained(model_id)

    # put model in eval mode BEFORE loading state_dict (we'll move to device after loading)
    model_ref.eval()

    # ---------------------------
    # Load weights (safetensors or pytorch .bin) robustly onto CPU, THEN move model to device
    # ---------------------------
    loaded_weights = False
    if local_weights and os.path.exists(local_weights):
        logger.info("Attempting to load local weights from %s", local_weights)

        # helper to attempt loading a dict into model (returns True on success)
        def try_load(sd_dict):
            try:
                model_ref.load_state_dict(sd_dict, strict=False)
                logger.info("Loaded state dict into model (strict=False).")
                return True
            except Exception as e:
                logger.warning("load_state_dict failed: %s", e)
                return False

        # 1) safetensors path
        if local_weights.endswith(".safetensors") and load_safetensors is not None:
            logger.info("Loading safetensors file...")
            sd = load_safetensors(local_weights)  # dict of numpy arrays
            # convert to torch tensors on CPU
            sd_torch = {k: torch.as_tensor(v).cpu() for k, v in sd.items()}

            # if top-level wrapper key exists (common: 'model' or 'state_dict'), unwrap
            if 'model' in sd_torch and isinstance(sd_torch['model'], dict):
                sd_torch = sd_torch['model']

            # try direct load
            if try_load(sd_torch):
                loaded_weights = True
            else:
                # remap keys: strip common prefixes like "model." or "module."
                remapped = {}
                for k, v in sd_torch.items():
                    newk = k
                    if newk.startswith("model."):
                        newk = newk[len("model."):]
                    if newk.startswith("module."):
                        newk = newk[len("module."):]
                    remapped[newk] = v
                if try_load(remapped):
                    loaded_weights = True
                else:
                    logger.warning("Remapped safetensors load failed; trying other heuristics...")

        # 2) fallback: try torch .bin or other torch checkpoint formats
        if (not loaded_weights):
            try:
                logger.info("Trying torch.load fallback for %s", local_weights)
                ckpt = torch.load(local_weights, map_location="cpu")
                # ckpt might be a state_dict directly, or a dict containing 'model'/'state_dict'
                if isinstance(ckpt, dict):
                    # common keys that wrap state dicts
                    candidates = []
                    if "state_dict" in ckpt:
                        candidates.append(ckpt["state_dict"])
                    if "model" in ckpt:
                        candidates.append(ckpt["model"])
                    # also allow ckpt itself to be the state dict
                    candidates.append(ckpt)
                    for cand in candidates:
                        if isinstance(cand, dict):
                            # ensure keys are strings and values are tensors
                            cand_torch = {k: (v.cpu() if isinstance(v, torch.Tensor) else torch.as_tensor(v).cpu())
                                        for k, v in cand.items()}
                            if try_load(cand_torch):
                                loaded_weights = True
                                break
                    if not loaded_weights:
                        logger.warning("torch.load found file but couldn't load any candidate dict into model.")
                else:
                    logger.warning("torch.load returned unexpected object; skipping.")
            except Exception as e:
                logger.warning("torch.load fallback failed: %s", e)

        if not loaded_weights:
            logger.warning("Could not load local weights fully. Model will run with hub weights (may be unfine-tuned).")
    else:
        logger.info("No local_weights provided or file not found: %s. Using hub weights as-is.", local_weights)

    # Now move model to device (after we attempted CPU loads)
    model_ref.to(device)
    model_ref.eval()

    os.makedirs(out_save_dir, exist_ok=True)
    logger.info("Saving model + processor to %s (this will save config + weights + processor files)", out_save_dir)
    try:
        model_ref.save_pretrained(out_save_dir)
    except Exception as e:
        logger.warning("Model.save_pretrained() raised: %s — attempting to save state_dict manually.", e)
        torch.save(model_ref.state_dict(), os.path.join(out_save_dir, "pytorch_model.bin"))

    processor_ref.save_pretrained(out_save_dir)
    logger.info("Saved processor; contents of %s: %s", out_save_dir, os.listdir(out_save_dir)[:20])

    num_pass = 0
    num_test = 0
    refs = []
    hyps = []
    for wav in glob.glob(os.path.join(extract_dir, '**', '*.wav'), recursive=True):
        fname = os.path.basename(wav)
        expected = re.sub(r"(?:_\d+)?\.wav$", "", fname)
        tmpdir = "tmp"
        os.makedirs(tmpdir, exist_ok=True)
        base = str(uuid.uuid4())
        norm_path = os.path.join(tmpdir, f"{base}_norm.wav")
        denoise_path = os.path.join(tmpdir, f"{base}_denoise.wav")
        if AudioSegment is None or wavfile is None or nr is None:
            logger.warning("Audio preprocessing libs not installed; skipping normalization/noise reduction and using original file.")
            denoise_path = wav
        else:
            normalize_audio_pydub(wav, norm_path)
            remove_noise(norm_path, denoise_path)

        pred = transcribe_wav2vec(denoise_path, processor_ref, model_ref, device)
        if run_postprocess:
            pred_pp = vietnamese_number_converter(pred)
        else:
            pred_pp = pred

        clean_pred = unidecode(pred_pp.lower().replace(' ', '_')) if unidecode else pred_pp.lower().replace(' ', '_')
        clean_exp  = unidecode(expected.lower().replace(' ', '_')) if unidecode else expected.lower().replace(' ', '_')
        ok = clean_exp in clean_pred
        status = 'PASS' if ok else 'FAIL'
        print(f"{status} | File: {fname} | Expected: {expected} | Got: {pred_pp}")
        if ok: num_pass += 1
        num_test += 1

        refs.append(expected.lower())
        hyps.append(pred_pp.lower())

    print('-'*30)
    if num_test == 0:
        print("❌ No .wav files found to evaluate.")
    else:
        print(f"Total pass: {num_pass}/{num_test} ~ {num_pass*100/num_test:.2f}%")
        try:
            if wer is None:
                logger.warning("jiwer not installed; cannot compute WER.")
            else:
                wer_score = wer(refs, hyps)
                print(f"JIwer WER (refs vs hyps): {wer_score:.4f}")
        except Exception as e:
            print("Could not compute jiwer WER:", e)


def main():
    parser = argparse.ArgumentParser(description='Evaluation script extracted from Colab.')
    parser.add_argument('--zip_path', type=str, default=ZIP_PATH_DEFAULT)
    parser.add_argument('--extract_dir', type=str, default=EXTRACT_DIR_DEFAULT)
    parser.add_argument('--local_weights', type=str, default=LOCAL_WEIGHTS_DEFAULT)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--out_save_dir', type=str, default=OUT_SAVE_DIR_DEFAULT)
    parser.add_argument('--run_postprocess', action='store_true')

    args = parser.parse_args()

    evaluate_folder(zip_path=args.zip_path, extract_dir=args.extract_dir, run_postprocess=args.run_postprocess, model_id=args.model_id, local_weights=args.local_weights, out_save_dir=args.out_save_dir)

if __name__ == '__main__':
    main()
