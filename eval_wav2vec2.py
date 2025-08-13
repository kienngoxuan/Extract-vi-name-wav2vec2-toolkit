# file: eval_wav2vec2.py
"""
Evaluation script that loads model architecture from hub (model_id) AND local safetensors
(model.safetensors) to ensure tokenizer <-> weights match the architecture used during finetune.

This version accepts both --model_id and --model_dir (local folder). If --model_dir is provided
it will prefer the local processor/tokenizer, and it will still download model_handling.py from
--model_id to instantiate the exact model architecture before loading local weights.

Usage example:
python eval_wav2vec2.py \
  --wav_dir /content/person_name_500/ \
  --model_dir /content/wav2vec2-finetuned \
  --local_weights /content/wav2vec2-finetuned/model.safetensors \
  --run_postprocess
"""
import os
import re
import uuid
import glob
import logging
import argparse
import csv
from importlib.machinery import SourceFileLoader

import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2Processor

# optional libraries (graceful fallback)
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

DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID_DEFAULT = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"

# ------------------ Audio helpers ------------------
def normalize_audio_pydub(input_file, output_file, target_level=-24):
    if AudioSegment is None:
        raise RuntimeError("pydub not available")
    audio = AudioSegment.silent(duration=500) + AudioSegment.from_file(input_file) + AudioSegment.silent(duration=500)
    normalized = audio.apply_gain(target_level - audio.dBFS)
    normalized.export(output_file, format=output_file.split(".")[-1])

def remove_noise(input_file, output_file):
    if wavfile is None or nr is None:
        raise RuntimeError("scipy.io.wavfile or noisereduce not available")
    rate, data = wavfile.read(input_file)
    if data.dtype != np.float32:
        data = data.astype("float32") / 32768.0
    reduced = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(output_file, rate, (reduced * 32768).astype("int16"))

# ------------------ Vietnamese normalization helpers ------------------
def vietnamese_number_converter(text):
    number_mapping = {
        "không":"0","hông":"0","một":"1","mốt":"1","hai":"2","ba":"3","bốn":"4",
        "năm":"5","sáu":"6","bảy":"7","tám":"8","chín":"9"
    }
    if not text:
        return text
    words, result, i = text.split(), [], 0
    while i < len(words):
        w = "".join(c for c in words[i].lower() if c.isalpha())
        if w in number_mapping:
            seq, punct, j = [], "", i
            while j < len(words) and "".join(c for c in words[j].lower() if c.isalpha()) in number_mapping:
                punct_tmp = "".join(c for c in words[j] if not c.isalpha())
                punct = punct_tmp or punct
                seq.append("".join(c for c in words[j].lower() if c.isalpha()))
                j += 1
            if len(seq) > 2:
                result.append("".join(number_mapping[x] for x in seq) + punct)
                i = j
            else:
                result.append(words[i]); i += 1
        else:
            result.append(words[i]); i += 1
    return " ".join(result)

def convert_vietnamese_diacritics(text):
    char_map = {
        'à':'a','á':'a','ả':'a','ã':'a','ạ':'a',
        'ă':'a','ằ':'a','ắ':'a','ẳ':'a','ẵ':'a','ặ':'a',
        'â':'a','ầ':'a','ấ':'a','ẩ':'a','ẫ':'a','ậ':'a',
        'đ':'d',
        'è':'e','é':'e','ẻ':'e','ẽ':'e','ẹ':'e',
        'ê':'e','ề':'e','ế':'e','ể':'e','ễ':'e','ệ':'e',
        'ì':'i','í':'i','ỉ':'i','ĩ':'i','ị':'i',
        'ò':'o','ó':'o','ỏ':'o','õ':'o','ọ':'o',
        'ô':'o','ồ':'o','ố':'o','ổ':'o','ỗ':'o','ộ':'o',
        'ơ':'o','ờ':'o','ớ':'o','ở':'o','ỡ':'o','ợ':'o',
        'ù':'u','ú':'u','ủ':'u','ũ':'u','ụ':'u',
        'ư':'u','ừ':'u','ứ':'u','ử':'u','ữ':'u','ự':'u',
        'ỳ':'i','ý':'i','ỷ':'i','ỹ':'i','ỵ':'i','y':'i'
    }
    return "".join(char_map.get(c, c) for c in text)

def convert_vietnamese_number(text: str) -> str:
    char_map = {'0':'không','1':'một','2':'hai','3':'ba','4':'bốn','5':'năm','6':'sáu','7':'bảy','8':'tám','9':'chín','10':'mười'}
    return "".join(char_map.get(ch, ch) for ch in text)

def normalize_speech_patterns(text: str) -> str:
    if text.startswith("l"):
        text = "n" + text[1:]
    if text.startswith("r"):
        text = "d" + text[1:]
    if text.startswith("gi"):
        text = "d" + text[2:]
    if text.startswith("s"):
        text = "x" + text[1:]
    if text.startswith("tr"):
        text = "ch" + text[2:]
    return text

def compare_support_dialect_tone(s1: str, s2: str) -> bool:
    s1 = re.sub(r"[^\w\s]", "", s1.replace("_", " ")).strip().lower()
    s2 = re.sub(r"[^\w\s]", "", s2.replace("_", " ")).strip().lower()
    s1 = convert_vietnamese_number(s1)
    s2 = convert_vietnamese_number(s2)
    s1 = convert_vietnamese_diacritics(s1)
    s2 = convert_vietnamese_diacritics(s2)
    words1 = [normalize_speech_patterns(w) for w in s1.split()]
    words2 = [normalize_speech_patterns(w) for w in s2.split()]
    return " ".join(words2) in " ".join(words1)

# ------------------ Transcription ------------------
def transcribe_wav2vec(audio_path, processor_ref, model_ref, device):
    if librosa is None:
        raise RuntimeError("librosa is required for audio loading")
    audio_arr, sr = librosa.load(audio_path, sr=16000)
    inputs = processor_ref(audio_arr, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model_ref(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor_ref.decode(pred_ids[0]).strip()

# ------------------ Checkpoint loading helper ------------------
def try_load_checkpoint_into_model(model, checkpoint_path):
    loaded = False
    # safetensors
    if checkpoint_path.endswith(".safetensors") and load_safetensors is not None:
        try:
            sd = load_safetensors(checkpoint_path)
            sd_torch = {k: torch.as_tensor(v).cpu() for k, v in sd.items()}
            if "model" in sd_torch and isinstance(sd_torch["model"], dict):
                sd_torch = sd_torch["model"]
            try:
                model.load_state_dict(sd_torch, strict=False)
                logger.info("Loaded safetensors (strict=False).")
                return True
            except Exception:
                # remap keys and retry
                remapped = { (k[len("model."): ] if k.startswith("model.") else (k[len("module."): ] if k.startswith("module.") else k)): v for k,v in sd_torch.items() }
                model.load_state_dict(remapped, strict=False)
                logger.info("Loaded safetensors after remap (strict=False).")
                return True
        except Exception as e:
            logger.warning("Failed reading safetensors: %s", e)

    # torch checkpoint fallback
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return False

    candidates = []
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            candidates.append(ckpt["state_dict"])
        if "model" in ckpt:
            candidates.append(ckpt["model"])
        candidates.append(ckpt)
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        cand_torch = {}
        for k, v in cand.items():
            if isinstance(v, torch.Tensor):
                cand_torch[k] = v
            else:
                try:
                    cand_torch[k] = torch.as_tensor(v)
                except Exception:
                    pass
        try:
            model.load_state_dict(cand_torch, strict=False)
            logger.info("Loaded torch checkpoint candidate (strict=False).")
            return True
        except Exception:
            remapped = {}
            for k, v in cand_torch.items():
                newk = k
                if newk.startswith("model."):
                    newk = newk[len("model."):]
                if newk.startswith("module."):
                    newk = newk[len("module."):]
                remapped[newk] = v
            try:
                model.load_state_dict(remapped, strict=False)
                logger.info("Loaded torch checkpoint after remapping (strict=False).")
                return True
            except Exception:
                continue
    return False

# ------------------ Main evaluation flow ------------------
def evaluate_folder(wav_dir, model_id=None, model_dir=None, local_weights=None, out_save_dir=None, run_postprocess=False, device=DEVICE_DEFAULT):
    if not wav_dir or not os.path.exists(wav_dir):
        logger.error("wav_dir not found: %s", wav_dir)
        return

    model_id = model_id or MODEL_ID_DEFAULT
    out_save_dir = out_save_dir or model_dir or (os.path.dirname(local_weights) if local_weights else "./out_eval")
    os.makedirs(out_save_dir, exist_ok=True)

    # try download model_handling.py to get exact architecture (optional)
    model_loader = None
    try:
        model_script = hf_hub_download(repo_id=model_id, filename="model_handling.py")
        model_loader = SourceFileLoader("model_handling", model_script).load_module()
        logger.info("Downloaded model_handling.py from %s", model_id)
    except Exception as e:
        logger.warning("Could not download model_handling.py: %s. Will fallback to AutoModelForCTC.", e)

    # load processor (prefer local)
    try:
        if model_dir and os.path.isdir(model_dir):
            logger.info("Loading processor from local model_dir: %s", model_dir)
            processor = Wav2Vec2Processor.from_pretrained(model_dir)
        else:
            logger.info("Loading processor from hub model_id: %s", model_id)
            processor = Wav2Vec2Processor.from_pretrained(model_id)
    except Exception as e:
        logger.error("Failed to load processor: %s", e)
        raise

    # instantiate model (use remote code if provided)
    if model_loader is not None and hasattr(model_loader, "Wav2Vec2ForCTC"):
        logger.info("Instantiating custom Wav2Vec2ForCTC from model_handling.py")
        ModelClass = model_loader.Wav2Vec2ForCTC
        model = ModelClass.from_pretrained(model_id, trust_remote_code=True)
    else:
        logger.info("Falling back to AutoModelForCTC.from_pretrained(model_id)")
        from transformers import AutoModelForCTC
        model = AutoModelForCTC.from_pretrained(model_id)

    model.eval()

    # attempt to load local weights into the model
    loaded = False
    if local_weights and os.path.exists(local_weights):
        logger.info("Attempting to load local weights from %s", local_weights)
        try:
            loaded = try_load_checkpoint_into_model(model, local_weights)
        except Exception as e:
            logger.warning("Error while loading local weights: %s", e)
        if not loaded:
            logger.warning("Could not load local_weights fully. Continuing with hub weights (may be unfine-tuned).")
    else:
        logger.info("No local_weights provided or not found: %s", local_weights)

    # move model to device and save snapshot
    model.to(device)
    model.eval()
    try:
        model.save_pretrained(out_save_dir)
    except Exception as e:
        logger.warning("model.save_pretrained() failed: %s; saving state_dict instead.", e)
        torch.save(model.state_dict(), os.path.join(out_save_dir, "pytorch_model.bin"))
    processor.save_pretrained(out_save_dir)
    logger.info("Saved processor and model state to %s", out_save_dir)

    # Evaluate WAV files
    csv_out = os.path.join(out_save_dir, "transcription_results_wav2vec2.csv")
    with open(csv_out, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["path_wav", "expected_name", "transcription"])
        num_pass = num_test = 0
        refs = []; hyps = []

        for wav in glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True):
            fname = os.path.basename(wav)
            expected = re.sub(r"(?:_\d+)?\.wav$", "", fname)
            tmpdir = "tmp"
            os.makedirs(tmpdir, exist_ok=True)
            base = str(uuid.uuid4())
            norm_path = os.path.join(tmpdir, f"{base}_norm.wav")
            denoise_path = wav

            if AudioSegment is not None and wavfile is not None and nr is not None:
                try:
                    normalize_audio_pydub(wav, norm_path)
                    remove_noise(norm_path, norm_path)  # overwrite norm_path or produce cleaned file
                    denoise_path = norm_path
                except Exception as e:
                    logger.warning("Preprocessing failed for %s: %s; using original file.", wav, e)
                    denoise_path = wav

            try:
                pred = transcribe_wav2vec(denoise_path, processor, model, device)
            except Exception as e:
                logger.exception("Transcription failed for %s: %s", denoise_path, e)
                pred = ""

            pred_pp = vietnamese_number_converter(pred) if run_postprocess else pred
            writer.writerow([wav, expected, pred_pp])
            csv_file.flush()

            clean_pred = (unidecode(pred_pp.lower().replace(" ", "_")) if unidecode else pred_pp.lower().replace(" ", "_"))
            clean_exp = (unidecode(expected.lower().replace(" ", "_")) if unidecode else expected.lower().replace(" ", "_"))
            ok = clean_exp in clean_pred
            status = "PASS" if ok else "FAIL"
            print(f"{status} | File: {fname} | Expected: {expected} | Got: {pred_pp}")
            if ok:
                num_pass += 1
            num_test += 1

            refs.append(expected.lower())
            hyps.append(pred_pp.lower())

    print("-" * 30)
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

    # run compare-name logic on saved CSV
    compare_csv_and_print_results(csv_out)

def compare_csv_and_print_results(file_csv: str):
    print("\nRunning compare-name logic on:", file_csv)
    num_pass = num_fail = total = 0
    with open(file_csv, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or len(row) < 3:
                continue
            audio_file, expected_name, model_transcription = row[0], row[1], row[2]
            result = compare_support_dialect_tone(model_transcription, expected_name)
            if not result:
                print(f"FAIL | audio_file: {audio_file} expected_name: {expected_name} model_transcription: {model_transcription}")
                num_fail += 1
            else:
                num_pass += 1
            total += 1
    print("\nCompare-name summary:")
    print("Số lượng pass: ", num_pass)
    print("Số lượng fail: ", num_fail)
    if total > 0:
        print("Tỉ lệ đúng:", f"{(num_pass*100/total):.2f}%")
    else:
        print("No rows in CSV to compare.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Eval wav2vec2 loading hub architecture + local safetensors")
    p.add_argument("--wav_dir", type=str, required=True, help="Directory with .wav files (recursive)")
    p.add_argument("--model_id", type=str, default=f"{MODEL_ID_DEFAULT}", help="Hub repo id (used to get model_handling.py and architecture)")
    p.add_argument("--model_dir", type=str, default=None, help="Local model folder (containing tokenizer/config); preferred for processor files")
    p.add_argument("--local_weights", type=str, default=None, help="Path to local safetensors or torch checkpoint (optional)")
    p.add_argument("--out_save_dir", type=str, default=None, help="Where to save processor + CSV (defaults to model_dir or local_weights dir)")
    p.add_argument("--run_postprocess", action="store_true", help="Apply vietnamese number postprocessing")
    p.add_argument("--device", type=str, default=DEVICE_DEFAULT)
    args = p.parse_args()

    evaluate_folder(
        wav_dir=args.wav_dir,
        model_id=args.model_id,
        model_dir=args.model_dir,
        local_weights=args.local_weights,
        out_save_dir=args.out_save_dir,
        run_postprocess=args.run_postprocess,
        device=args.device
    )
    
    """
    # Now run the evaluation with the exact command you wanted
    python eval_wav2vec2.py \
    --wav_dir /content/person_name_500/ \
    --model_dir /content/wav2vec2-finetuned \
    --local_weights /content/wav2vec2-finetuned/model.safetensors \
    --run_postprocess
    """
