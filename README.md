# 🚀 Extract-vi-name-wav2vec2-toolkit

**Fine-tune & evaluate Wav2Vec2 models for Vietnamese Name ASR.** Lightweight scripts for training and safetensors-based evaluation with optional audio preprocessing. Minimal, ready-to-run CLI.

---

<a target="_blank" href="https://colab.research.google.com/drive/15Om8zqpJJC3XnxYcogJIYS7dqj1rQXs_?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## 🔍 Quick TL;DR
- **Train**: `train_wav2vec2.py` — fine-tune from `metadata.csv` + `wavs/`  
- **Eval**: `eval_wav2vec2.py` — unzip `.wav` set, load `.safetensors`, transcribe, report PASS/FAIL + WER

---

## ⚙️ Highlights
- ✅ Clean CLI (Argparse) for train & eval  
- 🔁 Safe fallbacks (no `model_handling.py` → `AutoModelForCTC`)  
- 🎷 Optional audio preprocessing (pydub, noisereduce)  
- 🧓‍📋 WER reporting if `jiwer` installed  
- 🛠️ Attempts remapping keys when loading safetensors (`model.` / `module.`)

---

## 🧬 Name extraction helpers
- `normalize_audio_pydub(input_file, output_file, target_level=-24)` — normalize audio level and add short padding so recordings are amplitude-consistent before processing.  
- `remove_noise(input_file, output_file)` — run a noise-reduction pass (via `noisereduce`) and write a cleaned WAV for more robust ASR.  
- `transcribe_wav2vec(audio_path, processor_ref, model_ref, device)` — load audio with `librosa`, run the model, and return the decoded transcription string.  
- `vietnamese_number_converter(text)` — post-process spoken Vietnamese number-words (e.g., "một hai ba") into numeric digit sequences when helpful for name/ID matching.  
- `evaluate_folder(zip_path, extract_dir, ...)` — orchestrates unzip → optional normalize/denoise → transcription, prints a simple per-file "dialogue" (PASS/FAIL lines), and **extracts the expected name from each filename** (by stripping `_###.wav`) and compares it against the predicted transcript to decide PASS/FAIL.

---

## ▶️ Quick start (short)
```bash
# install core deps (add torch/torchaudio per your CUDA)
pip install -r requirements-vi.txt
sudo apt-get install -y ffmpeg
```

Train:
```bash
python train_wav2vec2.py \
  --extracted_dir /content/data_train_70.6 \
  --meta_csv /content/data_train_70.6/metadata.csv \
  --model_id <HF_REPO_ID> \
  --output_dir wav2vec2-finetuned
```

Eval:
```bash
python eval_wav2vec2.py \
  --zip_path /content/person_name_500.zip \
  --extract_dir /content/person_name_500 \
  --local_weights /content/model.safetensors \
  --model_id <HF_REPO_ID> \
  --out_save_dir /content/my_wav2vec2_large \
  --run_postprocess
```

---

## 📦 Requirements file
See **requirements-vi.txt**:
```text
datasets<4.0.0
transformers>=4.31.0
torchaudio
jiwer
accelerate
pyctcdecode>=0.5.0
git+https://github.com/kpu/kenlm.git
peft
bitsandbytes
safetensors
unidecode
noisereduce
```

---

## 📝 Tips
- Use `--hf_token` if model repo is private.  
- If running in Colab, call with `!python ...` (avoid kernel args).  
- For best results, install `noisereduce`, `pydub`, `librosa`, and `jiwer`.  
- If using the Trainer with W&B integration, make sure your **WANDB_API_KEY** is set in the environment before running training.

---

## 🏷️ Tags
`wav2vec2` `asr` `vietnamese` `huggingface` `pytorch`

---

## 📄 License
MIT


---

## 📊 Short Evaluation Report
- **MODEL_ID** = `nguyenvulebinh/wav2vec2-base-vi-vlsp2020`: 218/500 PASS (~43.60%)
- **MODEL_ID** = `nguyenvulebinh/wav2vec2-large-vi-vlsp2020`: 322/500 PASS (~64.40%)
- **Fine-tuned with given Trainer params**: 348/500 PASS (~69.60%)
- Higher PASS rate indicates better name recognition accuracy in the evaluation set.
- Results may vary with different preprocessing, training epochs, and learning rates.
