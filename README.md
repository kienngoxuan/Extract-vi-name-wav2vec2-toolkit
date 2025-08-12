# 🚀 vi-wav2vec2-toolkit

**Fine-tune & evaluate Wav2Vec2 models for Vietnamese Name ASR.** Lightweight scripts for training and safetensors-based evaluation with optional audio preprocessing. Minimal, ready-to-run CLI.

---

## 🔍 Quick TL;DR
- **Train**: `train_wav2vec2.py` — fine-tune from `metadata.csv` + `wavs/`  
- **Eval**: `eval_wav2vec2.py` — unzip `.wav` set, load `.safetensors`, transcribe, report PASS/FAIL + WER

---

## ⚙️ Highlights
- ✅ Clean CLI (Argparse) for train & eval  
- 🔁 Safe fallbacks (no `model_handling.py` → `AutoModelForCTC`)  
- 🎧 Optional audio preprocessing (pydub, noisereduce)  
- 🧾 WER reporting if `jiwer` installed  
- 🛠️ Attempts remapping keys when loading safetensors (`model.` / `module.`)

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
MIT — add `LICENSE` when pushing.
