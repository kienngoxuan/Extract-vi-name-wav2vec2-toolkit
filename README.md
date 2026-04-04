# 🚀 Extract-vi-name-wav2vec2-toolkit

**Fine-tune & evaluate Wav2Vec2 models for Vietnamese Name ASR.** Lightweight scripts for training and safetensors-based evaluation with optional audio preprocessing. Minimal, ready-to-run CLI.

---

<a target="_blank" href="https://colab.research.google.com/drive/13h_CLJ0T_p4-YeQGQjZvaz7eykR36P9L?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

# 📂 Dataset Overview

## 🪟 **Data/Files Summary**

### **1. `data_train_70.6/`**

* **Total:** 2,804 entries

  * **2,801** `.wav` audio files
  * **1** `metadata.csv`
* **Size:** \~269.5 MB
* **Description:**
  This is a large ASR training dataset containing original and augmented speech samples.
  Augmentation variants include: `aug1`, `aug2`, `aug3`, `noise`, `pitch`, `speed`, and `echo`.
  The folder is ready for training with both audio files and aligned transcriptions.

#### Folder Structure:

```
data_train_70.6/
├── wavs/               # Original & augmented audio files
└── metadata.csv        # Audio-to-text mapping
```

#### **`metadata.csv` Example Entries:**

| Filename                                                                        | Transcript                                                          |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| original\_sample\_022422.wav                                                    | không chỉ vậy justin vẫn còn rất bảo vệ cameron                     |
| aug2\_speed\_pitch\_pham\_thi\_huong\_2.wav                                     | phạm thị hương.                                                     |
| aug2\_echo\_sample\_014388.wav                                                  | nhiều lần người thân bạn bè của vân góp ý nhưng cô đã gạt đi tất cả |
| original\_sample\_050912.wav                                                    | cháu linh bị tuấn chém vào mặt gây thương tích mười lăm phần trăm.  |
| aug2\_noise\_speed\_volume\_reverb\_FPTOpenSpeechData\_Set001\_V0.1\_001096.wav | bà trùm nguyễn thị ca.                                              |

---

### **2. `person_name_500/`**

* **Total:** 501 entries

  * **500** `.wav` audio files
* **Size:** \~16.5 MB
* **Description:**
  A small dataset of recorded **Vietnamese personal names**, possibly for speaker identification or name pronunciation training.

#### Example Filenames:

```
bui_duc_kha.wav
bui_duc_khoi.wav
bui_duc_trung.wav
bui_gia_kien.wav
bui_manh_ha.wav
```


---

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
- 🧪 **Pytest test suite** (20+ unit tests for Vietnamese text normalization, evaluation utils)  
- 🎨 **Ruff/flake8 linting** with `pyproject.toml` configuration  
- 🚀 **Hub push integration** — push fine-tuned models to Hugging Face with `--push_to_hub` flag

---

## 🧬 Name extraction helpers
- `normalize_audio_pydub(input_file, output_file, target_level=-24)` — normalize audio level and add short padding so recordings are amplitude-consistent before processing.  
- `remove_noise(input_file, output_file)` — run a noise-reduction pass (via `noisereduce`) and write a cleaned WAV for more robust ASR.  
- `transcribe_wav2vec(audio_path, processor_ref, model_ref, device)` — load audio with `librosa`, run the model, and return the decoded transcription string.  
- `vietnamese_number_converter(text)` — post-process spoken Vietnamese number-words (e.g., "một hai ba") into numeric digit sequences when helpful for name/ID matching.  
- `evaluate_folder(zip_path, extract_dir, ...)` — orchestrates unzip → optional normalize/denoise → transcription, prints a simple per-file "dialogue" (PASS/FAIL lines), and **extracts the expected name from each filename** (by stripping `_###.wav`) and compares it against the predicted transcript to decide PASS/FAIL.

---

## 🧪 Code Quality & Testing

### Linting
The codebase uses **ruff** and **flake8** for code quality checks. Configuration is in `pyproject.toml`.

```bash
# Run ruff checks
ruff check .

# Run flake8
flake8 train_wav2vec2.py eval_wav2vec2.py

# Auto-format with ruff
ruff format .
```

### Testing
A comprehensive **pytest** suite (20+ tests) covers Vietnamese text normalization, audio processing, and evaluation utilities.

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests
pytest tests/ -v -m unit

# Run with coverage report
pytest tests/ --cov=. --cov-report=html
```

Test files:
- `tests/test_preprocessing.py` — Vietnamese diacritics, number conversion, dialect normalization
- `tests/test_eval_utils.py` — CSV handling, transcription validation, metrics computation
- `tests/conftest.py` — Pytest fixtures for common test data

---

## ▶️ Quick start (short)
```bash
# install core deps (add torch/torchaudio per your CUDA)
pip install -r requirements-vi.txt
sudo apt-get install -y ffmpeg
```

Train:
```bash
python /content/train_wav2vec2.py
```

**Train with Hub push:**
```bash
python train_wav2vec2.py \
  --extracted_dir /content/data_train_70.6 \
  --meta_csv /content/data_train_70.6/metadata.csv \
  --output_dir /content/wav2vec2-finetuned \
  --push_to_hub \
  --repo_id your_username/your_repo_id \
  --hf_token YOUR_HF_TOKEN
```

Eval:
```bash
python eval_wav2vec2.py \
  --wav_dir /content/person_name_500/ \
  --model_dir /content/wav2vec2-finetuned \
  --local_weights /content/wav2vec2-finetuned/model.safetensors \
  --run_postprocess
```

---

## 📦 Requirements file
See **requirements-vi.txt**:
```text
datasets<4.0.0
transformers==4.31.0
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
- Use `--push_to_hub` and `--repo_id your_username/your_repo_id` to automatically push the fine-tuned model to Hugging Face Hub after training.
- If running in Colab, call with `!python ...` (avoid kernel args).  
- For best results, install `noisereduce`, `pydub`, `librosa`, and `jiwer`.  
- If using the Trainer with W&B integration, make sure your **WANDB_API_KEY** is set in the environment before running training.
- Before submitting code, run `ruff check .` and `pytest tests/ -v` to ensure code quality.

---

## 🐳 Docker & Docker Compose

### Docker (Manual Build)

Build the Docker image with linting and test stages:
```bash
docker build -t wav2vec2-vi:latest .
```

Run training in container:
```bash
docker run --rm \
  -v /path/to/data:/data \
  wav2vec2-vi:latest \
  python train_wav2vec2.py \
    --extracted_dir /data/data_train \
    --meta_csv /data/metadata.csv \
    --output_dir /data/wav2vec2-finetuned
```

### Docker Compose (Recommended for Local Development)

**Setup:**
```bash
# Create local directories for data and models
mkdir -p data/train data/eval models outputs
```

**Run the pipeline:**
```bash
# Build and start all services
docker-compose up --build

# View specific service logs
docker-compose logs -f wav2vec2-train

# Stop services
docker-compose down
```

**Set environment variables:**
```bash
export WANDB_API_KEY=your_wandb_key
export HF_TOKEN=your_huggingface_token
docker-compose up --build
```

**docker-compose.yml services:**
- `wav2vec2-train` — Training pipeline
- `wav2vec2-eval` — Evaluation (runs after training)
- Shared volumes: `data/`, `models/`, `outputs/`

---

## 🚀 GitHub Actions CI/CD

Two workflows automatically run on every push and pull request:

### Workflow 1: Tests & Linting (`.github/workflows/tests.yml`)
Runs on every push to `main`/`develop` and all pull requests:
- **Linting:** ruff & flake8 (Python 3.10)
- **Tests:** pytest on Python 3.9, 3.10, 3.11
- **Coverage:** Codecov reports
- **Quality Gate:** Blocks merge if tests fail

### Workflow 2: Docker Build & Push (`.github/workflows/docker-build.yml`)
Runs on push to `main`/`develop` and version tags (v*):
- **Test:** Linting and pytest
- **Build:** Docker image from Dockerfile
- **Push:** To GitHub Container Registry (ghcr.io)
- **Smoke Test:** Runs linting/tests inside built image

**Docker image tags:**
```
ghcr.io/<owner>/<repo>:main        # Latest from main branch
ghcr.io/<owner>/<repo>:develop     # Latest from develop branch
ghcr.io/<owner>/<repo>:v1.0.0      # Version tags
ghcr.io/<owner>/<repo>:<commit>    # Commit SHA
```

**To use in your repository:**
1. Push code to GitHub
2. Workflows run automatically
3. View progress in Actions tab
4. Pull Docker image: `docker pull ghcr.io/<owner>/<repo>:main`

---

## 🏷️ Tags
`wav2vec2` `asr` `vietnamese` `huggingface` `pytorch`

---

## 📄 License
MIT


---

## 📊 Short Evaluation Report
- **MODEL_ID** = `nguyenvulebinh/wav2vec2-base-vi-vlsp2020`: 436/1000 PASS (~43.60%)
- **MODEL_ID** = `nguyenvulebinh/wav2vec2-large-vi-vlsp2020`: 644/1000 PASS (~ 64.40%)
- **Fine-tuned with given Trainer params**: 345/500 PASS (~69.00%)
- Higher PASS rate indicates better name recognition accuracy in the evaluation set.
- Results may vary with different preprocessing, training epochs, and learning rates.
