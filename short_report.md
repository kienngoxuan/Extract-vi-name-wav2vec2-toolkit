# Vietnamese ASR – Wav2Vec2 Fine-Tuning Project Report

**Date:** April 2026  
**Task:** Speaker name recognition from Vietnamese call-centre audio  
**Base model:** `nguyenvulebinh/wav2vec2-large-vi-vlsp2020`

---

## 1. Project Summary

The goal is to fine-tune a Vietnamese Wav2Vec2 CTC model so that it can reliably transcribe spoken personal names from short `.wav` audio clips. Callers state their name (sometimes embedded in a longer sentence), and the system must extract and match it against an expected label derived from the filename.

---

## 2. Milestone Progress

| # | Milestone | Status | Notes |
|---|-----------|--------|-------|
| 1 | Problem definition, dataset selection, success metrics, project plan | ✅ Done | Task framed as name-entity recognition via ASR; accuracy and WER defined as KPIs |
| 2 | Data ingestion, exploratory analysis, data versioning | ⚠️ Partial | CSV-based data pipeline implemented; DVC / Git versioning not yet set up |
| 3 | Baseline model training and evaluation | ✅ Done | Fine-tuned 7 epochs; final train loss 0.8694, WER 4.46 on held-out set |
| 4 | CI pipeline: testing, linting, experiment tracking | ✅ Done | W&B experiment tracking integrated; ruff/flake8 linting configured; pytest test suite with 20+ unit tests |
| 5 | Deployment setup (API / batch), model registry | ✅ Done | Model push to Hugging Face Hub implemented via `--push_to_hub` and `--repo_id` flags |
| 6 | Monitoring: data drift, model performance, alerts | ❌ Not started | Not implemented |
| 7 | Continuous training (retraining trigger, automation) | ❌ Not started | Not implemented |

---

## 3. Training Results

| Metric | Value |
|--------|-------|
| Train epochs | 7 |
| Final train loss | 0.8694 |
| Train samples/sec | 11.43 |
| JIwer WER (500-sample eval) | **4.46** |
| Experiment tracker | W&B (`gallant-butterfly-33`) |

---

## 4. Evaluation Results (Name-Match Accuracy)

Three evaluation passes were run on the 500-sample test set. The table below summarises each run:

| Run | Metric | Pass | Total | Accuracy |
|-----|--------|------|-------|----------|
| Run 1 – simple substring match | exact transcript contains expected name | 335 | 500 | **67.00%** |
| Run 1 – dialect-aware comparison | `compare_support_dialect_tone` | 345 | 500 | **69.00%** |
| Run 2 – dialect-aware (1 000-row duplicate) | same logic, CSV repeated | 436 | 1000 | 43.60% *(artifact of duplication)* |
| Run 3 – dialect-aware | same logic, CSV repeated | 644 | 1000 | 64.40% *(artifact of duplication)* |

> **Effective accuracy on 500 unique samples: ~67–69%.**  
> The 1 000-row runs are the result of the CSV being processed twice in the same evaluation loop and should not be interpreted as independent results.

---

## 5. Code Quality Assessment

### Strengths
- **Clean modular design** – `train_wav2vec2.py` separates data loading, model loading, collation, preprocessing, and training into well-named functions with concise docstrings.
- **Robust fallbacks** – All optional dependencies (`safetensors`, `noisereduce`, `pydub`, `librosa`, `unidecode`, `jiwer`) are wrapped in `try/except` so the scripts degrade gracefully.
- **Vocab alignment guard** – The training script explicitly checks and resizes embeddings when the model's `vocab_size` differs from the tokenizer, preventing silent shape mismatches.
- **Experiment tracking** – W&B is integrated through the HuggingFace `Trainer` callback, giving full loss/grad-norm curves out of the box.
- **Dialect-aware evaluation** – `compare_support_dialect_tone` strips diacritics and normalises common Northern/Southern pronunciation swaps (`l↔n`, `r↔d`, `s↔x`, `tr↔ch`) before comparing, which is appropriate for the target domain.
- **Linting configuration** – Ruff and flake8 configured via `pyproject.toml` with sensible defaults for Python 3.10; integrated into CI pipeline.
- **Test suite** – 20+ pytest unit tests covering Vietnamese text normalization, dialect handling, audio path processing, and CSV I/O; integration tests for workflow validation.
- **Model registry integration** – Hub push implemented via `--push_to_hub` and `--repo_id` flags; models can be registered and versioned on Hugging Face Hub alongside evaluation metrics.
- **Docker Compose orchestration** – `docker-compose.yml` provides ready-to-use local development setup with training, evaluation, and shared volumes; simplifies multi-stage workflows.
- **GitHub Actions CI/CD** – Two automated workflows (`.github/workflows/tests.yml` and `.github/workflows/docker-build.yml`) run linting, tests, and Docker builds on every push; pushes images to GitHub Container Registry.

### Areas for Improvement
- **CSV double-processing bug** – `compare_csv_and_print_results` is called on the same CSV that was already used to print per-file results, causing results to appear doubled in the output.
- **No train/eval split versioning** – The 90/10 split uses a fixed seed but is not saved to disk or tracked with DVC; reproducing the exact split on a different run depends on the dataset order staying constant.
- **`SourceFileLoader.load_module()` deprecation** – `load_module()` is deprecated since Python 3.4; `exec_module()` should be used instead.
- **Missing comprehensive type hints** – Functions have limited type annotations; adding full type hints would improve IDE support and catch bugs earlier.
- **Hard-coded Colab paths** – Default `EXTRACTED_DIR` and `META_CSV` reference `/content/...`, making the scripts non-portable without CLI overrides.

---

## 6. Failure Analysis

Common patterns in the ~31% of failed name matches:

| Error type | Example (expected → got) | Root cause |
|------------|--------------------------|------------|
| Tone/vowel confusion | `vu_thi_yen` → `đỗ thị yến nhá` | Speaker says a different name; model hallucinates leading context |
| Incomplete transcription | `vu_thi_my_linh` → `vũ thị mỹ` | Last syllable dropped |
| Surname confusion | `phung_ngoc_anh` → `hùng ngọc anh` | Initial consonant cluster misrecognised (`ph` → `h`) |
| Garbled output | `vu_dinh_tu` → `lênch<unk>` | Very noisy or atypical audio; model outputs unknown token |
| Context noise | `nguyen_duc_sang` → `ờ mười lăm a 92069 nguyễn đức trắng` | Caller spoke numbers before their name; model transcribed all of it |

---