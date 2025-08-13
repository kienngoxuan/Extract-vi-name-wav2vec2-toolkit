# file: train_wav2vec2.py
"""
Colab code
# Login to Hugging Face Hub to access models and datasets.
from huggingface_hub import login
login()
# Install the required libraries for the project.
!pip install -r requirements-vi.txt
!unzip /content/data_train_70.6.zip -d /content/
!unzip /content/person_name_500.zip -d /content/
"""

"""
Training script (standalone) with concise comments.
Usage example:
python train_wav2vec2.py --extracted_dir /content/data_train_70.6 \
    --meta_csv /content/data_train_70.6/metadata.csv \
    --model_id nguyenvulebinh/wav2vec2-large-vi-vlsp2020 \
    --output_dir wav2vec2-finetuned --mode train

Note: install required packages first (transformers, datasets, huggingface_hub, torch,
torchaudio, jiwer, safetensors as needed).
"""
import os
import re
import logging
import argparse
from dataclasses import dataclass
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import torch
from datasets import Audio, Dataset
from transformers import Wav2Vec2Processor, TrainingArguments, Trainer
try:
    from safetensors.torch import save_file as save_safetensors
except Exception:
    save_safetensors = None
from huggingface_hub import hf_hub_download, login
from importlib.machinery import SourceFileLoader


# configure root logger for simple output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Defaults used when CLI args are not provided
EXTRACTED_DIR = "/content/data_train_70.6"
META_CSV = os.path.join(EXTRACTED_DIR, 'metadata.csv')
MODEL_ID_DEFAULT = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"
OUTPUT_DIR_DEFAULT = "wav2vec2-finetuned"

# Optional HF login helper: call login() when token provided
def maybe_login(hf_token: str):
    if hf_token:
        try:
            login(hf_token)
            logger.info("Logged into Hugging Face Hub.")
        except Exception as e:
            logger.warning("Failed to login to HF Hub: %s", e)

# Prepare datasets from extracted archive + metadata
def prepare_datasets(extracted_dir: str = EXTRACTED_DIR, meta_csv: str = META_CSV, wav_dir: str = None):
    # default wav directory inside extracted_dir
    if wav_dir is None:
        wav_dir = os.path.join(extracted_dir, 'wavs')

    # ensure metadata CSV exists
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"Metadata file not found: {meta_csv}")

    # read metadata: two columns (filename|transcript)
    meta = pd.read_csv(meta_csv, sep='|', header=None, names=['filename','transcript'])

    # basic transcript cleaning: remove common punctuation and collapse whitespace
    pattern = r"[\,\?\.\!\-\;\:\"\‘\’\“\”\%\…]"
    meta['transcript'] = (
        meta['transcript']
            .str.lower()
            .str.replace(pattern, '', regex=True)
            .str.strip()
    )

    # attach full audio path for each filename
    meta['audio_path'] = meta['filename'].apply(lambda f: os.path.join(wav_dir, f))

    # create a Hugging Face dataset from the dataframe
    hf_ds = Dataset.from_pandas(meta[['audio_path','transcript']])

    # cast the audio path column to an Audio feature with sampling_rate 16k
    hf_ds = hf_ds.cast_column('audio_path', Audio(sampling_rate=16_000))

    # split into train/test (10% test)
    dsplits   = hf_ds.train_test_split(test_size=0.1, seed=42)
    train_ds  = dsplits['train']
    eval_ds   = dsplits['test']
    return train_ds, eval_ds

# Optional model loader: downloads model_handling.py helper if present
def load_model_and_processor(model_id: str = MODEL_ID_DEFAULT, device: str = None, trust_remote_code: bool = True):
    # try to fetch a custom model_handling.py from the hub repo
    try:
        model_script = hf_hub_download(repo_id=model_id, filename="model_handling.py")
        model_loader = SourceFileLoader("model_handling", model_script).load_module()
    except Exception as e:
        logger.warning("Could not download model_handling.py from the hub: %s. Proceeding without it.", e)
        model_loader = None

    # load the processor (feature extractor + tokenizer)
    processor = Wav2Vec2Processor.from_pretrained(model_id)

    # if the repo provided a local model class, prefer it (trust_remote_code used there)
    if model_loader is not None:
        ModelClass = getattr(model_loader, 'Wav2Vec2ForCTC', None)
        if ModelClass is not None:
            model = ModelClass.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        else:
            # fallback to generic AutoModelForCTC
            from transformers import AutoModelForCTC
            model = AutoModelForCTC.from_pretrained(model_id)
    else:
        from transformers import AutoModelForCTC
        model = AutoModelForCTC.from_pretrained(model_id)

    # move model to selected device (cuda if available)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, device

# Data collator that pads inputs and label sequences for CTC training
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate input arrays and label id lists
        inputs = [{"input_values": f["input_values"]} for f in features]
        labels = [f["labels"] for f in features]

        # Pad input_values using feature_extractor.pad -> returns tensors for model input
        batch = self.processor.feature_extractor.pad(inputs, padding=self.padding, return_tensors="pt")

        # Pad labels using tokenizer.pad, then replace pad token ids with -100 for CTC loss
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels}, padding=self.padding, return_tensors="pt"
        )["input_ids"]
        labels_batch = labels_batch.masked_fill(labels_batch == self.processor.tokenizer.pad_token_id, -100)

        batch["labels"] = labels_batch
        return batch


# Simple WER metric collector (not used directly by Trainer above, kept for completeness)
class WERMetric:
    def __init__(self):
        self._preds, self._refs = [], []
        self._punct = re.compile(r"[^\w\s]")
    def _norm(self, t): return self._punct.sub('', t.lower()).split()
    def add_batch(self, preds, refs):
        self._preds.extend(preds); self._refs.extend(refs)
    def compute(self):
        from jiwer import wer
        score = wer(self._refs, self._preds)
        return {"wer": score}

# compute_metrics function used by the Trainer to compute WER on eval
def compute_metrics(pred, processor_ref):
    from jiwer import wer
    # take argmax over logits to get token ids
    pred_ids = np.argmax(pred.predictions, axis=-1)
    # replace -100 with pad token id for decoding
    label_ids = np.where(pred.label_ids != -100,
                         pred.label_ids,
                         processor_ref.tokenizer.pad_token_id)
    # decode predictions and labels to strings
    pred_str  = processor_ref.batch_decode(pred_ids, group_tokens=True, skip_special_tokens=True)
    label_str = processor_ref.batch_decode(label_ids, group_tokens=True, skip_special_tokens=True)
    wer_score = wer(label_str, pred_str)
    return {"wer": wer_score}

# preprocessing for a single dataset example: extract features + tokenize labels
def prepare_batch(batch, processor_ref):
    # audio array and sampling rate from datasets' Audio feature
    arr = batch['audio_path']['array']
    sr = batch['audio_path']['sampling_rate']

    # run feature extractor (returns numpy input_values) and force float32
    inp = processor_ref.feature_extractor(arr, sampling_rate=sr, return_tensors='np').input_values[0]
    inp = np.asarray(inp, dtype=np.float32)

    # tokenize transcript into input ids for CTC labels
    lbl = processor_ref.tokenizer(batch['transcript']).input_ids
    return {'input_values': inp, 'labels': lbl}


# Run end-to-end training with sensible defaults
def run_training(extracted_dir: str = EXTRACTED_DIR,
                 meta_csv: str = META_CSV,
                 model_id: str = MODEL_ID_DEFAULT,
                 output_dir: str = OUTPUT_DIR_DEFAULT,
                 per_device_train_batch_size: int = 8,
                 gradient_accumulation_steps: int = 6,
                 learning_rate: float = 1e-5,
                 num_train_epochs: int = 7,
                 hf_token: str = None):

    # prepare train/eval HF datasets
    train_ds, eval_ds = prepare_datasets(extracted_dir, meta_csv)

    # load model+processor and pick device
    model, processor, device = load_model_and_processor(model_id)
        # --- ALIGN MODEL <-> TOKENIZER (ensure vocab sizes match) ---
    tokenizer_size = len(processor.tokenizer)
    model_vocab = getattr(model.config, "vocab_size", None)
    logger.info("Model vocab_size=%s   Tokenizer size=%s", model_vocab, tokenizer_size)

    if model_vocab != tokenizer_size:
        logger.info("Resizing model embeddings: %s -> %s", model_vocab, tokenizer_size)
        try:
            model.resize_token_embeddings(tokenizer_size)
        except Exception as e:
            logger.warning("resize_token_embeddings failed: %s", e)
        model.config.vocab_size = tokenizer_size

    # make sure pad token id is consistent
    if getattr(model.config, "pad_token_id", None) != processor.tokenizer.pad_token_id:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        logger.info("Set model.config.pad_token_id = %s", model.config.pad_token_id)


    # create collator instance for padding batches
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # map preprocessing function over datasets (removes original columns)
    train_prepped = train_ds.map(lambda b: prepare_batch(b, processor), remove_columns=train_ds.column_names)
    eval_prepped  = eval_ds.map(lambda b: prepare_batch(b, processor), remove_columns=eval_ds.column_names)

    # training hyperparameters and strategies
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        weight_decay=0.005,
        num_train_epochs=num_train_epochs,
        eval_steps=500,
        save_steps=500,
        logging_steps=200,
        eval_strategy="steps",
        save_total_limit=2,
        fp16=True if torch.cuda.is_available() else False,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    # instantiate Trainer with datasets and compute_metrics wrapper
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_prepped,
        eval_dataset=eval_prepped,
        data_collator=data_collator,
        tokenizer=processor,
        compute_metrics=lambda pred: compute_metrics(pred, processor)
    )

    # start training (blocking call)
    trainer.train()

    # evaluate after training
    eval_results = trainer.evaluate()

    # ensure output directory exists and save model + processor
    os.makedirs(output_dir, exist_ok=True)

    try:
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        logger.info("Saved model and processor to %s", output_dir)
    except Exception as e:
        logger.exception("Failed to save model/processor with save_pretrained: %s", e)

    # optionally save safetensors if available for smaller/binary checkpoint
    try:
        if save_safetensors is not None:
            sd = {k: v.cpu() for k, v in model.state_dict().items()}
            safetensors_path = os.path.join(output_dir, "model.safetensors")
            save_safetensors(sd, safetensors_path)
            logger.info("Saved safetensors to %s", safetensors_path)
        else:
            logger.warning("safetensors not installed; skipping model.safetensors save.")
    except Exception as e:
        logger.exception("Failed to write safetensors: %s", e)

    # log final eval WER if available
    try:
        logger.info(f"Final eval WER: {eval_results['eval_wer']:.4f}")
    except Exception:
        logger.info("Evaluation finished; check trainer.evaluate() results.")


def main():
    # Argument parser: include a short note about VRAM usage for the default config
    parser = argparse.ArgumentParser(description='Training script extracted from Colab.')

    # NOTE FOR USERS: default setup (batch 8 with gradient_accumulation_steps 4) may consume ~13.5GB VRAM on T4 GPU.
    # Reduce per_device_train_batch_size or gradient_accumulation_steps if you run out of memory.

    parser.add_argument('--extracted_dir', type=str, default=EXTRACTED_DIR,
                        help='Directory containing extracted dataset; expected to include a "wavs/" subfolder with audio files.')

    parser.add_argument('--meta_csv', type=str, default=META_CSV,
                        help='Path to the metadata CSV (format: filename|transcript) used to build the HF Dataset.')

    parser.add_argument('--model_id', type=str, default=MODEL_ID_DEFAULT,
                        help='Hugging Face model repo id or local path for the pretrained Wav2Vec2 model/processor to load (e.g. "username/model-name").')

    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR_DEFAULT,
                        help='Directory where the fine-tuned model, processor, and optional safetensors will be saved.')

    parser.add_argument('--hf_token', type=str, default=None,
                        help='(Optional) Hugging Face access token used to login and interact with the Hub (push/pull private repos).')

    parser.add_argument('--per_device_train_batch_size', type=int, default=8,
                        help='Per-device training batch size. Reduce this if you encounter out-of-memory (OOM) errors on your GPU.')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=6,
                        help='Number of steps to accumulate gradients before updating model weights (useful to simulate larger effective batch sizes).')

    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Initial learning rate for the optimizer.')

    parser.add_argument('--num_train_epochs', type=int, default=7,
                        help='Total number of training epochs to run.')

    args = parser.parse_args()
    maybe_login(args.hf_token)

    run_training(
        extracted_dir=args.extracted_dir,
        meta_csv=args.meta_csv,
        model_id=args.model_id,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        hf_token=args.hf_token
    )

if __name__ == '__main__':
    main()

#python /content/train_wav2vec2.py
