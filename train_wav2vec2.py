# file: train_wav2vec2.py
"""
Training script (standalone).
This file was extracted from the original Colab->.py conversion.
Usage examples:
python train_wav2vec2.py --extracted_dir /content/data_train_70.6 --meta_csv /content/data_train_70.6/metadata.csv --model_id nguyenvulebinh/wav2vec2-large-vi-vlsp2020 --output_dir wav2vec2-finetuned --mode train

Note: install required packages first (transformers, datasets, huggingface_hub, torch, torchaudio, jiwer, safetensors as needed).
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Defaults
EXTRACTED_DIR = "/content/data_train_70.6"
META_CSV = os.path.join(EXTRACTED_DIR, 'metadata.csv')
MODEL_ID_DEFAULT = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"
OUTPUT_DIR_DEFAULT = "wav2vec2-finetuned"

# Optional HF login
def maybe_login(hf_token: str):
    if hf_token:
        try:
            login(hf_token)
            logger.info("Logged into Hugging Face Hub.")
        except Exception as e:
            logger.warning("Failed to login to HF Hub: %s", e)

# Dataset preparation
def prepare_datasets(extracted_dir: str = EXTRACTED_DIR, meta_csv: str = META_CSV, wav_dir: str = None):
    if wav_dir is None:
        wav_dir = os.path.join(extracted_dir, 'wavs')

    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"Metadata file not found: {meta_csv}")
    meta = pd.read_csv(meta_csv, sep='|', header=None, names=['filename','transcript'])
    pattern = r"[\,\?\.\!\-\;\:\"\‘\’\“\”\%\…]"
    meta['transcript'] = (
        meta['transcript']
            .str.lower()
            .str.replace(pattern, '', regex=True)
            .str.strip()
    )
    meta['audio_path'] = meta['filename'].apply(lambda f: os.path.join(wav_dir, f))

    hf_ds = Dataset.from_pandas(meta[['audio_path','transcript']])
    hf_ds = hf_ds.cast_column('audio_path', Audio(sampling_rate=16_000))

    dsplits   = hf_ds.train_test_split(test_size=0.1, seed=42)
    train_ds  = dsplits['train']
    eval_ds   = dsplits['test']
    return train_ds, eval_ds

# Model loader (downloads model_handling.py and loads)
def load_model_and_processor(model_id: str = MODEL_ID_DEFAULT, device: str = None, trust_remote_code: bool = True):
    # download the model_handling.py helper if present in the repo
    try:
        model_script = hf_hub_download(repo_id=model_id, filename="model_handling.py")
        model_loader = SourceFileLoader("model_handling", model_script).load_module()
    except Exception as e:
        logger.warning("Could not download model_handling.py from the hub: %s. Proceeding without it.", e)
        model_loader = None

    # instantiate processor + model
    processor = Wav2Vec2Processor.from_pretrained(model_id)

    if model_loader is not None:
        ModelClass = getattr(model_loader, 'Wav2Vec2ForCTC', None)
        if ModelClass is not None:
            model = ModelClass.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        else:
            # fallback
            from transformers import AutoModelForCTC
            model = AutoModelForCTC.from_pretrained(model_id)
    else:
        from transformers import AutoModelForCTC
        model = AutoModelForCTC.from_pretrained(model_id)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, device

# Data collator
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool,str] = True

    def __call__(self, features: List[Dict[str,Union[List[int],torch.Tensor]]]) -> Dict[str,torch.Tensor]:
        inputs = [{"input_values": f["input_values"]} for f in features]
        labels = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(inputs, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=labels, padding=self.padding, return_tensors="pt")
        # mask padding on labels
        if 'attention_mask' in labels_batch:
            batch_labels = labels_batch["input_ids"].masked_fill(labels_batch['attention_mask'].ne(1), -100)
        else:
            batch_labels = labels_batch["input_ids"]
        batch["labels"] = batch_labels
        return batch

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

# compute_metrics used by Trainer
def compute_metrics(pred, processor_ref):
    from jiwer import wer
    pred_ids = np.argmax(pred.predictions, axis=-1)
    label_ids = np.where(pred.label_ids != -100,
                         pred.label_ids,
                         processor_ref.tokenizer.pad_token_id)
    pred_str  = processor_ref.batch_decode(pred_ids, group_tokens=True, skip_special_tokens=True)
    label_str = processor_ref.batch_decode(label_ids, group_tokens=True, skip_special_tokens=True)
    wer_score = wer(label_str, pred_str)
    return {"wer": wer_score}

# preprocessing for a single example
def prepare_batch(batch, processor_ref):
    arr = batch['audio_path']['array']
    sr  = batch['audio_path']['sampling_rate']
    inp = processor_ref.feature_extractor(arr, sampling_rate=sr, return_tensors='np').input_values[0]
    lbl = processor_ref.tokenizer(batch['transcript']).input_ids
    return {'input_values': inp, 'labels': lbl}

# Run training
def run_training(extracted_dir: str = EXTRACTED_DIR,
                 meta_csv: str = META_CSV,
                 model_id: str = MODEL_ID_DEFAULT,
                 output_dir: str = OUTPUT_DIR_DEFAULT,
                 per_device_train_batch_size: int = 8,
                 gradient_accumulation_steps: int = 4,
                 learning_rate: float = 5e-5,
                 num_train_epochs: int = 8,
                 hf_token: str = None):

    train_ds, eval_ds = prepare_datasets(extracted_dir, meta_csv)

    model, processor, device = load_model_and_processor(model_id)

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    train_prepped = train_ds.map(lambda b: prepare_batch(b, processor), remove_columns=train_ds.column_names)
    eval_prepped  = eval_ds.map(lambda b: prepare_batch(b, processor), remove_columns=eval_ds.column_names)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_prepped,
        eval_dataset=eval_prepped,
        data_collator=data_collator,
        tokenizer=processor,
        compute_metrics=lambda pred: compute_metrics(pred, processor)
    )

    trainer.train()

    eval_results = trainer.evaluate()

    # make sure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # trainer.save_model will call model.save_pretrained(...) and also save the config
        trainer.save_model(output_dir)
        # processor contains tokenizer + feature_extractor
        processor.save_pretrained(output_dir)
        logger.info("Saved model and processor to %s", output_dir)
    except Exception as e:
        logger.exception("Failed to save model/processor with save_pretrained: %s", e)

    try:
        if save_safetensors is not None:
            # ensure CPU tensors
            sd = {k: v.cpu() for k, v in model.state_dict().items()}
            safetensors_path = os.path.join(output_dir, "model.safetensors")
            save_safetensors(sd, safetensors_path)
            logger.info("Saved safetensors to %s", safetensors_path)
        else:
            logger.warning("safetensors not installed; skipping model.safetensors save.")
    except Exception as e:
        logger.exception("Failed to write safetensors: %s", e)

    try:
        logger.info(f"Final eval WER: {eval_results['eval_wer']:.4f}")
    except Exception:
        logger.info("Evaluation finished; check trainer.evaluate() results.")



def main():
    parser = argparse.ArgumentParser(description='Training script extracted from Colab.')
    parser.add_argument('--extracted_dir', type=str, default=EXTRACTED_DIR)
    parser.add_argument('--meta_csv', type=str, default=META_CSV)
    parser.add_argument('--model_id', type=str, default=MODEL_ID_DEFAULT)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_train_epochs', type=int, default=8)

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
