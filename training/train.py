#!/usr/bin/env python3

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import torch.nn as nn
import numpy as np
import gc
gc.collect()
torch.cuda.empty_cache()

import json
import argparse
import time
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional

from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoProcessor,
)
from transformers.trainer_callback import TrainerCallback
from model.salient_q_model import create_salient_q_v2_model


class SalientQNPZDataset(torch.utils.data.Dataset):

    def __init__(self, npz_dir, max_samples):
        self.npz_dir = npz_dir
        all_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
        if max_samples is not None:
            all_files = all_files[:max_samples]
        self.files = all_files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = os.path.join(self.npz_dir, self.files[idx])
        try:
            data = np.load(npz_path)
            return {
                "input_ids": torch.tensor(data['input_ids'], dtype=torch.long),
                "attention_mask": torch.tensor(data['attention_mask'], dtype=torch.long),
                "labels": torch.tensor(data['labels'], dtype=torch.long),
                "pixel_values": torch.tensor(data['pixel_values'], dtype=torch.float32),
                "image_grid_thw": torch.tensor(data['image_grid_thw'], dtype=torch.long),
                "saliency_masks": torch.tensor(data['saliency_masks'], dtype=torch.float32),
                "detection_boxes": torch.tensor(data['detection_boxes'], dtype=torch.float32),
                "detection_labels": torch.tensor(data['detection_labels'], dtype=torch.long),
            }
        except Exception as e:
            return self._get_dummy_example()

    def _get_dummy_example(self) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.zeros(8192, dtype=torch.long),
            "attention_mask": torch.zeros(8192, dtype=torch.long),
            "labels": torch.full((8192,), -100, dtype=torch.long),
            "pixel_values": torch.zeros(256, 3, 14, 14, dtype=torch.float32),
            "image_grid_thw": torch.tensor([1, 16, 16], dtype=torch.long),
            "saliency_masks": torch.zeros(64, dtype=torch.float32),
            "detection_boxes": torch.zeros(10, 4, dtype=torch.float32),
            "detection_labels": torch.zeros(10, dtype=torch.long),
        }


class SaveSalientQModulesCallback(TrainerCallback):

    def __init__(self, num_scouts):
        self.num_scouts = num_scouts

    def _get_salient_modules(self, model):
        unwrapped = model
        for _ in range(5):
            if hasattr(unwrapped, 'saliency_module'):
                return unwrapped
            if hasattr(unwrapped, 'base_model'):
                unwrapped = unwrapped.base_model
            elif hasattr(unwrapped, 'model'):
                unwrapped = unwrapped.model
            else:
                break
        return None

    def on_save(self, args, state, control, model, **kwargs):
        if model is None:
            return

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(checkpoint_dir):
            checkpoint_dir = args.output_dir

        unwrapped = self._get_salient_modules(model)
        if unwrapped is not None and hasattr(unwrapped, 'saliency_module'):
            salient_q_modules = {
                'saliency_module': unwrapped.saliency_module.state_dict(),
                'scout_module': unwrapped.scout_module.state_dict(),
                'num_scouts': self.num_scouts,
                'feature_dim': getattr(unwrapped, 'feature_dim', 4096),
                'step': state.global_step,
            }
            modules_path = os.path.join(checkpoint_dir, "salient_q_modules.pt")
            torch.save(salient_q_modules, modules_path)
            print(f"  Saved Salient-Q modules to: {modules_path}")
        else:
            print(f"  Could not find Salient-Q modules to save at step {state.global_step}")


class MonitoringCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0 and state.log_history:
            elapsed = time.time() - self.start_time
            steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
            eta_seconds = (state.max_steps - state.global_step) / steps_per_sec if steps_per_sec > 0 else 0
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))

            latest = state.log_history[-1]
            loss = latest.get('loss', 'N/A')
            grad_norm = latest.get('grad_norm', 'N/A')
            lr = latest.get('learning_rate', 'N/A')

            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
            grad_str = f"{grad_norm:.4f}" if isinstance(grad_norm, (int, float)) else str(grad_norm)
            lr_str = f"{lr:.2e}" if isinstance(lr, (int, float)) else str(lr)

            mem_str = "N/A"
            if torch.cuda.is_available():
                mem_str = f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}GB"

            print(
                f"Step {state.global_step:>5}/{state.max_steps} | "
                f"Loss: {loss_str} | Grad: {grad_str} | "
                f"LR: {lr_str} | Mem: {mem_str} | ETA: {eta_str}"
            )

    def on_save(self, args, state, control, **kwargs):
        print(f"\nCheckpoint saved at step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        print(f"\nTraining completed in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
        print(f"Total steps: {state.global_step}")
        if state.log_history:
            print(f"Final loss: {state.log_history[-1].get('loss', 'N/A')}")


def salient_q_collate_fn(batch):
    batch_size = len(batch)

    input_ids = torch.stack([ex["input_ids"] for ex in batch])
    attention_mask = torch.stack([ex["attention_mask"] for ex in batch])
    labels = torch.stack([ex["labels"] for ex in batch])

    all_pixel_values = []
    all_image_grid_thw = []

    for ex in batch:
        pv = ex['pixel_values']
        igt = ex['image_grid_thw']
        if pv.dim() == 3:
            pv = pv.unsqueeze(0)
        if igt.dim() == 1:
            igt = igt.unsqueeze(0)
        all_pixel_values.append(pv)
        all_image_grid_thw.append(igt)

    try:
        pixel_values = torch.cat(all_pixel_values, dim=0)
        image_grid_thw = torch.cat(all_image_grid_thw, dim=0)
    except Exception as e:
        warnings.warn(f"Failed to concat visual inputs: {e}")
        pixel_values = torch.zeros(batch_size * 256, 3, 14, 14)
        image_grid_thw = torch.ones(batch_size, 3, dtype=torch.long) * 16

    all_saliency_masks = [ex['saliency_masks'].view(-1) for ex in batch]
    try:
        saliency_masks = torch.cat(all_saliency_masks, dim=0)
    except Exception:
        saliency_masks = torch.zeros(pixel_values.shape[0] // 4)

    detection_boxes = torch.stack([ex['detection_boxes'] for ex in batch])
    detection_labels = torch.stack([ex['detection_labels'] for ex in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "saliency_masks": saliency_masks,
        "detection_boxes": detection_boxes,
        "detection_labels": detection_labels,
    }


class SalientQTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch):
        model_inputs = {
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
            "labels": inputs.get("labels"),
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "saliency_masks": inputs.get("saliency_masks"),
            "detection_boxes": inputs.get("detection_boxes"),
            "detection_labels": inputs.get("detection_labels"),
        }
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
        outputs = model(**model_inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def main():
    experiment_date = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description="Train Salient-Q model")

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--npz_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--num_scouts", type=int, default=16)

    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=10)

    parser.add_argument("--deepspeed", type=str, default="configs/ds_zero2_no_offload.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else AutoTokenizer.from_pretrained(args.base_model)

    model = create_salient_q_v2_model(
        base_model_path=args.base_model,
        num_scouts=args.num_scouts,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    model.enable_input_require_grads()
    model.config.use_cache = False

    train_dataset = SalientQNPZDataset(args.npz_dir, max_samples=args.max_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    num_devices = max(1, torch.cuda.device_count())
    effective_batch = args.batch_size * args.gradient_accumulation_steps * num_devices
    steps_per_epoch = len(train_dataset) // effective_batch
    total_steps = steps_per_epoch * args.epochs

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        weight_decay=0.01,
        logging_dir=log_dir,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to="tensorboard",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",
        gradient_checkpointing=True,
        bf16=True,
        bf16_full_eval=True,
        deepspeed=args.deepspeed if os.path.exists(args.deepspeed) else None,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        group_by_length=False,
    )

    trainer = SalientQTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=salient_q_collate_fn,
        tokenizer=tokenizer,
        callbacks=[MonitoringCallback(), SaveSalientQModulesCallback(num_scouts=args.num_scouts)],
    )

    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except KeyboardInterrupt:
        trainer.save_model(os.path.join(args.output_dir, "emergency_checkpoint"))
        raise
    trainer.save_model(args.output_dir)
    trainer.save_state()

    unwrapped = model
    for _ in range(5):
        if hasattr(unwrapped, 'saliency_module'):
            break
        if hasattr(unwrapped, 'base_model'):
            unwrapped = unwrapped.base_model
        elif hasattr(unwrapped, 'model'):
            unwrapped = unwrapped.model
        else:
            break

    if hasattr(unwrapped, 'saliency_module') and hasattr(unwrapped, 'scout_module'):
        salient_q_modules = {
            'saliency_module': unwrapped.saliency_module.state_dict(),
            'scout_module': unwrapped.scout_module.state_dict(),
            'num_scouts': args.num_scouts,
            'feature_dim': getattr(unwrapped, 'feature_dim', 4096),
        }
        modules_path = os.path.join(args.output_dir, "salient_q_modules.pt")
        torch.save(salient_q_modules, modules_path)

        verify = torch.load(modules_path, map_location='cpu')
        saliency_params = sum(v.numel() for v in verify['saliency_module'].values())
        scout_params = sum(v.numel() for v in verify['scout_module'].values())

        print(f"Saved Salient-Q modules to: {modules_path}")
        print(f"  Saliency module: {saliency_params:,} parameters")
        print(f"  Scout module: {scout_params:,} parameters")
    else:
        print("ERROR: Could not find Salient-Q modules!")
        fallback_path = os.path.join(args.output_dir, "full_model_fallback.pt")
        torch.save(model.state_dict(), fallback_path)
        print(f"  Saved full model as fallback: {fallback_path}")

    salient_q_config = {
        'base_model': args.base_model,
        'num_scouts': args.num_scouts,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'loss_weights': {
            'language_modeling': 1.0,
            'saliency': 0.5,
            'detection': 0.3,
        },
        'training': {
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'gradient_accumulation': args.gradient_accumulation_steps,
            'num_examples': len(train_dataset),
        },
        'files': {
            'lora_adapter': 'adapter_model.safetensors',
            'salient_q_modules': 'salient_q_modules.pt',
        },
        'timestamp': datetime.now().isoformat(),
    }

    config_path = os.path.join(args.output_dir, "salient_q_config.json")
    with open(config_path, 'w') as f:
        json.dump(salient_q_config, f, indent=2)

if __name__ == "__main__":
    main()

# Example train command:
# deepspeed --num_gpus=4 training/train.py \
#     --base_model /path/to/Qwen3-VL-8B-Instruct \
#     --npz_dir /path/to/cache_npz/train \
#     --output_dir ./output/salient_q_$(date +%Y%m%d%H) \
#     --epochs 1 \
#     --learning_rate 2e-5 \
#     --batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --num_scouts 32 \
#     --save_steps 200 \
#     --deepspeed configs/ds_zero2_no_offload.json
