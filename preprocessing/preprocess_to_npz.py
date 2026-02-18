#!/usr/bin/env python3
import os
import json
import argparse
import torch
import numpy as np
import gc
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from PIL import Image
from tqdm import tqdm


def downsample_mask_to_tokens(mask_image, target_height, target_width):
    target_height = max(1, int(target_height))
    target_width = max(1, int(target_width))
    mask_resized = mask_image.resize((target_width, target_height), Image.BILINEAR)
    return np.array(mask_resized).astype(np.float32) / 255.0


def load_salient_q_data(file_path, base_path):
    loaded_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                item = json.loads(line)
                image_path = item.get('image', '')
                mask_path = item.get('mask', '')
                conversations = item.get('conversations', [])

                if not image_path or not conversations:
                    continue

                full_image_path = os.path.join(base_path, image_path) if not image_path.startswith('/') else image_path
                full_mask_path = os.path.join(base_path, mask_path) if mask_path and not mask_path.startswith('/') else mask_path

                user_msg = next((msg for msg in conversations if msg.get('from') == 'human'), None)
                assistant_msg = next((msg for msg in conversations if msg.get('from') == 'gpt'), None)

                if user_msg and assistant_msg:
                    phi_elements = []
                    try:
                        assistant_data = json.loads(assistant_msg['value'])
                        phi_elements = assistant_data.get('phi_elements', [])
                    except Exception:
                        pass

                    loaded_data.append({
                        'image_path': full_image_path,
                        'mask_path': full_mask_path,
                        'question': user_msg.get('value', '').replace('<image>\n', ''),
                        'answer': assistant_msg.get('value', ''),
                        'phi_elements': phi_elements,
                    })
            except Exception:
                continue
    return loaded_data


def process_and_save_example(idx, example, processor, tokenizer, output_dir, max_length):
    category_to_id = {'name': 1, 'DOB': 2, 'SSN': 3, 'address': 4,'phone': 5, 'email': 6, 'medical record numbers': 7, 'sex': 8,'background': 0}
    SPATIAL_MERGE_SIZE = 2
    image_path = example['image_path']
    mask_path = example['mask_path']
    question = example['question']
    answer = example['answer']
    phi_elements = example['phi_elements']

    try:
        if not os.path.exists(image_path):
            return False

        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size

        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new('L', (img_width, img_height), 0)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]

        instruction_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        instruction_inputs = processor(
            text=[instruction_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )

        instruction_input_ids = instruction_inputs['input_ids'][0].tolist()
        instruction_attention_mask = instruction_inputs['attention_mask'][0].tolist()
        pixel_values = instruction_inputs['pixel_values'].squeeze(0).numpy()
        image_grid_thw = instruction_inputs['image_grid_thw'].squeeze(0).numpy()

        grid_t, grid_h, grid_w = image_grid_thw[0], image_grid_thw[1], image_grid_thw[2]
        merged_h = grid_h // SPATIAL_MERGE_SIZE
        merged_w = grid_w // SPATIAL_MERGE_SIZE

        mask_downsampled = downsample_mask_to_tokens(mask, merged_h, merged_w)
        mask_flat = np.tile(mask_downsampled.flatten(), grid_t) if grid_t > 1 else mask_downsampled.flatten()

        boxes, det_labels = [], []
        for elem in phi_elements:
            bbox_2d = elem.get('bbox_2d', [])
            if len(bbox_2d) == 4:
                x1, y1, x2, y2 = bbox_2d
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_width, x2), min(img_height, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height])
                    det_labels.append(category_to_id.get(elem.get('type', 'background'), 0))

        max_boxes = 10
        while len(boxes) < max_boxes:
            boxes.append([0.0, 0.0, 0.0, 0.0])
            det_labels.append(0)
        boxes, det_labels = boxes[:max_boxes], det_labels[:max_boxes]

        response = tokenizer(f"{answer}{tokenizer.eos_token}", add_special_tokens=False)
        response_input_ids = response['input_ids']
        response_attention_mask = response['attention_mask']

        remaining = max_length - len(instruction_input_ids)
        response_to_add = response_input_ids[:remaining] if remaining > 0 else []
        response_mask_to_add = response_attention_mask[:remaining] if remaining > 0 else []

        current_input_ids = instruction_input_ids + response_to_add
        current_attention_mask = instruction_attention_mask + response_mask_to_add
        current_labels = [-100] * len(instruction_input_ids) + response_to_add

        pad_id = tokenizer.pad_token_id or 0
        if len(current_input_ids) < max_length:
            pad_len = max_length - len(current_input_ids)
            current_input_ids += [pad_id] * pad_len
            current_attention_mask += [0] * pad_len
            current_labels += [-100] * pad_len

        current_input_ids = current_input_ids[:max_length]
        current_attention_mask = current_attention_mask[:max_length]
        current_labels = current_labels[:max_length]

        npz_path = os.path.join(output_dir, f"sample_{idx:06d}.npz")
        np.savez_compressed(
            npz_path,
            input_ids=np.array(current_input_ids, dtype=np.int32),
            attention_mask=np.array(current_attention_mask, dtype=np.int8),
            labels=np.array(current_labels, dtype=np.int32),
            pixel_values=pixel_values.astype(np.float16),
            image_grid_thw=image_grid_thw.astype(np.int16),
            saliency_masks=mask_flat.astype(np.float16),
            detection_boxes=np.array(boxes, dtype=np.float32),
            detection_labels=np.array(det_labels, dtype=np.int8),
        )

        del image, mask, image_inputs, video_inputs, instruction_inputs
        return True

    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Preprocess Salient-Q data to NPZ format")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    train_dir = os.path.join(args.output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_file = os.path.join(args.data_dir, "salient_q_train.jsonl")
    train_data = load_salient_q_data(train_file, args.base_path)

    success = 0
    failed = 0

    for idx, example in enumerate(tqdm(train_data, desc="Processing")):
        if process_and_save_example(idx, example, processor, tokenizer, train_dir, args.max_length):
            success += 1
        else:
            failed += 1

        if idx % 100 == 0:
            gc.collect()

    manifest = {
        "total_examples": success,
        "failed": failed,
        "format": "npz",
        "files": [f"sample_{i:06d}.npz" for i in range(success)],
    }
    with open(os.path.join(args.output_dir, "manifest.json"), 'w') as f:
        json.dump(manifest, f)

if __name__ == "__main__":
    main()
