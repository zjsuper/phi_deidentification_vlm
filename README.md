Salient-Q is a multi-task Vision-Language Model (VLM) designed to detect and localize Protected Health Information (PHI) burned into medical images. Built on top of Qwen3-VL-8B, it introduces two novel architectural components:

## Repository Structure

```
Salient-Q/
├── README.md
├── requirements.txt
├── LICENSE
├── configs/
│   └── ds_zero2_no_offload.json    # DeepSpeed ZeRO-2 config
├── data_generation/
│   ├── phi_generator_utils.py      # Synthetic PHI text + font utilities
│   ├── phi_realistic_degradation.py # Digital degradation pipeline
│   ├── generate_phi_images.py      # Step 1: Generate PHI images + masks
│   └── generate_training_jsonl.py  # Step 2: Generate JSONL training data
├── preprocessing/
│   └── preprocess_to_npz.py        # Step 3: Convert JSONL to NPZ
├── model/
│   └── salient_q_model.py          # Model architecture + factory
└── training/
    └── train.py                    # Step 4: Multi-task training script
```

## Pipeline

### Step 1: Generate PHI-Embedded Medical Images

```bash
python -m data_generation.generate_phi_images \
    --xray_dir /path/to/xray/images \
    --output_dir /path/to/dataset \
    --phi_pairs /path/to/phi_pairs.json \
    --num_samples 50000
```

This generates:
- Degraded images with burned-in PHI text (`images/`)
- Binary saliency masks (`masks/`)
- Bounding box annotations (`coordinates/`)

### Step 2: Generate JSONL Training Data

```bash
python -m data_generation.generate_training_jsonl \
    -c /path/to/dataset/coordinates \
    -o /path/to/training_data \
    -b /path/to/dataset
```


### Step 3: Preprocess to NPZ

```bash
python -m preprocessing.preprocess_to_npz \
    --data_dir /path/to/training_data \
    --base_model /path/to/Qwen3-VL-8B-Instruct \
    --base_path /path/to/dataset \
    --output_dir /path/to/npz_cache
```

Tokenizes text, processes images through the Qwen3-VL processor, and saves each example as a compressed NPZ file for efficient training.

### Step 4: Train

```bash
deepspeed --num_gpus=4 training/train.py \
    --base_model /path/to/Qwen3-VL-8B-Instruct \
    --npz_dir /path/to/npz_cache/train \
    --output_dir ./output/salient_q \
    --epochs 3 \
    --learning_rate 2e-5 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_scouts 32 \
    --deepspeed configs/ds_zero2_no_offload.json
```
Install dependencies:

```bash
pip install -r requirements.txt
```