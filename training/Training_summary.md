# Training Scripts Summary

## Cleaned Training Files

### 1. data_loader.py
**Purpose:** Load and preprocess ptychography data for training

**Key Features:**
- ✅ Loads images from disk for VLM training
- ✅ Extracts text-only content for LLM training
- ✅ Handles train/val/test splits
- ✅ Proper error handling for missing images
- ✅ Preserves metadata

**Usage:**
```python
from data_loader import PtychographyDataLoader

# For VLM (with images)
loader = PtychographyDataLoader(image_base_path="./data/images")
train, val, test = loader.load_all_splits("./data/splits")

# For LLM (text-only)
train, val, test = loader.load_all_splits_text_only("./data/splits")
```

### 2. train_vlm.py
**Purpose:** Train vision-language models for artifact detection

**Key Improvements:**
- ✅ All paths configurable via arguments
- ✅ No hardcoded directories
- ✅ Comprehensive argument parser
- ✅ Evaluation during and after training
- ✅ Saves training info JSON
- ✅ Clean code, no commented blocks

**Example Usage:**
```bash
python train_vlm.py \
    --data_dir ./data/splits \
    --image_base_path ./data/images \
    --model_name Llama-3.2-90B-Vision-Instruct \
    --output_dir ./models/vlm_90b \
    --num_epochs 50 \
    --learning_rate 2e-4 \
    --lora_r 16
```

### 3. train_llm.py
**Purpose:** Train language models for parameter recommendation

**Key Improvements:**
- ✅ All paths configurable via arguments
- ✅ Text-only processing (no images)
- ✅ Proper dataset tokenization
- ✅ Evaluation during and after training
- ✅ Saves training info JSON
- ✅ Clean code, no commented blocks

**Example Usage:**
```bash
python train_llm.py \
    --data_dir ./data/splits \
    --model_name Meta-Llama-3.1-70B-Instruct \
    --output_dir ./models/llm_70b \
    --num_epochs 50 \
    --learning_rate 2e-4 \
    --lora_r 16
```

## What Was Changed

### From Original Code:

**Removed:**
- ❌ Hardcoded paths (DATA_DIR, OUTPUT_DIR, IMAGE_BASE_PATH)
- ❌ Commented code blocks (# ''')
- ❌ Debug print statements
- ❌ Manual timestamp/filename parsing
- ❌ Hardcoded model selections

**Added:**
- ✅ Complete argparse configuration
- ✅ Flexible path handling
- ✅ Clear docstrings with usage examples
- ✅ Consistent error handling
- ✅ Training info logging

**Preserved:**
- ✅ All training logic (LoRA, SFT, evaluation)
- ✅ Model loading from Unsloth
- ✅ Data processing pipeline
- ✅ Evaluation functions
- ✅ Training statistics

## Training Configuration

### Recommended Settings

**VLM (Artifact Detection):**
```bash
--num_epochs 50
--learning_rate 2e-4
--lora_r 16
--lora_alpha 16
--batch_size 1
--gradient_accumulation_steps 8
--max_seq_length 2048
--eval_steps 25
--save_steps 50
```

**LLM (Parameter Recommendation):**
```bash
--num_epochs 50
--learning_rate 2e-4
--lora_r 16
--lora_alpha 16
--batch_size 4
--gradient_accumulation_steps 8
--max_seq_length 2048
--eval_steps 25
--save_steps 50
```

## Data Format

Your data splits should follow this structure:

```
data/
├── splits/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── images/
    ├── image_001.png
    ├── image_002.png
    └── ...
```

**JSON Format (VLM):**
```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Identify artifacts..."},
          {"type": "image", "image_path": "image_001.png"}
        ]
      },
      {
        "role": "assistant",
        "content": [
          {"type": "text", "text": "<answer>[Grid artifacts, Halo artifacts]</answer>"}
        ]
      }
    ],
    "metadata": {
      "sample_type": "NCM battery",
      "sample_id": "001"
    }
  }
]
```

**JSON Format (LLM - same structure, images ignored):**
The LLM data loader extracts only text content and ignores image fields.

## Hardware Requirements

### VLM Training
- **90B model**: 4-8 A100 GPUs (80GB each)
- **11B model**: 2-4 A100 GPUs (40-80GB each)
- Batch size 1 with gradient accumulation
- Mixed precision training (bf16/fp16)
- LoRA reduces memory significantly

### LLM Training
- **70B model**: 4-8 A100 GPUs (80GB each)
- **8B model**: 2 A100 GPUs (40GB each)
- Batch size 4 with gradient accumulation
- Mixed precision training
- LoRA adapter only

## Training Outputs

After training completes, you'll have:

```
output_dir/
├── adapter_config.json       # LoRA configuration
├── adapter_model.safetensors # LoRA weights
├── training_info.json        # Training metadata
├── tokenizer_config.json     # Tokenizer config
├── special_tokens_map.json   # Special tokens
└── tokenizer.json            # Tokenizer data
```

## Using Trained Models

Your trained models can be used directly with the inference scripts:

```bash
# Use your trained VLM
python inference/vlm_inference.py \
    --model_path ./models/vlm_90b \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --image_base_path ./data/images \
    --k_values 0 1 3 5 7 \
    --output_dir ./results
```

## Integration with Repository

**File Organization:**
```
ptychobench/
├── training/
│   ├── train_vlm.py
│   ├── train_llm.py
│   └── data_loader.py
├── inference/
│   ├── vlm_inference.py
│   ├── llm_inference.py
│   └── baseline_gpt4o.py
└── evaluation/
    ├── bootstrap.py
    ├── batch_bootstrap.py
    └── ptychobench_loader.py
```

## Notes

1. **Dependencies:** Training requires `unsloth`, `transformers`, `trl`, `torch`, `datasets`
2. **Data Access:** Request dataset access before training (see Data Access.md)
3. **Compute:** Training is expensive - consider using pre-trained checkpoints
4. **Reproducibility:** Set `--seed` for reproducible results
5. **Logging:** Use `--report_to wandb` for experiment tracking

## Quick Test

Test training scripts on a small subset:

```bash
# Quick VLM test (1 epoch, 10 samples)
python training/train_vlm.py \
    --data_dir ./data/splits \
    --image_base_path ./data/images \
    --model_name Llama-3.2-11B-Vision-Instruct \
    --output_dir ./test_models/vlm_test \
    --num_epochs 1 \
    --eval_samples 2

# Quick LLM test (1 epoch)
python training/train_llm.py \
    --data_dir ./data/splits \
    --model_name Meta-Llama-3.1-8B-Instruct \
    --output_dir ./test_models/llm_test \
    --num_epochs 1 \
    --eval_samples 2
```
