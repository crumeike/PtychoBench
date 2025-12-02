## Training Configuration

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
          {"type": "text", "text": "What types of artifacts are visible..."},
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

## Notes

1. **Dependencies:** Training requires `unsloth`, `transformers`, `trl`, `torch`, `datasets`
2. **Data Access:** Request dataset access before training (see [Data Access.md](Data Access.md))
3. **Compute:** Training is expensive - consider using pre-trained checkpoints
4. **Reproducibility:** Set `--seed` for reproducible results
5. **Logging:** Use `--report_to wandb` for experiment tracking
