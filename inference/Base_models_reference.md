# Base Models Quick Reference

## Overview

Our fine-tuned checkpoints are **LoRA adapters** applied to Unsloth-optimized base models. You can use either:
1. **Fine-tuned checkpoints** (SFT + ICL)
2. **Base models directly** (ICL only)

## Base Models Used

### Vision-Language Models (VLM) - Artifact Detection

| Size | Base Model | HuggingFace Link |
|------|------------|------------------|
| 11B  | `unsloth/Llama-3.2-11B-Vision-Instruct` | [Link](https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct) |
| 90B  | `unsloth/Llama-3.2-90B-Vision-Instruct` | [Link](https://huggingface.co/unsloth/Llama-3.2-90B-Vision-Instruct) |

### Language Models (LLM) - Parameter Recommendation

| Size | Base Model | HuggingFace Link |
|------|------------|------------------|
| 8B   | `unsloth/Meta-Llama-3.1-8B-Instruct` | [Link](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct) |
| 70B  | `unsloth/Meta-Llama-3.1-70B-Instruct` | [Link](https://huggingface.co/unsloth/Meta-Llama-3.1-70B-Instruct) |

## Usage Examples

### Using Fine-tuned Checkpoints (SFT + ICL)

```bash
# VLM with fine-tuned checkpoint
python inference/vlm_inference.py \
    --model_path ./checkpoints/vlm_90b_sft \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --image_base_path ./data/images \
    --k_values 0 1 3 5 7

# LLM with fine-tuned checkpoint
python inference/llm_inference.py \
    --model_path ./checkpoints/llm_70b_sft \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --k_values 0 1 3 5 7
```

### Using Base Models (ICL only)

```bash
# VLM with base model
python inference/vlm_inference.py \
    --model_path unsloth/Llama-3.2-90B-Vision-Instruct \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --image_base_path ./data/images \
    --k_values 0 1 3 5 7

# LLM with base model
python inference/llm_inference.py \
    --model_path unsloth/Meta-Llama-3.1-70B-Instruct \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --k_values 0 1 3 5 7
```

## Key Differences

| Aspect | Fine-tuned Checkpoints | Base Models |
|--------|----------------------|-------------|
| **Model Path** | Local path (e.g., `./checkpoints/vlm_90b_sft`) | HuggingFace ID (e.g., `unsloth/Llama-3.2-90B-Vision-Instruct`) |
| **Training** | Pre-trained with LoRA on ptychography data | No domain-specific training |
| **Performance** | Task-optimized | General-purpose |
| **Use Case** | SFT + ICL evaluation | ICL-only evaluation |
| **Download Required** | Yes (from our HuggingFace) | No (loaded automatically) |

## Paper Results

In our paper, we report results for:
1. **Base models** with 0-shot, 1-shot, 3-shot, 5-shot, 7-shot ICL
2. **Fine-tuned models** with 0-shot, 1-shot, 3-shot, 5-shot, 7-shot ICL
3. **GPT-4o baseline** with 0-shot through 7-shot

This allows comparison of:
- SFT vs. no SFT
- Different ICL strategies (random vs. similar selection)
- Model size effects (8B vs. 70B, 11B vs. 90B)
