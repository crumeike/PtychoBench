# PtychoBench: Adapting Foundation Models for X-ray Ptychography

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.02503-b31b1b.svg)](https://arxiv.org/abs/2511.02503)

## 📋 Overview

This repository contains the **PtychoBench** benchmark code for evaluating Vision-Language Models (VLMs) and Large Language Models (LLMs) on ptychographic image analysis tasks. We investigate two specialization strategies:
- **Supervised Fine-Tuning (SFT)** using LoRA
- **In-Context Learning (ICL)** with context-aware retrieval

### Key Findings
- Task-dependent optimal specialization pathways
- Contextual interference phenomenon in fine-tuned models under RFS strategy
- Rigid 'super expert' tendencies observed in large parameter models on the textual task
- **VLM artifact detection**: SFT + ICL complementary (Micro-F1: 0.728)
- **LLM parameter recommendation**: ICL on large base model superior (Micro-F1: 0.847)
- Context relevance is critical for both strategies

## 🏗️ Repository Structure

```
├── inference/
│   ├── vlm_inference.py          # VLM artifact detection
│   ├── llm_inference.py          # LLM parameter recommendation
│   └── baseline_gpt4o.py         # OpenAI GPT-4o baseline (both tasks)
│
├── training/
│   ├── train_vlm.py              # VLM training script
│   ├── train_llm.py              # LLM training script
│   └── data_loader.py            # Data loading utilities
│
├── evaluation/
│   ├── bootstrap.py              # Bootstrap analysis for single file
│   ├── batch_bootstrap.py        # Batch bootstrap analysis
│   └── ptychobench_loader.py     # Data loader utilities
│
├── checkpoints/                  # Download from HuggingFace
│   ├── vlm_11b_sft/
│   ├── vlm_90b_sft/
│   ├── llm_8b_sft/
│   └── llm_70b_sft/
│
├── Data Access.md                # Dataset access instructions
├── requirements.txt              # Python dependencies
├── CITATION.cff                  # Citation metadata
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/crumeike/ptychobench.git
cd ptychobench

# Install dependencies
pip install -r requirements.txt
```

### Download Checkpoints

Our fine-tuned LoRA adapters are hosted on HuggingFace:

```bash
# Download all checkpoints
huggingface-cli download crumeike/ptychobench-checkpoints --local-dir ./checkpoints
```

Or download individual models:
- [Llama 3.2-Vision 11B](https://huggingface.co/crumeike/ptychobench-checkpoints/tree/main/vlm_11b_sft)
- [Llama 3.2-Vision 90B](https://huggingface.co/crumeike/ptychobench-checkpoints/tree/main/vlm_90b_sft)
- [Llama 3.1 8B](https://huggingface.co/crumeike/ptychobench-checkpoints/tree/main/llm_8b_sft)
- [Llama 3.1 70B](https://huggingface.co/crumeike/ptychobench-checkpoints/tree/main/llm_70b_sft)

### Dataset Access

**To request access to the PtychoBench dataset:**
- Email: yjiang@anl.gov
- See [`Data Access.md`](Data%20Access.md) for full details

## 📊 Running Experiments

### 1. VLM Artifact Detection

#### Fine-tuned Model (SFT + ICL)
```bash
python inference/vlm_inference.py \
    --model_path ./checkpoints/vlm_90b_sft \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --image_base_path ./data/images \
    --k_values 0 1 3 5 7 \
    --selection_mode similar \
    --seed 42 \
    --output_dir ./results
```

#### Base Model (ICL only)
```bash
python inference/vlm_inference.py \
    --model_path unsloth/Llama-3.2-90B-Vision-Instruct \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --image_base_path ./data/images \
    --k_values 0 1 3 5 7 \
    --selection_mode similar \
    --seed 42 \
    --output_dir ./results
```

#### Baseline (GPT-4o)
```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

python inference/baseline_gpt4o.py \
    --task artifact_detection \
    --openai_model gpt-4o \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --image_base_path ./data/images \
    --k_values 0 1 3 5 7 \
    --selection_mode similar \
    --seed 42 \
    --output_dir ./results
```

### 2. LLM Parameter Recommendation

#### Fine-tuned Model (SFT + ICL)
```bash
python inference/llm_inference.py \
    --model_path ./checkpoints/llm_70b_sft \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --k_values 0 1 3 5 7 \
    --selection_mode similar \
    --seed 42 \
    --output_dir ./results
```

#### Base Model (ICL only)
```bash
python inference/llm_inference.py \
    --model_path unsloth/Meta-Llama-3.1-70B-Instruct \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --k_values 0 1 3 5 7 \
    --selection_mode similar \
    --seed 42 \
    --output_dir ./results
```

#### Baseline (GPT-4o)
```bash
python inference/baseline_gpt4o.py \
    --task parameter_recommendation \
    --openai_model gpt-4o \
    --train_file ./data/train.json \
    --test_file ./data/test.json \
    --k_values 0 1 3 5 7 \
    --selection_mode similar \
    --seed 42 \
    --output_dir ./results
```

## 🎯 Command-Line Arguments

### Common Arguments (All Scripts)
- `--train_file`: Path to training data JSON
- `--test_file`: Path to test data JSON
- `--k_values`: K-shot values to evaluate (default: 0 1 3 5 7)
- `--selection_mode`: Example selection strategy - `random` or `similar` (default: similar)
- `--num_test`: Number of test samples to evaluate (default: all)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output_dir`: Output directory for results (default: ./results)

### VLM-Specific
- `--model_path`: Path to fine-tuned VLM checkpoint
- `--image_base_path`: Directory containing ptychography images (required)

### LLM-Specific
- `--model_path`: Path to fine-tuned LLM checkpoint

### Baseline-Specific
- `--task`: Task type - `artifact_detection` or `parameter_recommendation`
- `--openai_model`: OpenAI model name (default: gpt-4o)
- `--image_base_path`: Required for artifact_detection task

## 📈 Output Files

Each evaluation produces:
- **JSON file**: Complete results with predictions and metrics
- **CSV file**: Summary table with per-sample-type breakdown

Example output structure:
```
results/
├── vlm_artifact_det_vlm_90b_sft_20250115_143022.json
├── vlm_artifact_det_vlm_90b_sft_20250115_143022_summary.csv
├── llm_param_rec_llm_70b_sft_20250115_150133.json
└── llm_param_rec_llm_70b_sft_20250115_150133_summary.csv
```

## 🏋️ Training Your Own Models

### VLM Training (Artifact Detection)

```bash
python training/train_vlm.py \
    --data_dir ./data/splits \
    --image_base_path ./data/images \
    --model_name Llama-3.2-90B-Vision-Instruct \
    --output_dir ./my_models/vlm_90b \
    --num_epochs 50 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --batch_size 1 \
    --gradient_accumulation_steps 8
```

### LLM Training (Parameter Recommendation)

```bash
python training/train_llm.py \
    --data_dir ./data/splits \
    --model_name Meta-Llama-3.1-70B-Instruct \
    --output_dir ./my_models/llm_70b \
    --num_epochs 50 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --batch_size 4 \
    --gradient_accumulation_steps 8
```

### Training Arguments

**Common Arguments:**
- `--data_dir`: Directory with train.json, val.json, test.json
- `--model_name`: Base model from Unsloth
- `--output_dir`: Where to save checkpoints
- `--num_epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 16)
- `--batch_size`: Per-device batch size
- `--gradient_accumulation_steps`: Gradient accumulation
- `--eval_steps`: Evaluation frequency (default: 25)
- `--save_steps`: Checkpoint save frequency (default: 50)
- `--report_to`: Logging destination (wandb, tensorboard, none)

**VLM-Specific:**
- `--image_base_path`: Directory containing images (required)
- `--load_in_4bit`: Enable 4-bit quantization

**Hardware Requirements:**
- VLM 90B: 4-8 A100 GPUs (80GB)
- VLM 11B: 2-4 A100 GPUs (40-80GB)
- LLM 70B: 4-8 A100 GPUs (80GB)
- LLM 8B: 2 A100 GPUs (40GB)

### Statistical Validation

Bootstrap confidence intervals and significance testing:

```bash
# Analyze a single result file
python evaluation/bootstrap.py \
    --results_file ./results/vlm_artifact_det_vlm_90b_sft_20250115_143022.json \
    --n_bootstrap 10000 \
    --output_file ./results/bootstrap_results.csv

# Batch analysis of all results in a directory
python evaluation/batch_bootstrap.py \
    --results_dir ./results \
    --output_dir ./bootstrap_output \
    --n_bootstrap 10000
```

The bootstrap analysis generates:
- CSV files with mean, std, and 95% confidence intervals
- Summary statistics

## 🔬 Model Architecture

### Fine-tuned Models (SFT with LoRA)

Our fine-tuned checkpoints are LoRA adapters applied to Unsloth-optimized base models:

| Checkpoint | Task | Base Model | Parameters | LoRA Rank | Download |
|------------|------|------------|-----------|-----------|----------|
| `vlm_11b_sft` | Artifact Detection | [unsloth/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct) | 11B | r=16 | [HF Link](https://huggingface.co/crumeike/ptychobench-checkpoints/tree/main/vlm_11b_sft) |
| `vlm_90b_sft` | Artifact Detection | [unsloth/Llama-3.2-90B-Vision-Instruct](https://huggingface.co/unsloth/Llama-3.2-90B-Vision-Instruct) | 90B | r=16 | [HF Link](https://huggingface.co/crumeike/ptychobench-checkpoints/tree/main/vlm_90b_sft) |
| `llm_8b_sft` | Parameter Rec. | [unsloth/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct) | 8B | r=16 | [HF Link](https://huggingface.co/crumeike/ptychobench-checkpoints/tree/main/llm_8b_sft) |
| `llm_70b_sft` | Parameter Rec. | [unsloth/Meta-Llama-3.1-70B-Instruct](https://huggingface.co/unsloth/Meta-Llama-3.1-70B-Instruct) | 70B | r=16 | [HF Link](https://huggingface.co/crumeike/ptychobench-checkpoints/tree/main/llm_70b_sft) |

**Note**: Our checkpoints contain only the LoRA adapter weights. The base models are automatically loaded from Unsloth's HuggingFace repository when running inference.

### Base Models for ICL

For In-Context Learning (ICL) experiments without fine-tuning, use the base models directly:
- **VLM**: `unsloth/Llama-3.2-11B-Vision-Instruct` or `unsloth/Llama-3.2-90B-Vision-Instruct`
- **LLM**: `unsloth/Meta-Llama-3.1-8B-Instruct` or `unsloth/Meta-Llama-3.1-70B-Instruct`

## 📖 Citation

If you use this code or the PtychoBench dataset, please cite:

```bibtex
@inproceedings{
umeike2025adapting,
title={Adapting general-purpose foundation models for X-ray Ptychography in Low-Data Regimes},
author={Robinson Umeike and Neil Getty and Xiangyu Yin and Yi Jiang},
booktitle={AI for Accelerated Materials Design - NeurIPS 2025 Workshop},
year={2025},
url={https://openreview.net/forum?id=zgLfoV5jjX}
}
```

## 🙏 Acknowledgments

This work was supported by:
- Laboratory Directed Research and Development (LDRD) Program at Argonne National Laboratory (Project 2025-0495)
- Advanced Photon Source, U.S. DOE Office of Science User Facility (Contract DE-AC02-06CH11357)

## 📧 Contact

For questions about the code:
- **Robinson Umeike**: crumeike@crimson.ua.edu
- **Neil Getty**: ngetty@anl.gov

For dataset access requests:
- **Yi Jiang**: yjiang@anl.gov
- **Xiangyu Yin**: xyin@anl.gov

## 🔗 Related Work

- **PEAR Framework**: [arXiv:2410.09034](https://arxiv.org/abs/2410.09034)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
