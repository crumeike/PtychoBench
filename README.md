# Adapting Foundation Models for X-ray Ptychography in Low-Data Regimes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/abs/)

This repository contains an implementation of "Adapting general-purpose foundation models for X-ray Ptychography in Low-Data Regimes" accepted at the NeurIPS 2025 AI4MAT Workshop.

## 📋 Overview

This repository contains the **PtychoBench** benchmark code for evaluating Vision-Language Models (VLMs) and Large Language Models (LLMs) on ptychographic image analysis tasks. We investigate two specialization strategies:
- **Supervised Fine-Tuning (SFT)** using LoRA
- **In-Context Learning (ICL)** with context-aware retrieval

### Key Findings
- Task-dependent optimal specialization pathways
- contextual interference phenomenon in fine-tuned models
- Rigid 'super expert' tendencies observed in large parameter models on the textual task
- VLM artifact detection: SFT + ICL complementary (Micro-F1: 0.728)
- LLM parameter recommendation: ICL on large base model superior (Micro-F1: 0.847)
- Context relevance is critical for both strategies

## 🏗️ Repository Structure

```
.
├── data/
│   ├── ptychobench/              # PtychoBench dataset (available upon request but subject to institutional approval)
│   │   ├── images/               # Ptychographic reconstruction images
│   │   ├── annotations/          # Expert annotations
│
├── models/
│   ├── vlm/                      # Vision-Language Model implementations
│   │   ├── inference.py          # VLM inference pipeline
│   ├── llm/                      # Language Model implementations
│   │   ├── inference.py          # LLM inference pipeline
│   └── baselines/                # Baseline implementations (DINOv3, GPT-4o)
│
├── evaluation/
│   ├── bootstrapping             # Statistical validation
│
│
├── checkpoints/                  # Pre-trained model checkpoints (LoRA adapters)
│   ├── vlm_11b_sft/
│   ├── vlm_90b_sft/
│   ├── llm_8b_sft/
│   └── llm_70b_sft/
│
├── notebooks/
│   ├── visualizations.ipynb      # Result plotting
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/crumeike/ptychobench.git
cd ptychobench

```

### Dataset Access

The PtychoBench dataset is available upon request but subject to institutional approval. 

**To request access:**
- Email: yjiang@anl.gov
- Include: Your name, institution, and intended use case
- Data will be provided under a data use agreement

Once approved, place the dataset in the `data/ptychobench/` directory.

### Running Experiments

#### 1. VLM Artifact Detection

```bash

# Evaluate with ICL (SSFS)
python fewshot_eval_unsloth.py
    --model_path unsloth/Llama-3.2-90B-Vision-Instruct
    --train_file ./ptycho_data_splits_20250814_101410/train.json
    --test_file ./ptycho_data_splits_20250814_101410/test.json
    --image_base_path /home/cumeike/ptycho-vlm-data/data/images
    --k_values 0 1 3 5 7
    --selection_mode random
    --num_test 79
    --seed 12345


# Evaluate with ICL (RFS)
python experiments/run_vlm_experiments.py \
    --model_name meta-llama/Llama-3.2-11B-vision \
    --task artifact_detection \
    --train_mode icl \
    --context_strategy ssfs \
    --num_shots 0 1 3 5 7 \
    --checkpoint_path ./checkpoints/vlm_11b_sft

```

#### 2. LLM Parameter Recommendation

```bash
# Fine-tune Llama 3.1 70B
python experiments/run_llm_experiments.py \
    --model_name meta-llama/Meta-Llama-3.1-70B-Instruct \
    --task parameter_recommendation \
    --train_mode sft \
    --output_dir ./checkpoints/llm_70b_sft

# Evaluate with ICL (SSFS)
python fewshot_eval_unsloth.py
    --model_path unsloth/Llama-3.2-11B-Vision-Instruct
    --train_file ./ptycho_data_splits_20250814_101410/train.json
    --test_file ./ptycho_data_splits_20250814_101410/test.json
    --image_base_path /home/cumeike/ptycho-vlm-data/data/images
    --k_values 0 1 3 5 7
    --selection_mode random
    --num_test 79
    --seed 1234

```

## 🎯 Pre-trained Models

LoRA adapters for all fine-tuned models are available in the `checkpoints/` directory:

| Model | Task | Parameters | Link |
|-------|------|-----------|------|
| Llama 3.2-Vision 11B | Artifact Detection | 11B + LoRA (r=16) | [Download](checkpoints/vlm_11b_sft) |
| Llama 3.2-Vision 90B | Artifact Detection | 90B + LoRA (r=16) | [Download](checkpoints/vlm_90b_sft) |
| Llama 3.1 8B | Parameter Rec. | 8B + LoRA (r=16) | [Download](checkpoints/llm_8b_sft) |
| Llama 3.1 70B | Parameter Rec. | 70B + LoRA (r=16) | [Download](checkpoints/llm_70b_sft) |


## 📈 Reproducing Results

### Main Results (Tables 1 & 2)

```bash
# Run all VLM experiments 
bash scripts/run_all_vlm_experiments.sh

# Run all LLM experiments
bash scripts/run_all_llm_experiments.sh

```
- Note that results may vary slightly (particularly for VLM experiments), as ICL utilizes a random image selection process for both SSFS and RFS.

### Statistical Validation

```bash
# Bootstrap confidence intervals (n=10,000)
python evaluation/bootstrap.py \
    --results_file ./results/vlm_results.json \
    --n_bootstrap 10000 \
    --output_file ./results/vlm_confidence_intervals.csv

# Generate results tables with confidence intervals
python evaluation/bootstrapping/generate_tables.py --output_dir ./results

```

## 📖 Citation

If you use this code or the PtychoBench dataset, please cite:

```bibtex
@inproceedings{umeike2025ptychobench,
  title={Adapting general-purpose foundation models for X-ray Ptychography in Low-Data Regimes},
  author={Umeike, Robinson and Getty, Neil and Yin, Xiangyu and Jiang, Yi},
  booktitle={NeurIPS 2025 Workshop on AI4MAT},
  year={2025}
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
- **Yin Xiangyu** xyin@anl.gov

For dataset access requests, contact: yjiang@anl.gov, xyin@anl.gov

## 🔗 Related Work

- **PEAR Framework**: [arXiv:2410.09034](https://arxiv.org/abs/2410.09034)
