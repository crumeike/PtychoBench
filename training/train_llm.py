#!/usr/bin/env python3
"""
LLM training script for ptychography parameter recommendation.
Fine-tunes language models using LoRA for parameter recommendation.

Usage:
    python train_llm.py \\
        --data_dir ./data/splits \\
        --model_name Meta-Llama-3.1-70B-Instruct \\
        --output_dir ./models/llm_70b \\
        --num_epochs 50 \\
        --learning_rate 2e-4

Requirements:
    - unsloth
    - transformers
    - trl
    - torch
    - datasets
"""

import json
import os
import argparse
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer, DataCollatorForSeq2Seq
from datasets import Dataset
from data_loader import PtychographyDataLoader


def evaluate_on_samples(model, tokenizer, samples, num_samples=3, split_name="validation"):
    """
    Evaluate model on text-only samples
    
    Args:
        model: Fine-tuned model
        tokenizer: Model tokenizer
        samples: List of evaluation samples
        num_samples: Number of samples to evaluate
        split_name: Name of the split for logging
    """
    print(f"\n🧪 Evaluating on {split_name} set ({num_samples} samples):")
    print("=" * 60)
    
    FastLanguageModel.for_inference(model)
    
    for i in range(min(num_samples, len(samples))):
        sample = samples[i]
        messages = sample["messages"]
        
        test_instruction = messages[0]["content"]
        expected_response = messages[1]["content"]
        
        # Apply chat template
        chat_messages = [{"role": "user", "content": test_instruction}]
        
        input_text = tokenizer.apply_chat_template(
            chat_messages, 
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")
        
        print(f"\n--- Sample {i+1} ---")
        print("Expected:", expected_response)
        print("\nModel response:")
        
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        _ = model.generate(
            **inputs, 
            streamer=text_streamer, 
            max_new_tokens=500,
            use_cache=True, 
            temperature=0.3,
            min_p=0.1
        )
        print("\n" + "-" * 40)


def prepare_dataset(dataset, tokenizer, max_seq_length):
    """
    Prepare dataset for training by tokenizing
    
    Args:
        dataset: Raw dataset with messages
        tokenizer: Model tokenizer
        max_seq_length: Maximum sequence length
        
    Returns:
        Tokenized dataset
    """
    def preprocess_function(examples):
        # Convert chat format to text
        texts = []
        for messages in examples['messages']:
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        
        # Tokenize
        model_inputs = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_tensors=None,
        )
        
        # For causal LM, labels are the same as input_ids
        model_inputs['labels'] = model_inputs['input_ids'].copy()
        
        return model_inputs
    
    if dataset is None or len(dataset) == 0:
        return None
    
    if not isinstance(dataset, Dataset):
        dataset = Dataset.from_dict(dataset)
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(
        description='Train LLM for ptychography parameter recommendation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing train.json, val.json, test.json')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='Meta-Llama-3.1-70B-Instruct',
                       help='Base model name (e.g., Meta-Llama-3.1-70B-Instruct)')
    parser.add_argument('--load_in_4bit', action='store_true',
                       help='Load model in 4-bit quantization')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for model checkpoints')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Per-device batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Gradient accumulation steps')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                       help='Maximum sequence length')
    
    # LoRA arguments
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    
    # Logging arguments
    parser.add_argument('--eval_steps', type=int, default=25,
                       help='Evaluation frequency in steps')
    parser.add_argument('--save_steps', type=int, default=50,
                       help='Checkpoint save frequency in steps')
    parser.add_argument('--logging_steps', type=int, default=1,
                       help='Logging frequency in steps')
    parser.add_argument('--report_to', type=str, default='none',
                       choices=['none', 'wandb', 'tensorboard'],
                       help='Logging destination')
    
    # Evaluation arguments
    parser.add_argument('--eval_samples', type=int, default=3,
                       help='Number of samples to evaluate during training')
    parser.add_argument('--seed', type=int, default=3407,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("🚀 Starting LLM Training for Ptychography Parameter Recommendation")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Load data (text-only)
    print("\n📥 Loading data splits (text-only)...")
    loader = PtychographyDataLoader(image_base_path="")  # Not used for text-only
    train_data, val_data, test_data = loader.load_all_splits_text_only(args.data_dir)
    
    # Convert to Dataset format
    train_data = Dataset.from_list(train_data) if train_data else None
    val_data = Dataset.from_list(val_data) if val_data else None
    test_data = Dataset.from_list(test_data) if test_data else None
    
    if not train_data:
        print("❌ No training data found!")
        return
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data) if val_data else 0}")
    print(f"Test samples: {len(test_data) if test_data else 0}")
    
    # Load model
    print(f"\n📥 Loading model: {args.model_name}")
    full_model_path = f"unsloth/{args.model_name}"
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        full_model_path,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
        device_map="auto",
    )
    
    # Configure LoRA
    print("⚙️ Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Evaluate before training
    if val_data:
        print("\n🧪 Baseline evaluation (before training):")
        evaluate_on_samples(model, tokenizer, val_data, num_samples=2, split_name="validation")
    
    # Prepare datasets
    print("\n📊 Tokenizing datasets...")
    tokenized_train_data = prepare_dataset(train_data, tokenizer, args.max_seq_length)
    tokenized_eval_data = prepare_dataset(val_data, tokenizer, args.max_seq_length) if val_data else None
    
    # Train
    print("\n🎯 Starting training...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_eval_data if tokenized_eval_data else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            eval_strategy="steps" if val_data else "no",
            eval_steps=args.eval_steps if val_data else None,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=3,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            output_dir=args.output_dir,
            report_to=args.report_to,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=16,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
        ),
    )
    
    print(f"🏃‍♂️ Training on {len(train_data)} samples...")
    if val_data:
        print(f"📊 Evaluating on {len(val_data)} validation samples every {args.eval_steps} steps")
    
    trainer_stats = trainer.train()
    
    # Print training stats
    print(f"\n✅ Training complete!")
    print(f"Time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
    print(f"Final training loss: {trainer_stats.metrics['train_loss']:.4f}")
    if 'eval_loss' in trainer_stats.metrics:
        print(f"Final evaluation loss: {trainer_stats.metrics['eval_loss']:.4f}")
    
    # Evaluate after training
    if val_data:
        print("\n🎉 Post-training evaluation:")
        evaluate_on_samples(model, tokenizer, val_data, num_samples=args.eval_samples, split_name="validation")
    
    if test_data:
        print(f"\n🔬 Final evaluation on test set:")
        evaluate_on_samples(model, tokenizer, test_data, num_samples=args.eval_samples, split_name="test")
    
    # Save model
    print(f"\n💾 Saving model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training info
    training_info = {
        "model_name": full_model_path,
        "data_dir": args.data_dir,
        "train_samples": len(train_data),
        "val_samples": len(val_data) if val_data else 0,
        "test_samples": len(test_data) if test_data else 0,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    
    with open(os.path.join(args.output_dir, "training_info.json"), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print("✅ Training complete!")
    print(f"📁 Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
