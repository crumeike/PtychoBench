#!/usr/bin/env python3
"""
VLM training script for ptychography artifact detection.
Fine-tunes vision-language models using LoRA for artifact identification.

Usage:
    python train_vlm.py \\
        --data_dir ./data/splits \\
        --image_base_path ./data/images \\
        --model_name Llama-3.2-90B-Vision-Instruct \\
        --output_dir ./models/vlm_90b \\
        --num_epochs 50 \\
        --learning_rate 2e-4

Requirements:
    - unsloth
    - transformers
    - trl
    - torch
"""

import json
import os
import argparse
import torch
from pathlib import Path
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from data_loader import PtychographyDataLoader

torch._dynamo.config.cache_size_limit = 256


def evaluate_on_samples(model, tokenizer, samples, num_samples=3, split_name="validation"):
    """
    Evaluate model on a set of samples
    
    Args:
        model: Fine-tuned model
        tokenizer: Model tokenizer
        samples: List of evaluation samples
        num_samples: Number of samples to evaluate
        split_name: Name of the split for logging
    """
    print(f"\n🧪 Evaluating on {split_name} set ({num_samples} samples):")
    print("=" * 60)
    
    FastVisionModel.for_inference(model)
    
    for i in range(min(num_samples, len(samples))):
        sample = samples[i]
        messages = sample["messages"]
        
        test_instruction = messages[0]["content"][0]["text"]
        
        # Find the image
        image_item = next((item for item in messages[0]["content"] if item["type"] == "image"), None)
        if not image_item or "image" not in image_item:
            print(f"❌ No image found in sample {i+1}")
            continue
            
        test_image = image_item["image"]
        expected_response = messages[1]["content"][0]["text"]
        
        # Prepare input
        chat_messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": test_instruction}
            ]}
        ]
        
        input_text = tokenizer.apply_chat_template(chat_messages, add_generation_prompt=True)
        
        inputs = tokenizer(
            test_image,
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
            max_new_tokens=200,
            use_cache=True, 
            temperature=0.3,
            min_p=0.1
        )
        print("\n" + "-" * 40)


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(
        description='Train VLM for ptychography artifact detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing train.json, val.json, test.json')
    parser.add_argument('--image_base_path', type=str, required=True,
                       help='Base directory containing images')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='Llama-3.2-90B-Vision-Instruct',
                       help='Base model name (e.g., Llama-3.2-90B-Vision-Instruct)')
    parser.add_argument('--load_in_4bit', action='store_true',
                       help='Load model in 4-bit quantization')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for model checkpoints')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
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
    
    print("🚀 Starting VLM Training for Ptychography Artifact Detection")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    print("\n📥 Loading data splits...")
    loader = PtychographyDataLoader(image_base_path=args.image_base_path)
    train_data, val_data, test_data = loader.load_all_splits(args.data_dir)
    
    if not train_data:
        print("❌ No training data found!")
        return
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Load model
    print(f"\n📥 Loading model: {args.model_name}")
    full_model_path = f"unsloth/{args.model_name}"
    
    model, tokenizer = FastVisionModel.from_pretrained(
        full_model_path,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
        device_map="balanced",
    )
    
    # Configure LoRA
    print("⚙️ Configuring LoRA...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Evaluate before training
    if val_data:
        print("\n🧪 Baseline evaluation (before training):")
        evaluate_on_samples(model, tokenizer, val_data, num_samples=2, split_name="validation")
    
    # Train
    print("\n🎯 Starting training...")
    FastVisionModel.for_training(model)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_data,
        eval_dataset=val_data if val_data else None,
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
        "val_samples": len(val_data),
        "test_samples": len(test_data),
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
