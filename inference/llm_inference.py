#!/usr/bin/env python3
"""
LLM inference for parameter recommendation task using fine-tuned Llama models.
Evaluates fine-tuned LLM with k-shot learning (k=0,1,3,5,7...).

Usage:
    python llm_inference.py \\
        --model_path ./checkpoints/llm_70b_sft \\
        --train_file ./data/train.json \\
        --test_file ./data/test.json \\
        --k_values 0 1 3 5 7 \\
        --selection_mode similar \\
        --seed 42 \\
        --output_dir ./results

Requirements:
    - unsloth
    - torch
    - transformers
"""

import ast
import os
import json
import random
import torch
import numpy as np
from typing import List, Dict
import re
from sklearn.metrics import f1_score, precision_score, recall_score
from unsloth import FastLanguageModel
from datetime import datetime

class FewShotEvaluator:
    def __init__(self, model_path: str, train_file: str, test_file: str):
        """
        Initialize evaluator with model and data
        
        Args:
            model_path: Path to fine-tuned model
            train_file: Path to training data JSON
            test_file: Path to test data JSON  
        """
        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=8192,
            load_in_4bit=True if "bnb-4bit" in model_path else False,
            use_gradient_checkpointing="unsloth",
            device_map="auto"
        )

        FastLanguageModel.for_inference(self.model)

        # Load and merge data
        print("Loading data...")
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        # Create pools
        self.all_samples = train_data + test_data
        self.test_indices = list(range(len(train_data), len(self.all_samples)))

        print(f"Total samples: {len(self.all_samples)}")
        print(f"Test samples: {len(self.test_indices)}")
        
    
    def get_similar_samples(self, test_sample: Dict, available_indices: List[int], k: int) -> List[int]:
        """
        Select k samples with same sample_type, fill remainder with random
        
        Args:
            test_sample: Current test sample
            available_indices: Indices available for selection
            k: Number of examples to select
        """
        test_sample_type = test_sample["metadata"].get("sample_type", "unknown")
        
        # Filter for same sample type
        matching_indices = [
            idx for idx in available_indices 
            if self.all_samples[idx]["metadata"].get("sample_type") == test_sample_type
        ]
        
        # Get non-matching indices for random fill
        non_matching_indices = [
            idx for idx in available_indices 
            if self.all_samples[idx]["metadata"].get("sample_type") != test_sample_type
        ]
        
        # Use all matching samples + random fill if needed
        if len(matching_indices) >= k:
            return random.sample(matching_indices, k)
        else:
            remaining_needed = k - len(matching_indices)
            if remaining_needed > 0 and non_matching_indices:
                random_fill = random.sample(non_matching_indices, 
                                           min(remaining_needed, len(non_matching_indices)))
                return matching_indices + random_fill
            return matching_indices


    def generate_with_examples(self, examples: List[Dict], query: Dict) -> str:
        """
        Build prompt with examples and generate response
        
        Args:
            examples: Few-shot examples
            query: Test sample to predict
        """
        messages = []
        
        # Add few-shot examples
        for ex in examples:
            user_content = ex["messages"][0]["content"]
            messages.append({
                "role": "user",
                "content": {
                    "type": "text",
                    "text": user_content[0]["text"]
                }
            })
            
            assistant_content = ex["messages"][1]["content"]
            messages.append({
                "role": "assistant",
                "content": {
                    "type": "text",
                    "text": assistant_content[0]["text"]
                }
            })
        
        # Add query (only user part)
        query_content = query["messages"][0]["content"]
        messages.append({
            "role": "user",
            "content": {
                "type": "text",
                "text": query_content[0]["text"]
            }
        })

        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        inputs = self.tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to("cuda")

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                use_cache=True,
                temperature=0.3,
                min_p=0.1,
                do_sample=True
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        )
        
        return response
    
    def extract_labels(self, response: str) -> List[str]:
        """Extract artifact labels from response"""
        match = re.search(r'<answer>\[?(.*?)\]?</answer>', response)
        if not match:
            return []
        
        labels_str = match.group(1)
        if not labels_str.strip():
            return []
        
        labels = [label.strip() for label in labels_str.split(',')]
        return [label for label in labels if label and label != "No obvious artifacts"]
    
    def compute_metrics(self, ground_truths: List[str], predictions: List[str]) -> Dict:
        """Compute classification metrics"""
        all_labels = set()
        y_true = []
        y_pred = []
        
        for gt, pred in zip(ground_truths, predictions):
            gt_labels = self.extract_labels(gt)
            pred_labels = self.extract_labels(pred)
            
            all_labels.update(gt_labels)
            all_labels.update(pred_labels)
            
            y_true.append(gt_labels)
            y_pred.append(pred_labels)
        
        all_labels = sorted(list(all_labels))
        
        # Convert to binary format
        def to_binary(labels_list):
            return [1 if label in labels_list else 0 for label in all_labels]
        
        y_true_binary = [to_binary(labels) for labels in y_true]
        y_pred_binary = [to_binary(labels) for labels in y_pred]

        # Calculate metrics
        metrics = {
            'micro_precision': precision_score(y_true_binary, y_pred_binary, average='micro', zero_division=0),
            'micro_recall': recall_score(y_true_binary, y_pred_binary, average='micro', zero_division=0),
            'micro_f1': f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0),
            'macro_precision': precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=0),
            'macro_recall': recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=0),
            'macro_f1': f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
        }
        
        return metrics
    
    def compute_metrics_per_sample_type(self, ground_truths: List[str], predictions: List[str], 
                                    sample_types: List[str]) -> Dict:
        """Compute metrics broken down by sample type"""
        sample_type_groups = {}
        for gt, pred, stype in zip(ground_truths, predictions, sample_types):
            if stype not in sample_type_groups:
                sample_type_groups[stype] = {'ground_truths': [], 'predictions': []}
            sample_type_groups[stype]['ground_truths'].append(gt)
            sample_type_groups[stype]['predictions'].append(pred)
        
        per_type_metrics = {}
        for stype, data in sample_type_groups.items():
            if len(data['ground_truths']) > 0:
                per_type_metrics[stype] = self.compute_metrics(
                    data['ground_truths'], 
                    data['predictions']
                )
                per_type_metrics[stype]['n_samples'] = len(data['ground_truths'])
    
        return per_type_metrics

    def evaluate_k_shot(self, k_values: List[int] = [0, 1, 3, 5, 7], 
                       selection_mode: str = 'similar',
                       num_test_samples: int = None) -> Dict:
        """
        Evaluate model across different k-shot settings
        
        Args:
            k_values: List of k values to test
            selection_mode: 'random' or 'similar' 
            num_test_samples: Number of test samples to evaluate (None = all)
        """
        results = {}
        test_indices_to_use = self.test_indices[:num_test_samples] if num_test_samples else self.test_indices
        
        for k in k_values:
            print(f"\nEvaluating {k}-shot with {selection_mode} selection...")
            predictions = []
            ground_truths = []
            sample_types = []

            for i, test_idx in enumerate(test_indices_to_use):
                print(f"  Sample {i+1}/{len(test_indices_to_use)}", end='\r')
                
                test_sample = self.all_samples[test_idx]
                sample_type = test_sample["metadata"].get("sample_type", "unknown")
                
                # Create pool excluding current test sample
                available_indices = [idx for idx in range(len(self.all_samples)) if idx != test_idx]

                # Select examples based on mode
                if k > 0:
                    if selection_mode == 'random':
                        example_indices = random.sample(available_indices, min(k, len(available_indices)))
                    elif selection_mode == 'similar':
                        example_indices = self.get_similar_samples(test_sample, available_indices, k)
                    
                    examples = [self.all_samples[idx] for idx in example_indices]
                else:
                    examples = []

                # Generate prediction
                try:
                    prediction = self.generate_with_examples(examples, test_sample)
                    
                    # Normalize prediction
                    if isinstance(prediction, dict):
                        pred_text = prediction.get('text', '<answer></answer>')
                    else:
                        try:
                            maybe = ast.literal_eval(prediction)
                            pred_text = maybe.get('text', '<answer></answer>') if isinstance(maybe, dict) else prediction
                        except Exception:
                            pred_text = prediction

                    predictions.append(pred_text)
                    ground_truths.append(test_sample["messages"][1]["content"][0]["text"])
                    sample_types.append(sample_type)
                except Exception as e:
                    print(f"\n  Error on sample {i+1}: {e}")
                    continue
            
            # Compute metrics
            if predictions:
                overall_metrics = self.compute_metrics(ground_truths, predictions)
                per_type_metrics = self.compute_metrics_per_sample_type(
                    ground_truths, predictions, sample_types
                )
                
                results[f"{k}-shot"] = {
                    'overall': overall_metrics,
                    'per_sample_type': per_type_metrics,
                    'sample_distribution': dict(zip(*np.unique(sample_types, return_counts=True))),
                    'raw_predictions': predictions,
                    'ground_truths': ground_truths
                }
                
                print(f"\n  {k}-shot Results:")
                print(f"    Overall - Micro F1: {overall_metrics['micro_f1']:.3f}, Macro F1: {overall_metrics['macro_f1']:.3f}")
                print(f"    Per Sample Type:")
                for stype, metrics in per_type_metrics.items():
                    print(f"      {stype}: F1={metrics['micro_f1']:.3f} (n={metrics['n_samples']})")
        
        return results
    
def extract_model_name(path):
    """Extract model name from path"""
    base_name = os.path.basename(path)
    if base_name.startswith("checkpoint-"):
        base_name = os.path.basename(os.path.dirname(path))
    return base_name

def sanitize_filename(name):
    """Sanitize model name for use in filenames"""
    name = name.replace("/", "-")
    sanitized_name = re.sub(r'[^\w\-.]', '_', name)
    return sanitized_name

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def save_detailed_results(results: Dict, output_file: str):
    """Save comprehensive results including per-sample-type breakdown"""
    import pandas as pd
    results = convert_numpy_types(results) 
    
    # Save JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary DataFrame
    summary_data = []
    for mode, mode_results in results.items():
        if mode == 'config' or mode == 'model_info':
            continue
        for k_shot, metrics in mode_results.items():
            row = {
                'selection_mode': mode,
                'k_shot': k_shot,
                'overall_micro_f1': metrics['overall']['micro_f1'],
                'overall_micro_precision': metrics['overall']['micro_precision'],
                'overall_micro_recall': metrics['overall']['micro_recall'],
                'overall_macro_f1': metrics['overall']['macro_f1'],
                'overall_macro_precision': metrics['overall']['macro_precision'],
                'overall_macro_recall': metrics['overall']['macro_recall'],
            }
            
            # Add per-sample-type F1 scores
            for stype, stype_metrics in metrics['per_sample_type'].items():
                row[f'{stype}_f1'] = stype_metrics['micro_f1']
                row[f'{stype}_n'] = stype_metrics['n_samples']
            
            summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    csv_file = output_file.replace('.json', '_summary.csv')
    df.to_csv(csv_file, index=False)
    print(f"Summary CSV saved to: {csv_file}")
    
    return df

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='LLM inference for parameter recommendation task',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to fine-tuned model')
    parser.add_argument('--train_file', type=str, required=True, 
                       help='Path to training data JSON')
    parser.add_argument('--test_file', type=str, required=True, 
                       help='Path to test data JSON')
    parser.add_argument('--k_values', type=int, nargs='+', default=[0, 1, 3, 5, 7], 
                       help='k values for k-shot evaluation')
    parser.add_argument('--selection_mode', type=str, choices=['random', 'similar'], 
                       default='similar', help='Example selection mode')
    parser.add_argument('--num_test', type=int, default=None, 
                       help='Number of test samples to evaluate (default: all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize evaluator
    evaluator = FewShotEvaluator(
        args.model_path,
        args.train_file,
        args.test_file
    )
    
    # Run evaluation
    all_results = {}
    all_results['config'] = {
        'model_path': args.model_path,
        'train_file': args.train_file,
        'test_file': args.test_file,
        'k_values': args.k_values,
        'selection_mode': args.selection_mode,
        'num_test': args.num_test,
        'seed': args.seed
    }

    if args.selection_mode in ['random', 'similar']:
        print("\n" + "="*60)
        print(f"{args.selection_mode.upper()} SELECTION MODE")
        print("="*60)
        results = evaluator.evaluate_k_shot(
            k_values=args.k_values,
            selection_mode=args.selection_mode,
            num_test_samples=args.num_test
        )
        all_results[args.selection_mode] = results
    
    # Save results
    print(f"Model Path: {args.model_path}")
    extracted_name = extract_model_name(args.model_path)
    valid_model_name = sanitize_filename(extracted_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"{args.output_dir}/llm_param_rec_{valid_model_name}_{timestamp}.json"

    summary_df = save_detailed_results(all_results, output_filename)
    
    # Print comparison
    print("\n" + "="*60)
    print("SAMPLE TYPE PERFORMANCE COMPARISON")
    print("="*60)
    
    if args.selection_mode in all_results:
        sample_types = set()
        for mode, mode_results in all_results.items():
            if mode == 'config' or mode == 'model_info':
                continue
            for k_results in mode_results.values():
                sample_types.update(k_results['per_sample_type'].keys())
        
        for stype in sorted(sample_types):
            print(f"\n{stype}:")
            for k in args.k_values:
                k_shot = f"{k}-shot"
                f1 = all_results[args.selection_mode][k_shot]['per_sample_type'].get(stype, {}).get('micro_f1', 0)
                print(f"  {k_shot}: F1={f1:.3f}")

if __name__ == "__main__":
    main()
