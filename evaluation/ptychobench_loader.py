"""
Data Loader for PtychoBench JSON Results
Extracts predictions and ground truth from the evaluation JSON files
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from collections import defaultdict

class PtychoBenchDataLoader:
    """
    Load and parse PtychoBench evaluation results
    """
    
    # Artifact classes (12 total)
    ARTIFACT_CLASSES = [
        "Grid artifacts",
        "Halo artifacts", 
        "Line artifacts",
        "Local distortion",
        "Low contrast",
        "No clear structure",
        "Noisy artifacts",
        "Non-uniform background",
        "Phase ramp",
        "Singularity artifacts",
        "Blurring",
        "No obvious artifacts"
    ]
    
    # Parameter recommendation classes (24 total)
    PARAM_CLASSES = [
        "Change number of batches to 1",
        "Disable momentum acceleration",
        "Disable multimodal update",
        "Disable position correction",
        "Enable momentum acceleration",
        "Enable multimodal update",
        "Enable position correction",
        "Increase batch size",
        "Increase number of OPR modes",
        "Increase number of probe modes",
        "Increase the number of iterations",
        "No changes needed",
        "Recenter diffraction patterns",
        "Reduce batch size",
        "Reduce diffraction pattern size by factor of 2",
        "Reduce or disable regularization",
        "Try other diffraction pattern orientations",
        "Turn off affine constraint",
        "Turn off variable probe correction (set OPR modes to 0)",
        "Turn on variable probe correction (set OPR modes to 1)",
        "Use Gaussian noise model",
        "Use compact batch selection scheme",
        "Use multislice model",
        "Use sparse batch selection scheme"
    ]
    
    def __init__(self, results_dir: Path):
        """
        Args:
            results_dir: Directory containing all result JSON files
        """
        self.results_dir = Path(results_dir)
        
    def parse_model_name(self, model_path: str) -> Tuple[str, bool]:
        """
        Extract model name and whether it's SFT or Base
        
        Args:
            model_path: Path from config
            
        Returns:
            (model_name, is_sft)
        """
        is_sft = model_path.startswith("/lus/eagle/projects/")
        
        # Extract model name from path
        if "Llama-3.2-11B-Vision" in model_path:
            model_name = "Llama 3.2-Vision 11B"
        elif "Llama-3.2-90B-Vision" in model_path:
            model_name = "Llama 3.2-Vision 90B"
        elif "Llama-3.1-8B" in model_path:
            model_name = "Llama 3.1 8B"
        elif "Llama-3.1-70B" in model_path:
            model_name = "Llama 3.1 70B"
        elif "gpt-4o" in model_path.lower():
            model_name = "GPT-4o"
        else:
            model_name = "Unknown Model"
        
        prefix = "SFT " if is_sft else "Base "
        if model_name != "GPT-4o":
            full_name = prefix + model_name
        else:
            full_name = model_name
            
        return full_name, is_sft
    
    def load_from_aggregated_metrics(self, json_file: Path) -> Dict:
        """
        Load results from JSON file with aggregated metrics
        
        Returns:
            Dictionary with structure for bootstrap analysis
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Parse configuration
        model_name, is_sft = self.parse_model_name(data['config']['model_path'])
        
        # Determine strategy from selection_mode
        strategy = "SSFS" if data['config']['selection_mode'] == 'similar' else "RFS"
        
        results = {
            'model': model_name,
            'is_sft': is_sft,
            'strategy': strategy,
            'n_test': data['config']['num_test'],
            'shot_results': {}
        }
        
        # Extract results for each shot configuration
        for shot_key in data.get('similar', data.get('random', {})).keys():
            shot_data = data.get('similar', data.get('random', {}))[shot_key]
            
            results['shot_results'][shot_key] = {
                'overall': shot_data['overall'],
                'per_sample_type': shot_data['per_sample_type'],
                'sample_distribution': shot_data['sample_distribution']
            }
        
        return results
    
    def load_all_results(self, task: str = "artifact") -> Dict:
        """
        Load all result files for a task
        
        Args:
            task: "artifact" or "parameter"
            
        Returns:
            Nested dictionary of all results
        """
        all_results = defaultdict(lambda: defaultdict(dict))
        
        # Find all JSON files
        pattern = f"*{task}*.json"
        for json_file in self.results_dir.glob(pattern):
            try:
                result = self.load_from_aggregated_metrics(json_file)
                
                model = result['model']
                strategy = result['strategy']
                
                all_results[model][strategy] = result['shot_results']
                
                print(f"Loaded: {model} - {strategy}")
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return dict(all_results)


def parse_answer_string(answer_str: str, task_type: str = "artifact") -> np.ndarray:
    """
    Parse answer string to binary array
    
    Args:
        answer_str: String like "<answer>[Grid artifacts, Halo artifacts]</answer>"
                    or "<answer>J, K, L</answer>"
        task_type: "artifact" or "param"
    
    Returns:
        Binary array of length 12 (artifact) or 24 (param)
    """
    # Remove <answer> tags
    answer_str = answer_str.replace("<answer>", "").replace("</answer>", "")
    
    # Remove brackets if present
    answer_str = answer_str.replace("[", "").replace("]", "")
    
    # Split by comma
    items = [item.strip() for item in answer_str.split(",")]
    
    if task_type == "artifact":
        # Artifact classes (12 total)
        all_classes = [
            "Grid artifacts",
            "Halo artifacts", 
            "Line artifacts",
            "Local distortion",
            "Low contrast",
            "No clear structure",
            "Noisy artifacts",
            "Non-uniform background",
            "Phase ramp",
            "Singularity artifacts",
            "Blurring",
            "No obvious artifacts"
        ]
        
        # Create binary array
        binary = np.zeros(len(all_classes), dtype=int)
        for item in items:
            if item in all_classes:
                idx = all_classes.index(item)
                binary[idx] = 1
        
        return binary
    
    else:  # param
        # Parameter classes (A-X)
        all_classes = list("ABCDEFGHIJKLMNOPQRSTUVWX")
        
        # Create binary array
        binary = np.zeros(len(all_classes), dtype=int)
        for item in items:
            if item in all_classes:
                idx = all_classes.index(item)
                binary[idx] = 1
        
        return binary


def extract_predictions_from_json(shot_data: Dict, task_type: str = "artifact") -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract actual predictions from JSON data
    
    Args:
        shot_data: Dictionary containing 'raw_predictions' and 'ground_truths'
        task_type: "artifact" or "param"
    
    Returns:
        (y_true, y_pred) arrays of shape (n_samples, n_classes)
    """
    raw_predictions = shot_data.get('raw_predictions', [])
    ground_truths = shot_data.get('ground_truths', [])
    
    if not raw_predictions or not ground_truths:
        raise ValueError("JSON must contain 'raw_predictions' and 'ground_truths' fields")
    
    if len(raw_predictions) != len(ground_truths):
        raise ValueError(f"Mismatch: {len(raw_predictions)} predictions vs {len(ground_truths)} ground truths")
    
    # Parse all predictions and ground truths
    y_pred_list = []
    y_true_list = []
    
    for pred_str, true_str in zip(raw_predictions, ground_truths):
        y_pred_list.append(parse_answer_string(pred_str, task_type))
        y_true_list.append(parse_answer_string(true_str, task_type))
    
    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true_list)
    
    return y_true, y_pred


def reconstruct_predictions_from_metrics(results: Dict, n_test: int = 79) -> Tuple[np.ndarray, np.ndarray]:
    """
    DEPRECATED: Use extract_predictions_from_json instead
    This function is kept for backward compatibility only
    """
    raise NotImplementedError(
        "This function is deprecated. Your JSON now contains actual predictions. "
        "Use extract_predictions_from_json() instead."
    )
