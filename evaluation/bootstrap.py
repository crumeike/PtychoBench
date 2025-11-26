#!/usr/bin/env python3
"""
Bootstrap statistical analysis for PtychoBench results.
Computes confidence intervals and significance tests for model performance.

Usage:
    python bootstrap.py \\
        --results_file ./results/vlm_artifact_det_vlm_90b_sft_20250115_143022.json \\
        --n_bootstrap 10000 \\
        --output_file ./results/bootstrap_results.csv

Requirements:
    - numpy
    - pandas
    - scikit-learn
    - scipy
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from pathlib import Path
from ptychobench_loader import extract_predictions_from_json

@dataclass
class BootstrapResults:
    """Store bootstrap analysis results"""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    bootstrap_samples: np.ndarray

class PtychoBenchBootstrap:
    """
    Bootstrap analysis for PtychoBench evaluation
    """
    
    def __init__(self, n_bootstrap: int = 10000, confidence_level: float = 0.95, seed: int = 42):
        """
        Initialize bootstrap analyzer
        
        Args:
            n_bootstrap: Number of bootstrap samples (10000 recommended)
            confidence_level: Confidence level for intervals (0.95 = 95% CI)
            seed: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.seed = seed
        np.random.seed(seed)
        
    def compute_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        average: str = 'micro') -> float:
        """
        Compute F1 score
        
        Args:
            y_true: Ground truth labels (N,) or (N, C) for multi-label
            y_pred: Predicted labels (same shape as y_true)
            average: 'micro', 'macro', or 'weighted'
        """
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average=average, zero_division=0)
    
    def bootstrap_metric(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        metric_func=None, average: str = 'micro') -> BootstrapResults:
        """
        Perform bootstrap resampling to estimate metric distribution
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            metric_func: Custom metric function (uses F1 if None)
            average: Averaging method for F1 score
            
        Returns:
            BootstrapResults object with statistics
        """
        if metric_func is None:
            metric_func = lambda yt, yp: self.compute_f1_score(yt, yp, average)
        
        n_samples = len(y_true)
        bootstrap_scores = np.zeros(self.n_bootstrap)
        
        # Perform bootstrap resampling
        for i in range(self.n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Compute metric on bootstrap sample
            bootstrap_scores[i] = metric_func(y_true_boot, y_pred_boot)
        
        # Compute statistics
        mean_score = np.mean(bootstrap_scores)
        std_score = np.std(bootstrap_scores)
        
        # Compute confidence intervals (percentile method)
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
        
        return BootstrapResults(
            mean=mean_score,
            std=std_score,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            bootstrap_samples=bootstrap_scores
        )
    
    def compare_models(self, y_true: np.ndarray, 
                      y_pred_1: np.ndarray, 
                      y_pred_2: np.ndarray,
                      metric_func=None,
                      average: str = 'micro',
                      model_1_name: str = "Model 1",
                      model_2_name: str = "Model 2") -> Dict:
        """
        Statistical comparison between two models using bootstrap
        
        Returns:
            Dictionary with comparison statistics including p-value
        """
        if metric_func is None:
            metric_func = lambda yt, yp: self.compute_f1_score(yt, yp, average)
        
        n_samples = len(y_true)
        differences = np.zeros(self.n_bootstrap)
        
        # Bootstrap comparison
        for i in range(self.n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            score_1 = metric_func(y_true[indices], y_pred_1[indices])
            score_2 = metric_func(y_true[indices], y_pred_2[indices])
            
            differences[i] = score_1 - score_2
        
        # Compute p-value (two-tailed test)
        p_value = np.mean(differences > 0) * 2
        p_value = min(p_value, 2 - p_value)
        
        # Effect size (Cohen's d)
        effect_size = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0
        
        return {
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences),
            'ci_lower': np.percentile(differences, 2.5),
            'ci_upper': np.percentile(differences, 97.5),
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05,
            'differences': differences
        }


def analyze_result_file(json_file: Path, n_bootstrap: int = 10000) -> pd.DataFrame:
    """
    Analyze a single result file and compute bootstrap statistics
    
    Args:
        json_file: Path to result JSON file
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        DataFrame with bootstrap results
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract metadata
    config = data.get('config', {})
    model_path = config.get('model_path', 'unknown')
    
    # Determine model name
    if 'unsloth' in model_path or 'meta-llama' in model_path.lower() or 'llama' in model_path.lower():
        model_name = Path(model_path).name
    else:
        model_name = Path(model_path).parent.name
    
    # Determine strategy
    if 'similar' in data:
        strategy = "SSFS"
        results_key = 'similar'
    elif 'random' in data:
        strategy = "RFS"
        results_key = 'random'
    else:
        raise ValueError("No strategy found in results")
    
    # Determine task type
    task_type = "param" if "param" in str(json_file).lower() else "artifact"
    
    print(f"\nProcessing: {model_name} - {strategy}")
    
    # Initialize bootstrap analyzer
    bootstrap = PtychoBenchBootstrap(n_bootstrap=n_bootstrap, seed=42)
    
    # Process each shot configuration
    results = []
    
    for shot_config in ['0-shot', '1-shot', '3-shot', '5-shot', '7-shot']:
        if shot_config not in data[results_key]:
            continue
        
        print(f"  Analyzing {shot_config}...", end='')
        
        shot_data = data[results_key][shot_config]
        
        try:
            # Extract predictions
            y_true, y_pred = extract_predictions_from_json(shot_data, task_type)
            print(f" [{len(y_true)} samples]", end='')
            
            # Perform bootstrap
            boot_result = bootstrap.bootstrap_metric(y_true, y_pred, average='micro')
            
            results.append({
                'Model': model_name,
                'Strategy': strategy,
                'Shot': shot_config,
                'Mean': boot_result.mean,
                'Std': boot_result.std,
                'CI_Lower': boot_result.ci_lower,
                'CI_Upper': boot_result.ci_upper,
                'CI_Width': boot_result.ci_upper - boot_result.ci_lower,
                'Original_F1': shot_data['overall']['micro_f1']
            })
            
            print(f" Mean: {boot_result.mean:.3f} ± {boot_result.std:.3f}, "
                  f"CI: [{boot_result.ci_lower:.3f}, {boot_result.ci_upper:.3f}]")
        
        except Exception as e:
            print(f" Error: {e}")
            continue
    
    return pd.DataFrame(results)


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Bootstrap statistical analysis for PtychoBench results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to result JSON file')
    parser.add_argument('--n_bootstrap', type=int, default=10000,
                       help='Number of bootstrap iterations (default: 10000)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output CSV file for results (default: auto-generated)')
    parser.add_argument('--confidence_level', type=float, default=0.95,
                       help='Confidence level for intervals (default: 0.95)')
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    
    if not results_file.exists():
        print(f"Error: File not found: {results_file}")
        return
    
    print("="*80)
    print("PTYCHOBENCH BOOTSTRAP ANALYSIS")
    print("="*80)
    print(f"Results file: {results_file}")
    print(f"Bootstrap iterations: {args.n_bootstrap}")
    print(f"Confidence level: {args.confidence_level}")
    
    # Analyze results
    df = analyze_result_file(results_file, n_bootstrap=args.n_bootstrap)
    
    if df.empty:
        print("\nNo results to analyze")
        return
    
    # Generate output filename
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = results_file.parent / f"{results_file.stem}_bootstrap.csv"
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Average Mean F1: {df['Mean'].mean():.3f}")
    print(f"Average Std: {df['Std'].mean():.3f}")
    print(f"Average CI Width: {df['CI_Width'].mean():.3f}")
    print(f"Min CI Width: {df['CI_Width'].min():.3f}")
    print(f"Max CI Width: {df['CI_Width'].max():.3f}")
    
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
