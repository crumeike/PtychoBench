#!/usr/bin/env python3
"""
Batch bootstrap analysis for multiple PtychoBench result files.
Processes all result files in a directory and generates summary tables.

Usage:
    python batch_bootstrap.py \\
        --results_dir ./results \\
        --output_dir ./bootstrap_output \\
        --n_bootstrap 10000

Requirements:
    - numpy
    - pandas
    - scikit-learn
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from bootstrap import PtychoBenchBootstrap, extract_predictions_from_json

class BatchBootstrapAnalyzer:
    """
    Batch bootstrap analysis for multiple result files
    """
    
    def __init__(self, results_dir: Path, output_dir: Path, n_bootstrap: int = 10000):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_bootstrap = n_bootstrap
        self.bootstrap = PtychoBenchBootstrap(n_bootstrap=n_bootstrap, seed=42)
    
    def process_result_file(self, json_file: Path) -> List[Dict]:
        """
        Process a single result file
        
        Returns:
            List of result dictionaries
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
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
            return []
        
        # Determine task type
        task_type = "param" if "param" in str(json_file).lower() else "artifact"
        
        print(f"\nProcessing: {model_name} - {strategy}")
        
        results = []
        
        for shot_config in ['0-shot', '1-shot', '3-shot', '5-shot', '7-shot']:
            if shot_config not in data[results_key]:
                continue
            
            print(f"  {shot_config}...", end='')
            
            shot_data = data[results_key][shot_config]
            
            try:
                y_true, y_pred = extract_predictions_from_json(shot_data, task_type)
                boot_result = self.bootstrap.bootstrap_metric(y_true, y_pred, average='micro')
                
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
                
                print(f" {boot_result.mean:.3f} [{boot_result.ci_lower:.3f}, {boot_result.ci_upper:.3f}]")
            
            except Exception as e:
                print(f" Error: {e}")
                continue
        
        return results
    
    def analyze_task(self, task: str) -> pd.DataFrame:
        """
        Analyze all result files for a task
        
        Args:
            task: 'artifact' or 'param'
        """
        pattern = f"*{task}*.json"
        json_files = list(self.results_dir.glob(pattern))
        
        if not json_files:
            print(f"Warning: No files found matching pattern {pattern} in {self.results_dir}")
            return pd.DataFrame()
        
        print(f"\nFound {len(json_files)} result files for {task}")
        
        all_results = []
        for json_file in json_files:
            try:
                results = self.process_result_file(json_file)
                all_results.extend(results)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        return pd.DataFrame(all_results)
    
    def run_analysis(self):
        """
        Run complete batch analysis
        """
        print("="*80)
        print("BATCH BOOTSTRAP ANALYSIS")
        print("="*80)
        print(f"Results directory: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Bootstrap iterations: {self.n_bootstrap}")
        
        # Analyze artifact detection
        print("\n" + "="*80)
        print("ARTIFACT DETECTION")
        print("="*80)
        artifact_df = self.analyze_task('artifact')
        
        if not artifact_df.empty:
            csv_file = self.output_dir / "artifact_bootstrap.csv"
            artifact_df.to_csv(csv_file, index=False)
            print(f"\nSaved: {csv_file}")
        
        # Analyze parameter recommendation
        print("\n" + "="*80)
        print("PARAMETER RECOMMENDATION")
        print("="*80)
        param_df = self.analyze_task('param')
        
        if not param_df.empty:
            csv_file = self.output_dir / "param_bootstrap.csv"
            param_df.to_csv(csv_file, index=False)
            print(f"\nSaved: {csv_file}")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        if not artifact_df.empty:
            print("\nArtifact Detection:")
            print(f"  Average CI Width: {artifact_df['CI_Width'].mean():.4f}")
            print(f"  Average Std: {artifact_df['Std'].mean():.4f}")
        
        if not param_df.empty:
            print("\nParameter Recommendation:")
            print(f"  Average CI Width: {param_df['CI_Width'].mean():.4f}")
            print(f"  Average Std: {param_df['Std'].mean():.4f}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch bootstrap analysis for PtychoBench results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default='./bootstrap_output',
                       help='Output directory for analysis results')
    parser.add_argument('--n_bootstrap', type=int, default=10000,
                       help='Number of bootstrap iterations (default: 10000)')
    
    args = parser.parse_args()
    
    analyzer = BatchBootstrapAnalyzer(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        n_bootstrap=args.n_bootstrap
    )
    
    analyzer.run_analysis()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
