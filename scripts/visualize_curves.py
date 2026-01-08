#!/usr/bin/env python3
"""
Visualize predicted vs actual automation curves.

Plots volume, EQ, and filter curves for transition analysis.
"""
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_curves(sample: Dict) -> Dict[str, np.ndarray]:
    """Extract curves from a sample."""
    curves = {}
    
    # Volume curves
    if 'volume_curve_a' in sample:
        curves['volume_a'] = np.array(sample['volume_curve_a'])
    if 'volume_curve_b' in sample:
        curves['volume_b'] = np.array(sample['volume_curve_b'])
    
    # EQ curves
    if 'eq_low_a' in sample:
        curves['eq_low_a'] = np.array(sample['eq_low_a'])
    if 'eq_low_b' in sample:
        curves['eq_low_b'] = np.array(sample['eq_low_b'])
    
    if 'eq_mid_a' in sample:
        curves['eq_mid_a'] = np.array(sample['eq_mid_a'])
    if 'eq_mid_b' in sample:
        curves['eq_mid_b'] = np.array(sample['eq_mid_b'])
    
    if 'eq_high_a' in sample:
        curves['eq_high_a'] = np.array(sample['eq_high_a'])
    if 'eq_high_b' in sample:
        curves['eq_high_b'] = np.array(sample['eq_high_b'])
    
    # Filter curves
    if 'filter_freq' in sample:
        curves['filter_freq'] = np.array(sample['filter_freq'])
    if 'filter_res' in sample:
        curves['filter_res'] = np.array(sample['filter_res'])
    
    return curves


def plot_curves(actual: Dict[str, np.ndarray],
                predicted: Dict[str, np.ndarray] = None,
                duration_sec: float = None,
                output_path: Path = None):
    """
    Plot automation curves.
    
    Args:
        actual: Ground truth curves
        predicted: Predicted curves (optional)
        duration_sec: Duration in seconds
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Transition Automation Curves', fontsize=14, fontweight='bold')
    
    # Determine time axis
    if duration_sec:
        duration = duration_sec
    else:
        # Infer from longest curve
        max_len = max([len(v) for v in actual.values()] if actual else [100])
        duration = max_len * 0.1  # Assume 100ms per frame
    
    # Find maximum length
    max_frames = max([len(v) for v in actual.values()] if actual else [100])
    time = np.linspace(0, duration, max_frames)
    
    # Volume curves
    ax1 = axes[0]
    if 'volume_a' in actual:
        ax1.plot(time[:len(actual['volume_a'])], actual['volume_a'], 
                'b-', label='Volume A (actual)', linewidth=2)
    if 'volume_b' in actual:
        ax1.plot(time[:len(actual['volume_b'])], actual['volume_b'], 
                'r-', label='Volume B (actual)', linewidth=2)
    
    if predicted:
        if 'volume_a' in predicted:
            ax1.plot(time[:len(predicted['volume_a'])], predicted['volume_a'], 
                    'b--', label='Volume A (predicted)', linewidth=1.5, alpha=0.7)
        if 'volume_b' in predicted:
            ax1.plot(time[:len(predicted['volume_b'])], predicted['volume_b'], 
                    'r--', label='Volume B (predicted)', linewidth=1.5, alpha=0.7)
    
    ax1.set_ylabel('Volume (0-1)', fontsize=11)
    ax1.set_title('Volume Crossfade', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # EQ curves
    ax2 = axes[1]
    eq_params = ['eq_low_a', 'eq_low_b', 'eq_mid_a', 'eq_mid_b', 
                 'eq_high_a', 'eq_high_b']
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']
    
    for i, param in enumerate(eq_params):
        if param in actual:
            label = param.replace('_a', ' A').replace('_b', ' B').replace('eq_', '')
            ax2.plot(time[:len(actual[param])], actual[param], 
                    f'{colors[i]}-', label=f'{label} (actual)', linewidth=1.5)
    
    if predicted:
        for i, param in enumerate(eq_params):
            if param in predicted:
                label = param.replace('_a', ' A').replace('_b', ' B').replace('eq_', '')
                ax2.plot(time[:len(predicted[param])], predicted[param], 
                        f'{colors[i]}--', label=f'{label} (pred)', 
                        linewidth=1, alpha=0.6)
    
    ax2.set_ylabel('EQ Gain (normalized)', fontsize=11)
    ax2.set_title('EQ Automation', fontsize=12)
    ax2.legend(loc='best', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Filter curves
    ax3 = axes[2]
    if 'filter_freq' in actual:
        ax3.plot(time[:len(actual['filter_freq'])], actual['filter_freq'], 
                'g-', label='Filter Freq (actual)', linewidth=2)
    if 'filter_res' in actual:
        ax3_twin = ax3.twinx()
        ax3_twin.plot(time[:len(actual['filter_res'])], actual['filter_res'], 
                     'orange', linestyle='-', label='Filter Res (actual)', linewidth=2)
        ax3_twin.set_ylabel('Resonance (0-1)', fontsize=11, color='orange')
        ax3_twin.tick_params(axis='y', labelcolor='orange')
        ax3_twin.legend(loc='upper right')
    
    if predicted:
        if 'filter_freq' in predicted:
            ax3.plot(time[:len(predicted['filter_freq'])], predicted['filter_freq'], 
                    'g--', label='Filter Freq (predicted)', linewidth=1.5, alpha=0.7)
        if 'filter_res' in predicted and 'filter_freq' in actual:
            ax3_twin.plot(time[:len(predicted['filter_res'])], predicted['filter_res'], 
                         'orange', linestyle='--', label='Filter Res (pred)', 
                         linewidth=1.5, alpha=0.7)
    
    ax3.set_xlabel('Time (seconds)', fontsize=11)
    ax3.set_ylabel('Frequency (normalized)', fontsize=11, color='g')
    ax3.set_title('Filter Automation', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='g')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize automation curves')
    parser.add_argument('--sample', '-s', type=str, required=True,
                       help='Path to sample JSON file or index in data file')
    parser.add_argument('--data', '-d', type=str, default='data/training_splits/test.json',
                       help='Path to data file (if sample is an index)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for plot (default: display)')
    
    args = parser.parse_args()
    
    # Load sample
    if Path(args.sample).exists():
        # Direct file
        with open(args.sample) as f:
            sample = json.load(f)
    else:
        # Index in data file
        try:
            idx = int(args.sample)
            with open(args.data) as f:
                samples = json.load(f)
            sample = samples[idx]
            print(f"Loaded sample {idx} from {args.data}")
        except:
            print(f"ERROR: Could not load sample from {args.sample} or as index")
            return
    
    # Extract curves
    actual_curves = load_curves(sample)
    
    if not actual_curves:
        print("ERROR: No curves found in sample")
        return
    
    # Get duration
    duration = sample.get('duration_sec', None)
    if not duration and 'transition_start_sec' in sample and 'transition_end_sec' in sample:
        duration = sample['transition_end_sec'] - sample['transition_start_sec']
    
    # Plot
    output_path = Path(args.output) if args.output else None
    plot_curves(actual_curves, predicted=None, duration_sec=duration, output_path=output_path)


if __name__ == '__main__':
    main()

