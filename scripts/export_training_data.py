#!/usr/bin/env python3
"""
Export training data from mix analyses for ML model training.

This script converts transition analyses into ML-ready training examples.
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training_data_extractor import TrainingDataExtractor
from src.database import MusicDatabase


def main():
    parser = argparse.ArgumentParser(
        description='Export training data from mix analyses'
    )
    parser.add_argument(
        'mix_analysis_path',
        type=str,
        help='Path to mix analysis JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/training/transitions',
        help='Directory to save training examples'
    )
    parser.add_argument(
        '--song-analyses-dir',
        type=str,
        default='data/analyses',
        help='Directory containing song analyses'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/music_analysis.db',
        help='Database path (optional, for storing metadata)'
    )
    
    args = parser.parse_args()
    
    # Load mix analysis
    print(f"Loading mix analysis: {args.mix_analysis_path}")
    with open(args.mix_analysis_path, 'r') as f:
        mix_analysis = json.load(f)
    
    # Initialize extractor
    extractor = TrainingDataExtractor(
        song_analyses_dir=args.song_analyses_dir
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract training data for each transition
    print(f"\nExtracting training data from {len(mix_analysis.get('analyzed_transitions', []))} transitions...")
    
    training_examples = []
    
    for i, transition_data in enumerate(mix_analysis.get('analyzed_transitions', [])):
        print(f"\nProcessing transition {i+1}...")
        
        try:
            # Extract training example
            training_example = extractor.extract_training_data(
                mix_analysis=mix_analysis,
                transition_data=transition_data,
                transition_index=i
            )
            
            if training_example is None:
                print(f"  ⚠ Skipped (missing data)")
                continue
            
            # Save training example
            transition_id = training_example['metadata']['transition_id']
            output_path = output_dir / f"{transition_id}.json"
            
            with open(output_path, 'w') as f:
                json.dump(training_example, f, indent=2)
            
            training_examples.append(training_example)
            print(f"  ✓ Saved: {output_path.name}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Save summary
    summary = {
        "mix_id": mix_analysis.get('mix_id'),
        "total_transitions": len(mix_analysis.get('analyzed_transitions', [])),
        "successful_extractions": len(training_examples),
        "training_examples": [ex['metadata']['transition_id'] for ex in training_examples]
    }
    
    summary_path = output_dir / f"{mix_analysis.get('mix_id', 'unknown')}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training data export complete!")
    print(f"  Total transitions: {summary['total_transitions']}")
    print(f"  Successful extractions: {summary['successful_extractions']}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

