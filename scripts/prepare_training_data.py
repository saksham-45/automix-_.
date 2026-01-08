#!/usr/bin/env python3
"""
Prepare training data with proper splits and preprocessing.

Splits data into:
- Train: 70% (1,428 samples)
- Validation: 15% (306 samples)
- Test: 15% (306 samples)
"""
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_validated_samples():
    """Load validated samples."""
    path = Path('data/validated_training_samples.json')
    if path.exists():
        with open(path) as f:
            return json.load(f)
    
    # Fallback: load directly
    from scripts.validate_training_data import load_all_samples, validate_sample
    samples = load_all_samples()
    valid_samples = [s for s in samples if validate_sample(s)[0]]
    return valid_samples


def estimate_genre(source_name: str) -> str:
    """Estimate genre from source name for stratification."""
    name_lower = source_name.lower()
    
    if any(x in name_lower for x in ['syber', 'baby_j', 'baile', 'jersey']):
        return 'Baile/Jersey'
    elif any(x in name_lower for x in ['boiler', 'techno', 'house', 'electronic']):
        return 'Electronic'
    elif any(x in name_lower for x in ['hip', 'rap', 'trap', 'rnb', 'rb']):
        return 'Hip-hop/R&B'
    elif any(x in name_lower for x in ['reggaeton', 'dancehall', 'afrobeats', 'amapiano']):
        return 'World/Global'
    else:
        return 'Mixed'


def get_sample_genre(sample: dict) -> str:
    """Get genre for a sample based on analysis file source."""
    # Try to infer from sample keys or default to Mixed
    return estimate_genre(sample.get('from_track', '') + ' ' + sample.get('to_track', ''))


def split_data(samples: list, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
    """
    Split data into train/val/test sets with stratification.
    
    Args:
        samples: List of sample dictionaries
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        (train, val, test) lists
    """
    # Create genre labels for stratification
    genres = [get_sample_genre(s) for s in samples]
    
    # First split: train + (val + test)
    train_ratio = 1 - test_size - val_size
    val_plus_test_ratio = test_size + val_size
    
    X = np.arange(len(samples))
    
    X_train, X_temp = train_test_split(
        X,
        test_size=val_plus_test_ratio,
        stratify=genres,
        random_state=random_state
    )
    
    # Second split: val and test
    val_in_temp = val_size / val_plus_test_ratio
    
    X_val, X_test = train_test_split(
        X_temp,
        test_size=(1 - val_in_temp),
        stratify=[genres[i] for i in X_temp],
        random_state=random_state
    )
    
    train_samples = [samples[i] for i in X_train]
    val_samples = [samples[i] for i in X_val]
    test_samples = [samples[i] for i in X_test]
    
    return train_samples, val_samples, test_samples


def main():
    print("="*60)
    print("PREPARING TRAINING DATA")
    print("="*60)
    
    # Load validated samples
    print("\nLoading validated samples...")
    samples = load_validated_samples()
    print(f"Loaded {len(samples)} samples")
    
    if len(samples) == 0:
        print("ERROR: No samples found!")
        return
    
    # Split data
    print("\nSplitting data (70/15/15)...")
    train, val, test = split_data(samples, test_size=0.15, val_size=0.15)
    
    print(f"  Train: {len(train)} samples ({len(train)/len(samples)*100:.1f}%)")
    print(f"  Validation: {len(val)} samples ({len(val)/len(samples)*100:.1f}%)")
    print(f"  Test: {len(test)} samples ({len(test)/len(samples)*100:.1f}%)")
    
    # Show genre distribution
    def count_genres(sample_list):
        genres = {}
        for s in sample_list:
            genre = get_sample_genre(s)
            genres[genre] = genres.get(genre, 0) + 1
        return genres
    
    print("\nGenre distribution:")
    print("  Train:", count_genres(train))
    print("  Val:  ", count_genres(val))
    print("  Test: ", count_genres(test))
    
    # Save splits
    output_dir = Path('data/training_splits')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'train.json', 'w') as f:
        json.dump(train, f, indent=2, default=str)
    
    with open(output_dir / 'val.json', 'w') as f:
        json.dump(val, f, indent=2, default=str)
    
    with open(output_dir / 'test.json', 'w') as f:
        json.dump(test, f, indent=2, default=str)
    
    print(f"\n✓ Saved splits to: {output_dir}/")
    print(f"  - train.json ({len(train)} samples)")
    print(f"  - val.json ({len(val)} samples)")
    print(f"  - test.json ({len(test)} samples)")
    
    return train, val, test


if __name__ == '__main__':
    main()

