#!/usr/bin/env python3
"""
Validate training data quality and completeness.

Checks:
- Missing/NaN values
- Feature distributions
- Outliers
- Required fields present
"""
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.decision_nn import prepare_features, prepare_targets


def load_all_samples():
    """Load all training samples from analysis files."""
    data_dirs = [Path('data'), Path('data/youtube_mixes'), Path('data/premixed_albums')]
    
    all_samples = []
    
    for d in data_dirs:
        if d.exists():
            for fpath in d.glob('*_analysis.json'):
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_samples.extend(data)
                except Exception as e:
                    print(f"Warning: Could not load {fpath}: {e}")
    
    return all_samples


def validate_sample(sample: dict) -> tuple[bool, list[str]]:
    """Validate a single sample. Returns (is_valid, errors)."""
    errors = []
    
    # Required fields
    required = ['tempo_outgoing', 'tempo_incoming', 'energy_before', 'energy_after']
    for field in required:
        if field not in sample:
            errors.append(f"Missing required field: {field}")
        elif sample[field] is None:
            errors.append(f"Null value in field: {field}")
    
    # Check numeric fields
    numeric_fields = ['tempo_outgoing', 'tempo_incoming', 'energy_before', 
                     'energy_after', 'harmonic_tension', 'spectral_smoothness']
    
    def parse_numeric(val):
        """Parse numeric value from various formats."""
        if isinstance(val, (int, float)):
            return float(val)
        elif isinstance(val, list):
            return float(val[0]) if len(val) > 0 else None
        elif isinstance(val, str):
            return float(val.strip('[]'))
        return None
    
    for field in numeric_fields:
        if field in sample:
            try:
                val = parse_numeric(sample[field])
                if val is None:
                    errors.append(f"Could not parse {field}: {sample[field]}")
                elif np.isnan(val) or np.isinf(val):
                    errors.append(f"Invalid numeric value in {field}: {val}")
            except (ValueError, TypeError) as e:
                errors.append(f"Non-numeric value in {field}: {sample[field]}")
    
    # Check key fields
    if 'key_outgoing' not in sample or not sample['key_outgoing']:
        errors.append("Missing key_outgoing")
    if 'key_incoming' not in sample or not sample['key_incoming']:
        errors.append("Missing key_incoming")
    
    return len(errors) == 0, errors


def analyze_feature_distributions(samples: list):
    """Analyze feature distributions and identify outliers."""
    stats = defaultdict(list)
    
    for sample in samples:
        try:
            # Parse tempo
            def parse_tempo(t):
                if isinstance(t, str):
                    return float(t.strip('[]'))
                elif isinstance(t, list):
                    return float(t[0])
                return float(t)
            
            stats['tempo_outgoing'].append(parse_tempo(sample.get('tempo_outgoing', 120)))
            stats['tempo_incoming'].append(parse_tempo(sample.get('tempo_incoming', 120)))
            stats['energy_before'].append(sample.get('energy_before', 0.5))
            stats['energy_after'].append(sample.get('energy_after', 0.5))
            
            if 'harmonic_tension' in sample:
                stats['harmonic_tension'].append(sample['harmonic_tension'])
        except:
            pass
    
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTIONS")
    print("="*60)
    
    for field, values in stats.items():
        if values:
            arr = np.array(values)
            print(f"\n{field}:")
            print(f"  Mean: {np.mean(arr):.3f}")
            print(f"  Std:  {np.std(arr):.3f}")
            print(f"  Min:  {np.min(arr):.3f}")
            print(f"  Max:  {np.max(arr):.3f}")
            print(f"  Median: {np.median(arr):.3f}")
            
            # Check for outliers (beyond 3 std)
            mean = np.mean(arr)
            std = np.std(arr)
            outliers = np.sum(np.abs(arr - mean) > 3 * std)
            if outliers > 0:
                print(f"  ⚠ {outliers} outliers (>3σ)")
            
            # Check reasonable ranges
            if field.startswith('tempo'):
                if np.any(arr < 60) or np.any(arr > 200):
                    print(f"  ⚠ Some values outside typical range (60-200 BPM)")


def main():
    print("="*60)
    print("VALIDATING TRAINING DATA")
    print("="*60)
    
    # Load all samples
    print("\nLoading samples...")
    samples = load_all_samples()
    print(f"Loaded {len(samples)} samples")
    
    if len(samples) == 0:
        print("ERROR: No samples found!")
        return
    
    # Validate each sample
    print("\nValidating samples...")
    valid_samples = []
    invalid_samples = []
    
    for i, sample in enumerate(samples):
        is_valid, errors = validate_sample(sample)
        if is_valid:
            valid_samples.append(sample)
        else:
            invalid_samples.append((i, errors))
    
    print(f"\n✓ Valid samples: {len(valid_samples)}")
    print(f"✗ Invalid samples: {len(invalid_samples)}")
    
    if invalid_samples:
        print("\nInvalid samples (first 10):")
        for idx, errors in invalid_samples[:10]:
            print(f"  Sample {idx}: {', '.join(errors[:3])}")
    
    # Analyze feature distributions
    analyze_feature_distributions(valid_samples)
    
    # Test feature extraction
    print("\n" + "="*60)
    print("TESTING FEATURE EXTRACTION")
    print("="*60)
    
    successful = 0
    failed = 0
    
    for sample in valid_samples[:100]:  # Test first 100
        try:
            features, key_a, key_b = prepare_features(sample)
            targets = prepare_targets(sample)
            
            # Verify shapes
            assert features.shape == (13,), f"Wrong feature shape: {features.shape}"
            assert isinstance(key_a, int) and 0 <= key_a < 12, f"Invalid key_a: {key_a}"
            assert isinstance(key_b, int) and 0 <= key_b < 12, f"Invalid key_b: {key_b}"
            
            successful += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  ✗ Failed to extract features: {e}")
    
    print(f"\nFeature extraction:")
    print(f"  ✓ Successful: {successful}/100")
    print(f"  ✗ Failed: {failed}/100")
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total samples: {len(samples)}")
    print(f"Valid samples: {len(valid_samples)} ({len(valid_samples)/len(samples)*100:.1f}%)")
    print(f"Invalid samples: {len(invalid_samples)} ({len(invalid_samples)/len(samples)*100:.1f}%)")
    
    if len(valid_samples) >= 1500:
        print("\n✓ SUFFICIENT DATA FOR TRAINING")
        print(f"  Ready for NN: {len(valid_samples) >= 500}")
        print(f"  Ready for LSTM: {len(valid_samples) >= 2000}")
    else:
        print(f"\n⚠ May need more data (have {len(valid_samples)}, need 2000 for LSTM)")
    
    # Save valid samples
    output_path = Path('data/validated_training_samples.json')
    with open(output_path, 'w') as f:
        json.dump(valid_samples, f, indent=2, default=str)
    
    print(f"\n✓ Saved {len(valid_samples)} valid samples to: {output_path}")
    
    return valid_samples


if __name__ == '__main__':
    main()

