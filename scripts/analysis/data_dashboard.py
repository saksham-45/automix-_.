#!/usr/bin/env python3
"""
Data Collection Dashboard

Shows current progress toward training data goals.
Usage:
    python scripts/data_dashboard.py
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_all_analyses():
    """Load all analysis files and return samples."""
    data_dirs = [Path('data'), Path('data/youtube_mixes'), Path('data/premixed_albums')]
    
    all_samples = []
    sources = {}
    
    for d in data_dirs:
        if d.exists():
            for fpath in d.glob('*_analysis.json'):
                try:
                    with open(fpath) as file:
                        transitions = json.load(file)
                    
                    if isinstance(transitions, list):
                        name = fpath.stem.replace('_analysis', '')[:40]
                        sources[name] = len(transitions)
                        all_samples.extend(transitions)
                except Exception as e:
                    pass
    
    return all_samples, sources


def calculate_technique_stats(samples):
    """Calculate technique usage statistics."""
    techniques = defaultdict(int)
    
    for s in samples:
        for t in s.get('techniques_used', []):
            techniques[t] += 1
    
    return dict(techniques)


def calculate_quality_stats(samples):
    """Calculate quality metrics."""
    if not samples:
        return {}
    
    smoothness = [s.get('perceived_smoothness', 0) for s in samples]
    
    bass_swaps = sum(1 for s in samples if s.get('bass_swap_detected') in [True, 'True'])
    beat_aligned = sum(1 for s in samples if s.get('beat_aligned') in [True, 'True'])
    key_compat = sum(1 for s in samples if s.get('key_compatible') == True)
    
    return {
        'avg_smoothness': sum(smoothness) / len(smoothness) if smoothness else 0,
        'bass_swap_pct': bass_swaps / len(samples) * 100 if samples else 0,
        'beat_aligned_pct': beat_aligned / len(samples) * 100 if samples else 0,
        'key_compat_pct': key_compat / len(samples) * 100 if samples else 0,
    }


def estimate_genre(source_name):
    """Estimate genre from source name."""
    name_lower = source_name.lower()
    
    if any(x in name_lower for x in ['heroes', 'villains', 'metro', 'trap']):
        return 'Hip-hop/Trap'
    elif any(x in name_lower for x in ['boiler', 'techno', 'house']):
        return 'Electronic'
    elif any(x in name_lower for x in ['dubstep', 'skream', 'benga']):
        return 'Dubstep'
    elif any(x in name_lower for x in ['syber', 'baby_j', 'baile', 'jersey']):
        return 'Baile/Jersey'
    elif any(x in name_lower for x in ['lex', 'rb', 'hip hop']):
        return 'Hip-hop/R&B'
    elif any(x in name_lower for x in ['yusuke', 'tokyo', 'osaka']):
        return 'Electronic/J-Music'
    else:
        return 'Mixed'


def print_dashboard():
    """Print the data collection dashboard."""
    samples, sources = load_all_analyses()
    
    # Targets
    TARGET_NN = 500
    TARGET_LSTM = 2000
    
    total = len(samples)
    
    print()
    print("╔" + "═"*68 + "╗")
    print("║" + " AI DJ TRAINING DATA DASHBOARD ".center(68) + "║")
    print("╠" + "═"*68 + "╣")
    
    # Progress bars
    nn_pct = min(100, total / TARGET_NN * 100)
    lstm_pct = min(100, total / TARGET_LSTM * 100)
    
    nn_bar = "█" * int(nn_pct / 5) + "░" * (20 - int(nn_pct / 5))
    lstm_bar = "█" * int(lstm_pct / 5) + "░" * (20 - int(lstm_pct / 5))
    
    print("║" + f"  TOTAL TRANSITIONS: {total:,}".ljust(68) + "║")
    print("║" + "".ljust(68) + "║")
    print("║" + f"  Neural Network Target (500):  [{nn_bar}] {nn_pct:5.1f}%".ljust(68) + "║")
    print("║" + f"  LSTM Target (2000):           [{lstm_bar}] {lstm_pct:5.1f}%".ljust(68) + "║")
    
    print("╠" + "═"*68 + "╣")
    print("║" + " DATA SOURCES ".center(68) + "║")
    print("╠" + "═"*68 + "╣")
    
    # Sources by count
    for name, count in sorted(sources.items(), key=lambda x: -x[1]):
        genre = estimate_genre(name)
        line = f"  {count:3} │ {name[:35]:<35} │ {genre}"
        print("║" + line.ljust(68) + "║")
    
    print("╠" + "═"*68 + "╣")
    print("║" + " TECHNIQUE DISTRIBUTION ".center(68) + "║")
    print("╠" + "═"*68 + "╣")
    
    techniques = calculate_technique_stats(samples)
    for tech, count in sorted(techniques.items(), key=lambda x: -x[1])[:8]:
        pct = count / total * 100 if total else 0
        bar = "▓" * int(pct / 5)
        line = f"  {tech:25} {count:4} ({pct:4.1f}%) {bar}"
        print("║" + line.ljust(68) + "║")
    
    print("╠" + "═"*68 + "╣")
    print("║" + " QUALITY METRICS ".center(68) + "║")
    print("╠" + "═"*68 + "╣")
    
    stats = calculate_quality_stats(samples)
    print("║" + f"  Average Smoothness:   {stats.get('avg_smoothness', 0)*100:5.1f}%".ljust(68) + "║")
    print("║" + f"  Bass Swap Usage:      {stats.get('bass_swap_pct', 0):5.1f}%".ljust(68) + "║")
    print("║" + f"  Beat Aligned:         {stats.get('beat_aligned_pct', 0):5.1f}%".ljust(68) + "║")
    print("║" + f"  Key Compatible:       {stats.get('key_compat_pct', 0):5.1f}%".ljust(68) + "║")
    
    print("╠" + "═"*68 + "╣")
    print("║" + " NEXT STEPS ".center(68) + "║")
    print("╠" + "═"*68 + "╣")
    
    if total < TARGET_NN:
        needed = TARGET_NN - total
        mixes_needed = needed // 25 + 1
        print("║" + f"  → Need {needed} more transitions for NN training".ljust(68) + "║")
        print("║" + f"  → Approximately {mixes_needed} more mixes with tracklists".ljust(68) + "║")
        print("║" + "".ljust(68) + "║")
        print("║" + "  Run: python scripts/add_mix_with_tracklist.py".ljust(68) + "║")
    elif total < TARGET_LSTM:
        needed = TARGET_LSTM - total
        mixes_needed = needed // 25 + 1
        print("║" + f"  ✓ Ready for NN training! ({total}/{TARGET_NN})".ljust(68) + "║")
        print("║" + f"  → Need {needed} more transitions for LSTM".ljust(68) + "║")
        print("║" + f"  → Approximately {mixes_needed} more mixes".ljust(68) + "║")
    else:
        print("║" + f"  ✓ Ready for NN training! ({total}/{TARGET_NN})".ljust(68) + "║")
        print("║" + f"  ✓ Ready for LSTM training! ({total}/{TARGET_LSTM})".ljust(68) + "║")
        print("║" + "".ljust(68) + "║")
        print("║" + "  Run: python scripts/train_model.py".ljust(68) + "║")
    
    print("╚" + "═"*68 + "╝")
    print()
    
    # Return data for programmatic use
    return {
        'total': total,
        'sources': sources,
        'nn_ready': total >= TARGET_NN,
        'lstm_ready': total >= TARGET_LSTM,
        'techniques': techniques,
        'quality': stats
    }


def export_summary(output_path: str = 'data/dataset_summary.json'):
    """Export dataset summary to JSON."""
    samples, sources = load_all_analyses()
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_transitions': len(samples),
        'sources': sources,
        'technique_counts': calculate_technique_stats(samples),
        'quality_metrics': calculate_quality_stats(samples),
        'nn_ready': len(samples) >= 500,
        'lstm_ready': len(samples) >= 2000,
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary exported to: {output_path}")
    return summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Collection Dashboard')
    parser.add_argument('--export', '-e', action='store_true', 
                       help='Export summary to JSON')
    parser.add_argument('--json', '-j', action='store_true',
                       help='Output as JSON instead of dashboard')
    
    args = parser.parse_args()
    
    if args.export:
        export_summary()
    elif args.json:
        samples, sources = load_all_analyses()
        print(json.dumps({
            'total': len(samples),
            'sources': sources,
            'nn_ready': len(samples) >= 500,
            'lstm_ready': len(samples) >= 2000,
        }, indent=2))
    else:
        print_dashboard()

