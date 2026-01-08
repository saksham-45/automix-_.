#!/usr/bin/env python3
"""
Deep analyze transitions with manually provided timestamps.

Usage:
    python scripts/deep_analyze_transitions.py audio.wav --transitions transitions.txt
    python scripts/deep_analyze_transitions.py audio.wav  # Uses built-in transitions
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deep_transition_analyzer import (
    DeepTransitionAnalyzer, 
    TransitionPoint,
    parse_transitions_from_text
)


# Heroes & Villains transitions (from user)
HEROES_VILLAINS_TRANSITIONS = """
0:00 Around Me - Metro Boomin, Don Toliver
3:10 Superhero - Metro Boomin, Future
5:09 Trance - Metro Boomin, Travis Scott, Young Thug
8:04 Lock On Me - Metro Boomin, Travis Scott, Future
10:54 Too Many Nights - Metro Boomin, Don Toliver, Future
12:54 The Magic Piano
14:00 Niagara Falls - Metro Boomin, Travis Scott, 21 Savage
"""


def main():
    parser = argparse.ArgumentParser(
        description="Deep analyze transitions with known timestamps"
    )
    parser.add_argument('audio_path', help="Path to audio file")
    parser.add_argument('--transitions', '-t', help="File with transition timestamps")
    parser.add_argument('--output', '-o', default='data/deep_transition_analysis.json',
                       help="Output JSON path")
    parser.add_argument('--context', '-c', type=float, default=10.0,
                       help="Seconds of context before/after transition")
    
    args = parser.parse_args()
    
    # Load transitions
    if args.transitions:
        with open(args.transitions, 'r') as f:
            text = f.read()
        transitions = parse_transitions_from_text(text)
    else:
        # Use built-in Heroes & Villains transitions
        transitions = parse_transitions_from_text(HEROES_VILLAINS_TRANSITIONS)
    
    print(f"\nLoaded {len(transitions)} transitions to analyze:")
    for i, t in enumerate(transitions, 1):
        print(f"  {i}. {t.time_sec/60:.2f}min: {t.from_track} → {t.to_track}")
    print()
    
    # Run deep analysis
    analyzer = DeepTransitionAnalyzer()
    analyses = analyzer.analyze_transitions(
        args.audio_path,
        transitions,
        context_before_sec=args.context,
        context_after_sec=args.context
    )
    
    # Export
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    analyzer.export_training_data(analyses, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL TRANSITIONS")
    print("="*60)
    
    techniques_count = {}
    for a in analyses:
        for t in a.techniques_used:
            techniques_count[t] = techniques_count.get(t, 0) + 1
    
    print("\nMost Used Techniques:")
    for technique, count in sorted(techniques_count.items(), key=lambda x: -x[1]):
        print(f"  {technique}: {count} times")
    
    print(f"\nAverage Smoothness: {sum(a.perceived_smoothness for a in analyses)/len(analyses):.0%}")
    print(f"Average Complexity: {sum(a.technique_complexity for a in analyses)/len(analyses):.0%}")
    
    print(f"\nBass Swap Used: {sum(1 for a in analyses if a.bass_swap_detected)}/{len(analyses)} transitions")
    print(f"Beat Aligned: {sum(1 for a in analyses if a.beat_aligned)}/{len(analyses)} transitions")
    print(f"Key Compatible: {sum(1 for a in analyses if a.key_compatible)}/{len(analyses)} transitions")
    
    # Clean up audio file after analysis
    audio_path = Path(args.audio_path)
    if audio_path.exists() and audio_path.suffix == '.wav':
        try:
            audio_path.unlink()
            print(f"\n✓ Cleaned up: {audio_path.name}")
        except Exception as e:
            print(f"\n⚠ Could not delete {audio_path.name}: {e}")


if __name__ == '__main__':
    main()

