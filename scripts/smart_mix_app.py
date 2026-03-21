#!/usr/bin/env python3
"""
Smart Song Mixer App

Simple interface to select two songs and create smooth, beat-matched transitions.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.smart_mixer import SmartMixer
from src.song_analyzer import SongAnalyzer
from scripts.inference import DJTransitionPredictor
import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description='Smart Song Mixer - Create smooth transitions')
    parser.add_argument('--song-a', type=str, required=True,
                       help='Path to Song A (outgoing)')
    parser.add_argument('--song-b', type=str, required=True,
                       help='Path to Song B (incoming)')
    parser.add_argument('--output', '-o', type=str, default='smooth_mix.wav',
                       help='Output mix file')
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Transition duration in seconds (default: 30)')
    parser.add_argument('--use-ai', action='store_true',
                       help='Use AI model for volume curves')
    parser.add_argument('--model-dir', type=str, default='models/',
                       help='Directory containing AI models')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SMART SONG MIXER")
    print("="*60)
    print(f"\nSong A: {Path(args.song_a).name}")
    print(f"Song B: {Path(args.song_b).name}")
    print(f"Transition duration: {args.duration}s")
    
    # Initialize components
    mixer = SmartMixer()
    
    # Optional: Analyze songs for better compatibility scoring
    song_a_analysis = None
    song_b_analysis = None
    ai_data = None
    
    if args.use_ai:
        print("\nUsing AI model for transition curves...")
        analyzer = SongAnalyzer()
        print("Analyzing Song A...")
        song_a_analysis = analyzer.analyze(args.song_a)
        print("Analyzing Song B...")
        song_b_analysis = analyzer.analyze(args.song_b)
        
        # Get AI prediction
        model_dir = Path(args.model_dir)
        predictor = DJTransitionPredictor(
            decision_nn_path=str(model_dir / 'decision_nn.pt') if (model_dir / 'decision_nn.pt').exists() else None,
            curve_lstm_path=str(model_dir / 'curve_lstm.pt') if (model_dir / 'curve_lstm.pt').exists() else None
        )
        
        # Extract features for AI
        song_a_features = {
            'tempo': song_a_analysis.get('tempo', {}).get('bpm', 120),
            'key': song_a_analysis.get('harmony', {}).get('key', 'C'),
            'energy': song_a_analysis.get('energy', {}).get('overall_energy', 0.5),
        }
        song_b_features = {
            'tempo': song_b_analysis.get('tempo', {}).get('bpm', 120),
            'key': song_b_analysis.get('harmony', {}).get('key', 'C'),
            'energy': song_b_analysis.get('energy', {}).get('overall_energy', 0.5),
        }
        
        ai_data = predictor.predict_from_features(song_a_features, song_b_features, generate_curves=True)
    
    # Create smooth mix
    print("\n" + "="*60)
    print("CREATING SMOOTH TRANSITION")
    print("="*60)
    
    mixed_audio = mixer.create_smooth_mix(
        args.song_a,
        args.song_b,
        transition_duration=args.duration,
        song_a_analysis=song_a_analysis,
        song_b_analysis=song_b_analysis,
        ai_transition_data=ai_data
    )
    
    # Save
    print(f"\nSaving mix to: {args.output}")
    sf.write(args.output, mixed_audio, mixer.sr)
    
    duration = len(mixed_audio) / mixer.sr
    print(f"\n{'='*60}")
    print("✅ SMOOTH MIX COMPLETE!")
    print(f"{'='*60}")
    print(f"\n🎵 File: {args.output}")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"\n🎧 Features:")
    print(f"   ✓ Optimal transition points found automatically")
    print(f"   ✓ Beat-matched alignment")
    print(f"   ✓ Smooth, gradual volume curves")
    print(f"\n🚀 Play it: open {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
