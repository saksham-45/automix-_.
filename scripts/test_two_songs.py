#!/usr/bin/env python3
"""
Test AI DJ transition prediction on two audio files.

Usage:
    python scripts/test_two_songs.py --song-a path/to/song_a.wav --song-b path/to/song_b.wav
    
    # Or with manual features (if audio analysis not available)
    python scripts/test_two_songs.py --song-a-features tempo:128,key:Am,energy:0.7 --song-b-features tempo:130,key:Em,energy:0.8
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.inference import DJTransitionPredictor, visualize_curves
import json


def parse_features(feature_string: str) -> dict:
    """Parse features from string like 'tempo:128,key:Am,energy:0.7'"""
    features = {}
    for item in feature_string.split(','):
        if ':' in item:
            key, value = item.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Try to convert to appropriate type
            if key == 'tempo':
                features[key] = float(value)
            elif key == 'energy':
                features[key] = float(value)
            else:
                features[key] = value
    
    return features


def main():
    parser = argparse.ArgumentParser(description='Test AI DJ on two songs')
    parser.add_argument('--song-a', type=str, help='Path to song A (outgoing) audio file')
    parser.add_argument('--song-b', type=str, help='Path to song B (incoming) audio file')
    parser.add_argument('--song-a-features', type=str, 
                       help='Song A features: tempo:128,key:Am,energy:0.7')
    parser.add_argument('--song-b-features', type=str,
                       help='Song B features: tempo:130,key:Em,energy:0.8')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Visualize generated curves')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Save prediction to JSON file')
    parser.add_argument('--model-dir', default='models/',
                       help='Directory containing model checkpoints')
    
    args = parser.parse_args()
    
    # Initialize predictor
    model_dir = Path(args.model_dir)
    decision_nn_path = model_dir / 'decision_nn.pt'
    curve_lstm_path = model_dir / 'curve_lstm.pt'
    
    print("="*60)
    print("AI DJ TRANSITION TEST")
    print("="*60)
    print("\nLoading models...")
    
    predictor = DJTransitionPredictor(
        decision_nn_path=str(decision_nn_path) if decision_nn_path.exists() else None,
        curve_lstm_path=str(curve_lstm_path) if curve_lstm_path.exists() else None
    )
    
    # Get song features
    song_a_features = None
    song_b_features = None
    
    if args.song_a and args.song_b:
        # Use audio files
        print(f"\nAnalyzing audio files...")
        print(f"  Song A: {Path(args.song_a).name}")
        print(f"  Song B: {Path(args.song_b).name}")
        
        try:
            result = predictor.predict_from_audio(args.song_a, args.song_b)
            if 'error' in result:
                print(f"\n⚠ Error analyzing audio: {result['error']}")
                print("Try using manual features instead with --song-a-features and --song-b-features")
                return
        except Exception as e:
            print(f"\n⚠ Error: {e}")
            print("Try using manual features instead with --song-a-features and --song-b-features")
            return
    
    elif args.song_a_features and args.song_b_features:
        # Use manual features
        song_a_features = parse_features(args.song_a_features)
        song_b_features = parse_features(args.song_b_features)
        
        print(f"\nUsing manual features:")
        print(f"  Song A: {song_a_features}")
        print(f"  Song B: {song_b_features}")
        
        result = predictor.predict_from_features(song_a_features, song_b_features, 
                                                generate_curves=True)
    
    else:
        print("\n⚠ Please provide either:")
        print("  --song-a and --song-b (audio files)")
        print("  OR")
        print("  --song-a-features and --song-b-features (manual features)")
        print("\nExample:")
        print("  python scripts/test_two_songs.py --song-a song1.wav --song-b song2.wav")
        print("  python scripts/test_two_songs.py --song-a-features tempo:128,key:Am,energy:0.7 --song-b-features tempo:130,key:Em,energy:0.8")
        return
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"\n🎵 Transition Decision:")
    print(f"  Technique: {result.get('technique', 'unknown').upper()}")
    print(f"  Confidence: {result.get('technique_confidence', 0)*100:.1f}%")
    print(f"  Duration: {result.get('duration_bars', 5)} bars")
    print(f"  Bass Swap: {'YES' if result.get('bass_swap') else 'NO'}")
    print(f"  Low Cut (Incoming): {'YES' if result.get('low_cut_incoming') else 'NO'}")
    print(f"  High Cut (Outgoing): {'YES' if result.get('high_cut_outgoing') else 'NO'}")
    print(f"  Key Compatible: {'YES' if result.get('key_compatible') else 'NO'}")
    
    if 'curves' in result:
        curves = result['curves']
        print(f"\n🎛️  Automation Curves Generated:")
        print(f"  Frames: {len(curves.get('time', []))}")
        print(f"  Volume A: {min(curves.get('volume_a', [0])):.2f} → {max(curves.get('volume_a', [1])):.2f}")
        print(f"  Volume B: {min(curves.get('volume_b', [0])):.2f} → {max(curves.get('volume_b', [1])):.2f}")
        print(f"  Duration: {max(curves.get('time', [0])):.1f} seconds")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Saved results to: {output_path}")
    
    # Visualize
    if args.visualize and 'curves' in result:
        print("\n📊 Generating visualization...")
        viz_path = args.output.replace('.json', '.png') if args.output else 'transition_curves.png'
        visualize_curves(result, output_path=viz_path)
        print(f"✓ Saved visualization to: {viz_path}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

