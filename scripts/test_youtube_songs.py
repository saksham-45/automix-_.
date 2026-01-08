#!/usr/bin/env python3
"""
Test AI DJ transition on two YouTube videos.

Downloads audio, analyzes both songs, and predicts the transition.
"""
import sys
import argparse
import subprocess
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.inference import DJTransitionPredictor
from src.song_analyzer import SongAnalyzer


def download_youtube_audio(url: str, output_path: Path) -> Path:
    """Download audio from YouTube URL."""
    print(f"Downloading audio from: {url}")
    
    # Use yt-dlp to download audio
    cmd = [
        'yt-dlp',
        '-x',  # Extract audio
        '--audio-format', 'wav',
        '--audio-quality', '0',  # Best quality
        '-o', str(output_path),
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Downloaded: {output_path.name}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error downloading: {e}")
        print(f"stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        print("ERROR: yt-dlp not found. Please install it:")
        print("  pip install yt-dlp")
        print("  or: brew install yt-dlp")
        raise


def main():
    parser = argparse.ArgumentParser(description='Test AI DJ on YouTube videos')
    parser.add_argument('url1', help='YouTube URL for song 1 (outgoing)')
    parser.add_argument('url2', help='YouTube URL for song 2 (incoming)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Save prediction to JSON file')
    parser.add_argument('--keep-audio', action='store_true',
                       help='Keep downloaded audio files (default: delete after analysis)')
    parser.add_argument('--temp-dir', type=str, default=None,
                       help='Temporary directory for downloads')
    
    args = parser.parse_args()
    
    print("="*60)
    print("AI DJ TRANSITION TEST - YouTube Videos")
    print("="*60)
    
    # Create temp directory
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix='dj_test_'))
        print(f"Using temp directory: {temp_dir}")
    
    audio_files = []
    
    try:
        # Download audio
        audio1_path = temp_dir / "song_a.wav"
        audio2_path = temp_dir / "song_b.wav"
        
        download_youtube_audio(args.url1, audio1_path)
        audio_files.append(audio1_path)
        
        download_youtube_audio(args.url2, audio2_path)
        audio_files.append(audio2_path)
        
        # Load models
        print("\nLoading AI DJ models...")
        model_dir = Path('models')
        predictor = DJTransitionPredictor(
            decision_nn_path=str(model_dir / 'decision_nn.pt') if (model_dir / 'decision_nn.pt').exists() else None,
            curve_lstm_path=str(model_dir / 'curve_lstm.pt') if (model_dir / 'curve_lstm.pt').exists() else None
        )
        
        # Analyze and predict
        print("\nAnalyzing songs and predicting transition...")
        result = predictor.predict_from_audio(str(audio1_path), str(audio2_path), duration_sec=10.0)
        
        if 'error' in result:
            print(f"\n⚠ Error: {result['error']}")
            print("Falling back to manual analysis...")
            
            # Manual analysis
            analyzer = SongAnalyzer()
            print(f"\nAnalyzing {audio1_path.name}...")
            analysis_a = analyzer.analyze(str(audio1_path))
            print(f"Analyzing {audio2_path.name}...")
            analysis_b = analyzer.analyze(str(audio2_path))
            
            song_a_features = {
                'tempo': analysis_a.get('tempo', {}).get('bpm', 120),
                'key': analysis_a.get('harmony', {}).get('key', 'C'),
                'energy': analysis_a.get('energy', {}).get('overall_energy', 0.5),
            }
            
            song_b_features = {
                'tempo': analysis_b.get('tempo', {}).get('bpm', 120),
                'key': analysis_b.get('harmony', {}).get('key', 'C'),
                'energy': analysis_b.get('energy', {}).get('overall_energy', 0.5),
            }
            
            print(f"\nSong A features: {song_a_features}")
            print(f"Song B features: {song_b_features}")
            
            result = predictor.predict_from_features(song_a_features, song_b_features, generate_curves=True)
        
        # Display results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        
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
            print(f"\n✓ Full automation curves ready for mixer control!")
        
        # Save results
        if args.output:
            import json
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Saved results to: {output_path}")
    
    finally:
        # Cleanup
        if not args.keep_audio:
            print(f"\nCleaning up temporary files...")
            for audio_file in audio_files:
                if audio_file.exists():
                    audio_file.unlink()
            if not args.temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
                print(f"✓ Removed temp directory: {temp_dir}")
        else:
            print(f"\nAudio files kept in: {temp_dir}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

