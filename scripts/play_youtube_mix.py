#!/usr/bin/env python3
"""
Download YouTube videos, generate transition, and render the mix.

One-stop script to go from YouTube URLs to a mixed audio file.
"""
import sys
import subprocess
import json
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.inference import DJTransitionPredictor
from src.song_analyzer import SongAnalyzer
from scripts.render_transition import render_transition


def download_youtube_audio(url: str, output_path: Path) -> Path:
    """Download audio from YouTube URL."""
    print(f"Downloading: {url}")
    cmd = [
        'yt-dlp',
        '-x',
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '-o', str(output_path),
        url
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download YouTube videos and render AI DJ mix')
    parser.add_argument('url1', help='YouTube URL for song 1')
    parser.add_argument('url2', help='YouTube URL for song 2')
    parser.add_argument('--output', '-o', default='ai_dj_mix.wav',
                       help='Output audio file')
    parser.add_argument('--keep-audio', action='store_true',
                       help='Keep downloaded audio files')
    
    args = parser.parse_args()
    
    temp_dir = Path(tempfile.mkdtemp(prefix='dj_mix_'))
    audio1_path = temp_dir / "song_a.wav"
    audio2_path = temp_dir / "song_b.wav"
    
    try:
        # Download
        print("="*60)
        print("STEP 1: Downloading audio...")
        print("="*60)
        download_youtube_audio(args.url1, audio1_path)
        download_youtube_audio(args.url2, audio2_path)
        
        # Analyze and predict
        print("\n" + "="*60)
        print("STEP 2: Analyzing and predicting transition...")
        print("="*60)
        
        model_dir = Path('models')
        predictor = DJTransitionPredictor(
            decision_nn_path=str(model_dir / 'decision_nn.pt') if (model_dir / 'decision_nn.pt').exists() else None,
            curve_lstm_path=str(model_dir / 'curve_lstm.pt') if (model_dir / 'curve_lstm.pt').exists() else None
        )
        
        analyzer = SongAnalyzer()
        print(f"\nAnalyzing {audio1_path.name}...")
        analysis_a = analyzer.analyze(str(audio1_path))
        print(f"Analyzing {audio2_path.name}...")
        analysis_b = analyzer.analyze(str(audio2_path))
        
        # Extract features
        key_a = analysis_a.get('harmony', {}).get('key', 'C')
        if isinstance(key_a, dict):
            key_a = key_a.get('key', 'C')
        elif isinstance(key_a, list) and len(key_a) > 0:
            key_a = key_a[0] if isinstance(key_a[0], str) else 'C'
        
        key_b = analysis_b.get('harmony', {}).get('key', 'C')
        if isinstance(key_b, dict):
            key_b = key_b.get('key', 'C')
        elif isinstance(key_b, list) and len(key_b) > 0:
            key_b = key_b[0] if isinstance(key_b[0], str) else 'C'
        
        song_a_features = {
            'tempo': analysis_a.get('tempo', {}).get('bpm', 120) if isinstance(analysis_a.get('tempo'), dict) else 120,
            'key': key_a,
            'energy': analysis_a.get('energy', {}).get('overall_energy', 0.5) if isinstance(analysis_a.get('energy'), dict) else 0.5,
        }
        
        song_b_features = {
            'tempo': analysis_b.get('tempo', {}).get('bpm', 120) if isinstance(analysis_b.get('tempo'), dict) else 120,
            'key': key_b,
            'energy': analysis_b.get('energy', {}).get('overall_energy', 0.5) if isinstance(analysis_b.get('energy'), dict) else 0.5,
        }
        
        result = predictor.predict_from_features(song_a_features, song_b_features, generate_curves=True)
        
        print(f"\n✓ Prediction: {result.get('technique', 'unknown').upper()} transition")
        print(f"  Duration: {result.get('duration_bars', 5)} bars")
        
        # Save transition data
        transition_json = temp_dir / "transition.json"
        with open(transition_json, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Render mix
        print("\n" + "="*60)
        print("STEP 3: Rendering audio mix...")
        print("="*60)
        
        render_transition(
            str(audio1_path),
            str(audio2_path),
            result,
            args.output
        )
        
        print("\n" + "="*60)
        print("✓ COMPLETE!")
        print("="*60)
        print(f"\n🎵 Your AI DJ mix is ready: {args.output}")
        print(f"\nYou can now listen to it!")
        
    finally:
        if not args.keep_audio:
            shutil.rmtree(temp_dir)
        else:
            print(f"\nAudio files kept in: {temp_dir}")


if __name__ == '__main__':
    main()

