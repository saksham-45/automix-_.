#!/usr/bin/env python3
"""
Script to analyze a single song and extract all features.
"""
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.song_analyzer import SongAnalyzer
from src.database import MusicDatabase, compute_song_id


def main():
    parser = argparse.ArgumentParser(description='Analyze a song and extract all features')
    parser.add_argument('audio_path', type=str, help='Path to audio file')
    parser.add_argument('--output-dir', type=str, default='data/analyses',
                       help='Directory to save analysis JSON')
    parser.add_argument('--save-to-db', action='store_true',
                       help='Save analysis to database')
    parser.add_argument('--db-path', type=str, default='data/music_analysis.db',
                       help='Database path')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze song
    print(f"Analyzing song: {args.audio_path}")
    analyzer = SongAnalyzer()
    analysis = analyzer.analyze(args.audio_path)
    
    # Save to JSON
    song_id = analysis['song_id']
    output_path = output_dir / f"{song_id}.json"
    
    print(f"Saving analysis to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Save to database if requested
    if args.save_to_db:
        print(f"Saving to database: {args.db_path}")
        db = MusicDatabase(args.db_path)
        db.save_song_analysis(song_id, analysis, str(output_path))
        print("Saved to database!")
    
    print(f"\nAnalysis complete!")
    print(f"Song ID: {song_id}")
    print(f"BPM: {analysis.get('tempo', {}).get('bpm')}")
    print(f"Key: {analysis.get('harmony', {}).get('key', {}).get('estimated_key')}")
    print(f"Energy: {analysis.get('energy', {}).get('energy_statistics', {}).get('mean', 0):.2f}")


if __name__ == '__main__':
    main()

