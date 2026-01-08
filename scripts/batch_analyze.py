#!/usr/bin/env python3
"""
Batch analyze multiple songs or mixes.
"""
import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.song_analyzer import SongAnalyzer
from src.database import MusicDatabase, compute_song_id


def analyze_single_song(args):
    """Analyze a single song (for multiprocessing)."""
    audio_path, output_dir, db_path, save_to_db = args
    
    try:
        analyzer = SongAnalyzer()
        analysis = analyzer.analyze(audio_path)
        
        song_id = analysis['song_id']
        output_path = Path(output_dir) / f"{song_id}.json"
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        if save_to_db:
            db = MusicDatabase(db_path)
            db.save_song_analysis(song_id, analysis, str(output_path))
        
        return {'success': True, 'song_id': song_id, 'path': audio_path}
    except Exception as e:
        return {'success': False, 'path': audio_path, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Batch analyze songs')
    parser.add_argument('input_dir', type=str, help='Directory containing audio files')
    parser.add_argument('--output-dir', type=str, default='data/analyses',
                       help='Directory to save analyses')
    parser.add_argument('--extensions', type=str, nargs='+',
                       default=['.mp3', '.wav', '.flac', '.m4a'],
                       help='Audio file extensions to process')
    parser.add_argument('--save-to-db', action='store_true',
                       help='Save to database')
    parser.add_argument('--db-path', type=str, default='data/music_analysis.db',
                       help='Database path')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Find all audio files
    input_dir = Path(args.input_dir)
    audio_files = []
    for ext in args.extensions:
        audio_files.extend(input_dir.rglob(f'*{ext}'))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for workers
    worker_args = [
        (str(f), str(output_dir), args.db_path, args.save_to_db)
        for f in audio_files
    ]
    
    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_single_song, args): args[0] 
                  for args in worker_args}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            if result['success']:
                print(f"✓ {Path(result['path']).name}")
            else:
                print(f"✗ {Path(result['path']).name}: {result['error']}")
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\nBatch analysis complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == '__main__':
    main()

