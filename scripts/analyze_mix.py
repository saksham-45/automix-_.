#!/usr/bin/env python3
"""
Script to analyze a DJ mix and extract transitions.
"""
import sys
import json
import argparse
from pathlib import Path
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mix_analyzer import MixAnalyzer
from src.database import MusicDatabase


def main():
    parser = argparse.ArgumentParser(description='Analyze a DJ mix and extract transitions')
    parser.add_argument('mix_path', type=str, help='Path to mix audio file')
    parser.add_argument('--output-dir', type=str, default='data/mix_analyses',
                       help='Directory to save analysis JSON')
    parser.add_argument('--track-library', type=str,
                       help='JSON file mapping song_id -> audio_path')
    parser.add_argument('--save-to-db', action='store_true',
                       help='Save transitions to database')
    parser.add_argument('--db-path', type=str, default='data/music_analysis.db',
                       help='Database path')
    parser.add_argument('--mix-id', type=str,
                       help='Mix identifier (defaults to filename)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load track library if provided
    track_library = None
    if args.track_library:
        with open(args.track_library, 'r') as f:
            track_library = json.load(f)
    
    # Analyze mix
    print(f"Analyzing mix: {args.mix_path}")
    analyzer = MixAnalyzer()
    
    mix_id = args.mix_id or Path(args.mix_path).stem
    analysis = analyzer.analyze_mix(args.mix_path, track_library, mix_id)
    
    # Save to JSON
    output_path = output_dir / f"{mix_id}.json"
    print(f"Saving analysis to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Save transitions to database if requested
    if args.save_to_db:
        print(f"Saving transitions to database...")
        db = MusicDatabase(args.db_path)
        
        for transition_data in analysis['analyzed_transitions']:
            transition_id = str(uuid.uuid4())
            
            # Format transition data for database
            transition_record = {
                'transition': transition_data['transition'],
                'context': {
                    'source_mix_id': mix_id,
                    'dj': None,  # Could be extracted from filename/metadata
                    'genre': None
                },
                'track_a_features': transition_data.get('track_a_info', {}),
                'track_b_features': transition_data.get('track_b_info', {}),
                'transition_execution': transition_data['analysis'],
                'compatibility_metrics': transition_data['analysis'].get('compatibility', {}),
                'quality_assessment': transition_data['analysis'].get('quality', {})
            }
            
            transition_json_path = output_dir / f"transition_{transition_id}.json"
            with open(transition_json_path, 'w') as f:
                json.dump(transition_record, f, indent=2)
            
            db.save_transition(transition_id, transition_record, str(transition_json_path))
        
        print(f"Saved {len(analysis['analyzed_transitions'])} transitions to database!")
    
    print(f"\nMix analysis complete!")
    print(f"Mix ID: {mix_id}")
    print(f"Total tracks: {analysis['total_tracks']}")
    print(f"Total transitions: {analysis['total_transitions']}")


if __name__ == '__main__':
    main()

