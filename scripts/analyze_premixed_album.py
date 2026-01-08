#!/usr/bin/env python3
"""
Analyze a premixed album (DJ mix) and extract training data for AI.

This script:
1. Analyzes the mix to detect transitions
2. Extracts detailed transition information
3. Stores everything in the database
4. Exports training data for ML models
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import uuid
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mix_analyzer import MixAnalyzer
from src.song_analyzer import SongAnalyzer
from src.transition_detector import TransitionDetector
from src.transition_analyzer import TransitionAnalyzer
from src.training_data_extractor import TrainingDataExtractor
from src.database import MusicDatabase
from src.curve_parameterizer import CurveParameterizer
import librosa
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def analyze_premixed_album(
    mix_path: str,
    mix_title: str = None,
    dj_name: str = None,
    genre: str = None,
    track_library: Optional[Dict[str, str]] = None,
    save_to_db: bool = True,
    export_training_data: bool = True,
    db_path: str = "data/music_analysis.db",
    output_dir: str = "data/premixed_albums"
) -> Dict:
    """
    Complete analysis of a premixed album.
    
    Args:
        mix_path: Path to the premixed album audio file
        mix_title: Title of the mix/album
        dj_name: Name of the DJ (if known)
        genre: Genre of the mix
        track_library: Optional dict mapping song_id -> audio_path for track identification
        save_to_db: Whether to save to database
        export_training_data: Whether to export ML training data
        db_path: Database path
        output_dir: Output directory for analysis files
        
    Returns:
        Complete analysis dictionary
    """
    print("=" * 70)
    print(f"ANALYZING PREMIXED ALBUM")
    print("=" * 70)
    print(f"Mix: {mix_path}")
    if mix_title:
        print(f"Title: {mix_title}")
    if dj_name:
        print(f"DJ: {dj_name}")
    print()
    
    # Initialize analyzers
    song_analyzer = SongAnalyzer()
    mix_analyzer = MixAnalyzer(song_analyzer=song_analyzer)
    transition_detector = TransitionDetector()
    transition_analyzer = TransitionAnalyzer()
    training_extractor = TrainingDataExtractor()
    curve_param = CurveParameterizer()
    
    # Load mix audio
    print("Loading mix audio...")
    mix_audio, sr = librosa.load(mix_path, sr=44100)
    duration = len(mix_audio) / sr
    print(f"✓ Loaded: {duration/60:.1f} minutes")
    print()
    
    # Generate mix ID
    mix_id = Path(mix_path).stem if not mix_title else mix_title.replace(" ", "_")
    
    # Step 1: Detect transitions
    print("Step 1: Detecting transitions...")
    transitions = transition_detector.detect_all(mix_path)
    print(f"✓ Detected {len(transitions)} transitions")
    print()
    
    # Step 2: Analyze each transition in detail
    print("Step 2: Analyzing transitions in detail...")
    analyzed_transitions = []
    
    for i, transition in enumerate(transitions):
        print(f"  Analyzing transition {i+1}/{len(transitions)} "
              f"({transition.start_sec:.1f}s - {transition.end_sec:.1f}s)...")
        
        # Extract transition segment
        start_idx = int(transition.start_sec * sr)
        end_idx = int(transition.end_sec * sr)
        transition_segment = mix_audio[start_idx:end_idx]
        
        # Analyze transition execution
        transition_analysis = transition_analyzer.analyze_transition(
            mix_audio=mix_audio,
            transition_start_sec=transition.start_sec,
            transition_end_sec=transition.end_sec
        )
        
        # Convert to dict for storage
        transition_dict = {
            "transition_id": str(uuid.uuid4()),
            "start_time_sec": transition.start_sec,
            "end_time_sec": transition.end_sec,
            "duration_sec": transition.end_sec - transition.start_sec,
            "confidence": transition.confidence,
            "detection_method": transition.detection_method,
            "analysis": {
                "technique_primary": transition_analysis.technique_primary,
                "technique_secondary": transition_analysis.technique_secondary,
                "technique_confidence": transition_analysis.technique_confidence,
                "volume_curves": {
                    "times_relative_sec": transition_analysis.volume_curves.times_relative_sec,
                    "track_a_gain_db": transition_analysis.volume_curves.track_a_gain_db,
                    "track_b_gain_db": transition_analysis.volume_curves.track_b_gain_db,
                    "crossfade_type": transition_analysis.volume_curves.crossfade_type
                },
                "eq_automation": {
                    "bass_swap_detected": transition_analysis.eq_automation.bass_swap_detected,
                    "bass_swap_point_sec": transition_analysis.eq_automation.bass_swap_point_sec,
                    "highpass_sweep_detected": transition_analysis.eq_automation.highpass_sweep_detected,
                    "lowpass_sweep_detected": transition_analysis.eq_automation.lowpass_sweep_detected
                },
                "beat_alignment": {
                    "phase_offset_ms": transition_analysis.beat_alignment.phase_offset_ms,
                    "pitch_shift_semitones": transition_analysis.beat_alignment.pitch_shift_semitones,
                    "aligned_on_downbeat": transition_analysis.beat_alignment.aligned_on_downbeat,
                    "alignment_quality": transition_analysis.beat_alignment.alignment_quality
                },
                "energy_during_transition": transition_analysis.energy_during_transition,
                "spectral_during_transition": transition_analysis.spectral_during_transition,
                "quality_assessment": transition_analysis.quality_assessment
            }
        }
        
        analyzed_transitions.append(transition_dict)
    
    print(f"✓ Analyzed {len(analyzed_transitions)} transitions")
    print()
    
    # Step 3: Estimate track boundaries (if no library provided)
    print("Step 3: Estimating track boundaries...")
    estimated_tracks = []
    for i, transition in enumerate(transitions):
        if i == 0:
            start_sec = 0.0
        else:
            start_sec = transitions[i-1].end_sec
        
        end_sec = transition.start_sec
        
        estimated_tracks.append({
            "track_index": i,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": end_sec - start_sec,
            "estimated": True
        })
    
    # Add final track
    if transitions:
        last_transition = transitions[-1]
        estimated_tracks.append({
            "track_index": len(transitions),
            "start_sec": last_transition.end_sec,
            "end_sec": duration,
            "duration_sec": duration - last_transition.end_sec,
            "estimated": True
        })
    
    print(f"✓ Estimated {len(estimated_tracks)} tracks")
    print()
    
    # Step 4: Build complete mix analysis
    mix_analysis = {
        "mix_id": mix_id,
        "mix_title": mix_title or Path(mix_path).stem,
        "dj_name": dj_name,
        "genre": genre,
        "mix_path": str(mix_path),
        "duration_sec": duration,
        "duration_minutes": duration / 60,
        "sample_rate": sr,
        "analyzed_at": datetime.now().isoformat(),
        "total_transitions": len(transitions),
        "total_tracks": len(estimated_tracks),
        "detected_transitions": [
            {
                "start_sec": t.start_sec,
                "end_sec": t.end_sec,
                "duration_sec": t.end_sec - t.start_sec,
                "confidence": t.confidence,
                "detection_method": t.detection_method
            }
            for t in transitions
        ],
        "estimated_tracks": estimated_tracks,
        "analyzed_transitions": analyzed_transitions
    }
    
    # Step 5: Save analysis to JSON
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    analysis_output_path = output_dir_path / f"{mix_id}_analysis.json"
    print(f"Step 4: Saving analysis...")
    print(f"  Output: {analysis_output_path}")
    
    with open(analysis_output_path, 'w') as f:
        json.dump(mix_analysis, f, indent=2, cls=NumpyEncoder)
    
    print(f"✓ Saved analysis")
    print()
    
    # Step 6: Save to database
    if save_to_db:
        print("Step 5: Saving to database...")
        db = MusicDatabase(db_path)
        
        # Save mix metadata
        db.save_mix_metadata(
            mix_id=mix_id,
            title=mix_analysis["mix_title"],
            dj=dj_name,
            genre=genre,
            duration_sec=duration,
            source_url=str(analysis_output_path)
        )
        
        # Save transitions
        for transition_data in analyzed_transitions:
            db.save_transition(
                transition_id=transition_data["transition_id"],
                transition_data=transition_data,
                transition_json_path=str(analysis_output_path)
            )
        
        print(f"✓ Saved to database: {db_path}")
        print()
    
    # Step 7: Export training data (if requested)
    training_examples = []
    if export_training_data:
        print("Step 6: Extracting training data...")
        print("  Note: Full training data extraction requires individual track analyses.")
        print("  For now, exporting transition execution data...")
        
        training_output_dir = output_dir_path / f"{mix_id}_training_data"
        training_output_dir.mkdir(exist_ok=True)
        
        for i, transition_data in enumerate(analyzed_transitions):
            # Create simplified training example
            # (Full extraction requires track analyses - see export_training_data.py)
            training_example = {
                "transition_id": transition_data["transition_id"],
                "mix_id": mix_id,
                "transition_index": i,
                "input_features": {
                    "transition_start_sec": transition_data["start_time_sec"],
                    "transition_duration_sec": transition_data["duration_sec"],
                    "mix_position_sec": transition_data["start_time_sec"],
                    "mix_position_percent": transition_data["start_time_sec"] / duration
                },
                "output_labels": {
                    "technique": transition_data["analysis"]["technique_primary"],
                    "technique_secondary": transition_data["analysis"]["technique_secondary"],
                    "duration_bars": int(transition_data["duration_sec"] / 2.0),  # Estimate
                    "volume_curves": transition_data["analysis"]["volume_curves"],
                    "eq_automation": transition_data["analysis"]["eq_automation"],
                    "beat_alignment": transition_data["analysis"]["beat_alignment"]
                },
                "quality_labels": transition_data["analysis"]["quality_assessment"]
            }
            
            training_examples.append(training_example)
            
            # Save individual training example
            example_path = training_output_dir / f"transition_{i+1:03d}.json"
            with open(example_path, 'w') as f:
                json.dump(training_example, f, indent=2, cls=NumpyEncoder)
        
        # Save summary
        training_summary = {
            "mix_id": mix_id,
            "total_examples": len(training_examples),
            "techniques": {},
            "quality_stats": {
                "mean_quality": np.mean([
                    ex["quality_labels"]["overall_transition_quality"]
                    for ex in training_examples
                ]),
                "min_quality": np.min([
                    ex["quality_labels"]["overall_transition_quality"]
                    for ex in training_examples
                ]),
                "max_quality": np.max([
                    ex["quality_labels"]["overall_transition_quality"]
                    for ex in training_examples
                ])
            }
        }
        
        # Count techniques
        for ex in training_examples:
            technique = ex["output_labels"]["technique"]
            training_summary["techniques"][technique] = \
                training_summary["techniques"].get(technique, 0) + 1
        
        summary_path = training_output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, cls=NumpyEncoder)
        
        print(f"✓ Exported {len(training_examples)} training examples")
        print(f"  Output: {training_output_dir}")
        print()
    
    # Summary
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Mix ID: {mix_id}")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Transitions detected: {len(transitions)}")
    print(f"Tracks estimated: {len(estimated_tracks)}")
    print(f"Analysis saved: {analysis_output_path}")
    if save_to_db:
        print(f"Database: {db_path}")
    if export_training_data:
        print(f"Training data: {training_output_dir}")
    print()
    
    return mix_analysis


def main():
    parser = argparse.ArgumentParser(
        description='Analyze a premixed album and extract training data'
    )
    parser.add_argument('mix_path', type=str, help='Path to premixed album audio file')
    parser.add_argument('--title', type=str, help='Title of the mix/album')
    parser.add_argument('--dj', type=str, help='Name of the DJ')
    parser.add_argument('--genre', type=str, help='Genre of the mix')
    parser.add_argument('--track-library', type=str,
                       help='JSON file mapping song_id -> audio_path')
    parser.add_argument('--no-db', action='store_true',
                       help='Skip saving to database')
    parser.add_argument('--no-training', action='store_true',
                       help='Skip exporting training data')
    parser.add_argument('--db-path', type=str, default='data/music_analysis.db',
                       help='Database path')
    parser.add_argument('--output-dir', type=str, default='data/premixed_albums',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load track library if provided
    track_library = None
    if args.track_library:
        with open(args.track_library, 'r') as f:
            track_library = json.load(f)
    
    # Analyze
    analysis = analyze_premixed_album(
        mix_path=args.mix_path,
        mix_title=args.title,
        dj_name=args.dj,
        genre=args.genre,
        track_library=track_library,
        save_to_db=not args.no_db,
        export_training_data=not args.no_training,
        db_path=args.db_path,
        output_dir=args.output_dir
    )
    
    print("✓ Done!")


if __name__ == '__main__':
    main()

