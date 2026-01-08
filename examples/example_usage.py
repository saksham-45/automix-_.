"""
Example usage of the DJ Transition Analysis System.

This script demonstrates how to:
1. Analyze a single song
2. Analyze a DJ mix
3. Extract and analyze transitions
4. Store results in database
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.song_analyzer import SongAnalyzer
from src.mix_analyzer import MixAnalyzer
from src.transition_analyzer import TransitionAnalyzer
from src.database import TransitionDatabase
import json


def example_analyze_song():
    """Example: Analyze a single song and extract all features."""
    print("=" * 60)
    print("EXAMPLE 1: Analyzing a Single Song")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SongAnalyzer()
    
    # Path to your audio file
    audio_path = Path("path/to/your/song.wav")
    
    if not audio_path.exists():
        print(f"⚠️  Audio file not found: {audio_path}")
        print("   Please update the path to a real audio file.")
        return
    
    print(f"Analyzing: {audio_path}")
    print("This may take a few minutes...")
    
    # Analyze the song
    analysis = analyzer.analyze(audio_path)
    
    # Save results
    output_path = Path("output") / f"{analysis['song_id']}_analysis.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"✅ Analysis complete!")
    print(f"   Song ID: {analysis['song_id']}")
    print(f"   BPM: {analysis['tempo']['bpm']}")
    print(f"   Key: {analysis['key']['estimated_key']} {analysis['key']['mode']}")
    print(f"   Camelot: {analysis['key']['camelot']}")
    print(f"   Energy: {analysis['energy']['energy_statistics']['mean']:.2f}")
    print(f"   Duration: {analysis['duration_sec']:.1f}s")
    print(f"   Saved to: {output_path}")
    print()


def example_analyze_mix():
    """Example: Analyze a DJ mix and detect transitions."""
    print("=" * 60)
    print("EXAMPLE 2: Analyzing a DJ Mix")
    print("=" * 60)
    
    # Initialize mix analyzer
    mix_analyzer = MixAnalyzer()
    
    # Path to your DJ mix
    mix_path = Path("path/to/your/dj_mix.wav")
    
    if not mix_path.exists():
        print(f"⚠️  Mix file not found: {mix_path}")
        print("   Please update the path to a real mix file.")
        return
    
    print(f"Analyzing mix: {mix_path}")
    print("This will take several minutes...")
    
    # Analyze the mix
    mix_analysis = mix_analyzer.analyze_mix(mix_path)
    
    # Save results
    output_path = Path("output") / f"{mix_analysis['mix_id']}_mix_analysis.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(mix_analysis, f, indent=2)
    
    print(f"✅ Mix analysis complete!")
    print(f"   Mix ID: {mix_analysis['mix_id']}")
    print(f"   Detected {len(mix_analysis['detected_transitions'])} transitions")
    print(f"   Saved to: {output_path}")
    print()


def example_analyze_transition():
    """Example: Analyze a specific transition in detail."""
    print("=" * 60)
    print("EXAMPLE 3: Analyzing a Transition")
    print("=" * 60)
    
    # Initialize transition analyzer
    transition_analyzer = TransitionAnalyzer()
    
    # You need:
    # 1. The mix audio
    # 2. The two original tracks (if available)
    # 3. Transition timing
    
    mix_path = Path("path/to/mix.wav")
    track_a_path = Path("path/to/track_a.wav")
    track_b_path = Path("path/to/track_b.wav")
    
    transition_start_sec = 218.5
    transition_end_sec = 252.3
    
    if not all(p.exists() for p in [mix_path, track_a_path, track_b_path]):
        print("⚠️  Audio files not found. Please update paths.")
        return
    
    print(f"Analyzing transition from {transition_start_sec}s to {transition_end_sec}s")
    
    # Analyze the transition
    transition_analysis = transition_analyzer.analyze_transition(
        mix_path=mix_path,
        track_a_path=track_a_path,
        track_b_path=track_b_path,
        transition_start_sec=transition_start_sec,
        transition_end_sec=transition_end_sec
    )
    
    # Save results
    output_path = Path("output") / f"{transition_analysis['transition_id']}_transition.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(transition_analysis, f, indent=2)
    
    print(f"✅ Transition analysis complete!")
    print(f"   Transition ID: {transition_analysis['transition_id']}")
    print(f"   Technique: {transition_analysis['transition_execution']['technique_primary']}")
    print(f"   Duration: {transition_analysis['transition_execution']['duration_sec']:.1f}s")
    print(f"   Quality: {transition_analysis['quality_assessment']['overall_transition_quality']:.2f}")
    print(f"   Saved to: {output_path}")
    print()


def example_store_in_database():
    """Example: Store analysis results in database."""
    print("=" * 60)
    print("EXAMPLE 4: Storing in Database")
    print("=" * 60)
    
    # Initialize database
    db = TransitionDatabase("output/transitions.db")
    db.initialize()
    
    # Load a song analysis
    song_analysis_path = Path("output/song_analysis.json")
    if not song_analysis_path.exists():
        print(f"⚠️  Song analysis not found: {song_analysis_path}")
        print("   Run example_analyze_song() first.")
        return
    
    with open(song_analysis_path, 'r') as f:
        song_analysis = json.load(f)
    
    # Store song
    db.store_song_analysis(song_analysis)
    print(f"✅ Stored song: {song_analysis['song_id']}")
    
    # Load a transition analysis
    transition_analysis_path = Path("output/transition_analysis.json")
    if not transition_analysis_path.exists():
        print(f"⚠️  Transition analysis not found: {transition_analysis_path}")
        print("   Run example_analyze_transition() first.")
        return
    
    with open(transition_analysis_path, 'r') as f:
        transition_analysis = json.load(f)
    
    # Store transition
    db.store_transition(transition_analysis)
    print(f"✅ Stored transition: {transition_analysis['transition_id']}")
    
    # Query examples
    print("\n📊 Query Examples:")
    
    # Find compatible tracks
    compatible = db.find_compatible_tracks(
        bpm=128,
        key="8A",
        energy=0.7,
        limit=5
    )
    print(f"   Found {len(compatible)} compatible tracks")
    
    # Find transitions by technique
    transitions = db.get_transitions_by_technique("long_blend", limit=10)
    print(f"   Found {len(transitions)} transitions using 'long_blend'")
    
    db.close()
    print()


def example_full_pipeline():
    """Example: Complete pipeline from mix to database."""
    print("=" * 60)
    print("EXAMPLE 5: Full Pipeline")
    print("=" * 60)
    
    print("This example shows the complete workflow:")
    print("1. Analyze DJ mix → detect transitions")
    print("2. For each transition, analyze in detail")
    print("3. Store everything in database")
    print()
    print("See scripts/analyze_mix.py for the full implementation.")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DJ TRANSITION ANALYSIS SYSTEM - EXAMPLES")
    print("=" * 60)
    print()
    
    # Run examples (comment out ones you don't want to run)
    example_analyze_song()
    # example_analyze_mix()
    # example_analyze_transition()
    # example_store_in_database()
    # example_full_pipeline()
    
    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)

