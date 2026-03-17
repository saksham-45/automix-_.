import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

# Add src to path
sys.path.append(str(Path.cwd()))

from src.smart_mixer import SmartMixer

def main():
    # Song paths
    song_a = "/Users/saksham/automix-_./data/cache/stream/DawrlSwHUiM.wav"  # Overnight Celebrity
    song_b = "/Users/saksham/automix-_./data/cache/stream/WhF7A4834Gc.wav"  # Les
    
    if not os.path.exists(song_a) or not os.path.exists(song_b):
        print(f"Error: One or more song files missing.\n  A: {song_a}\n  B: {song_b}")
        return

    print(f"Mixing {Path(song_a).name} and {Path(song_b).name}...")
    
    # Initialize mixer
    mixer = SmartMixer(sr=44100)
    
    # Enable and configure superhuman engine for morphing
    # We'll explicitly force 'progressive_morph' to show off the new feature
    if mixer.superhuman_enabled:
        mixer.superhuman_engine.configure(
            stem_morphing_enabled=True,
            stem_morph_depth=0.9,
            stem_morph_strategy='best_match',
            conversation_type='progressive_morph',  # Force our new morphing conversation
            force_stem_orchestration=True           # Ensure we go through stem path
        )
    
    # Create the mix
    # We'll use a 32-bar transition (approx 60s) for a deep morphing experience
    try:
        mixed_audio, metadata = mixer.create_superhuman_mix(
            song_a,
            song_b,
            transition_duration=60.0,
            force_stem_orchestration=True,
            conversation_type_override='progressive_morph',
            return_metadata=True
        )
        
        # Save the result
        output_path = "/Users/saksham/automix-_./overnight_les_morph_mix.wav"
        sf.write(output_path, mixed_audio, 44100)
        
        print(f"\n✨ Mix Complete! ✨")
        print(f"Output saved to: {output_path}")
        print(f"Transition Duration: 60.0s")
        
    except Exception as e:
        print(f"Error during mixing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
