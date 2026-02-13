#!/usr/bin/env python3
import sys
import os
import numpy as np
sys.path.insert(0, os.getcwd())

from src.smart_mixer import SmartMixer
from src.technique_executor import TechniqueExecutor
from src.superhuman_engine import SuperhumanDJEngine

def test_improvements():
    print("🧪 Verifying Sonic Improvements...")
    
    # Create dummy audio (stereo, 10 seconds at 44.1kHz)
    sr = 44100
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration))
    # Sine wave pluck
    audio_a = np.sin(2 * np.pi * 440 * t) * np.exp(-t)
    audio_b = np.sin(2 * np.pi * 554 * t) * np.exp(-t)
    
    # Make stereo
    audio_a = np.column_stack([audio_a, audio_a])
    audio_b = np.column_stack([audio_b, audio_b])
    
    # Test TechniqueExecutor Methods
    print("\n[1] Testing TechniqueExecutor...")
    executor = TechniqueExecutor(sr=sr)
    
    # Test longer blend with EQ
    print("  Testing _execute_long_blend (with EQ)...")
    try:
        mixed = executor._execute_long_blend(audio_a, audio_b, {'duration_sec': 10.0})
        print(f"    ✓ Success (shape: {mixed.shape})")
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test filter sweep (STFT)
    print("  Testing _execute_filter_sweep (STFT)...")
    try:
        mixed = executor._execute_filter_sweep(audio_a, audio_b, {'filter_type': 'low_pass'})
        print(f"    ✓ Success (shape: {mixed.shape})")
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test echo out (Multi-tap)
    print("  Testing _execute_echo_out (Multi-tap)...")
    try:
        mixed = executor._execute_echo_out(audio_a, audio_b, {'tempo': 128.0})
        print(f"    ✓ Success (shape: {mixed.shape})")
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        
    # Test staggered stem mix balancing
    print("  Testing _execute_staggered_stem_mix (Balancing)...")
    try:
        stems_a = {'drums': audio_a, 'bass': audio_a, 'vocals': audio_a, 'other': audio_a}
        stems_b = {'drums': audio_b, 'bass': audio_b, 'vocals': audio_b, 'other': audio_b}
        mixed = executor._execute_staggered_stem_mix(audio_a, audio_b, {'tempo_a': 120, 'tempo_b': 120}, stems_a, stems_b)
        print(f"    ✓ Success (shape: {mixed.shape})")
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ Verification Complete")

if __name__ == "__main__":
    test_improvements()
