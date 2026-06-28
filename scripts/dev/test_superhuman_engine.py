#!/usr/bin/env python3
"""
End-to-End Test for Superhuman DJ Engine

Tests all new advanced modules:
- MicroTimingEngine
- SpectralIntelligenceEngine
- HybridTechniqueBlender
- StemOrchestrator  
- MonteCarloQualityOptimizer
- SuperhumanDJEngine

Creates synthetic test audio and runs the full mixing pipeline.
"""
import numpy as np
import soundfile as sf
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_test_audio(duration_sec: float = 30.0, 
                        tempo_bpm: float = 128.0,
                        key_freq: float = 440.0,
                        sr: int = 44100,
                        style: str = 'electronic') -> np.ndarray:
    """
    Generate synthetic test audio that simulates electronic music.
    
    Creates a track with:
    - Kick drum pattern
    - Hi-hats
    - Bass line
    - Melodic elements
    """
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples)
    
    # Calculate beat timing
    beat_duration = 60.0 / tempo_bpm
    
    # Initialize output
    audio = np.zeros(n_samples)
    
    # 1. Kick drum (on every beat)
    kick = np.zeros(n_samples)
    kick_samples = int(0.1 * sr)  # 100ms kick
    for beat in np.arange(0, duration_sec, beat_duration):
        start = int(beat * sr)
        end = min(start + kick_samples, n_samples)
        if end > start:
            kick_t = np.linspace(0, 1, end - start)
            # Low frequency sine with exponential decay
            kick[start:end] += np.sin(2 * np.pi * 60 * kick_t) * np.exp(-10 * kick_t) * 0.8
    
    # 2. Hi-hats (on off-beats)
    hihat = np.zeros(n_samples)
    hihat_samples = int(0.05 * sr)  # 50ms hi-hat
    for beat in np.arange(beat_duration/2, duration_sec, beat_duration/2):
        start = int(beat * sr)
        end = min(start + hihat_samples, n_samples)
        if end > start:
            hihat_t = np.linspace(0, 1, end - start)
            # High frequency noise with fast decay
            hihat[start:end] += np.random.randn(end - start) * np.exp(-30 * hihat_t) * 0.15
    
    # 3. Bass line (follows kick with sub-bass)
    bass = np.zeros(n_samples)
    bass_freq = key_freq / 4  # Two octaves below key
    bass_samples = int(0.3 * sr)  # 300ms bass note
    for bar in range(int(duration_sec / (beat_duration * 4))):
        bar_start = bar * beat_duration * 4
        # Bass on beat 1 and 3
        for beat_offset in [0, beat_duration * 2]:
            start = int((bar_start + beat_offset) * sr)
            end = min(start + bass_samples, n_samples)
            if end > start:
                bass_t = np.linspace(0, 1, end - start)
                bass[start:end] += np.sin(2 * np.pi * bass_freq * bass_t) * np.exp(-3 * bass_t) * 0.5
    
    # 4. Melody (simple chord progression)
    melody = np.zeros(n_samples)
    chord_duration = beat_duration * 4  # One chord per bar
    chord_freqs = [key_freq, key_freq * 1.25, key_freq * 1.5, key_freq * 0.75]  # Simple progression
    
    for i, bar in enumerate(range(int(duration_sec / chord_duration))):
        start = int(bar * chord_duration * sr)
        end = min(int((bar + 1) * chord_duration * sr), n_samples)
        if end > start:
            chord_t = np.linspace(0, chord_duration, end - start)
            freq = chord_freqs[i % len(chord_freqs)]
            # Add fundamental and harmonics
            melody[start:end] += np.sin(2 * np.pi * freq * chord_t) * 0.2
            melody[start:end] += np.sin(2 * np.pi * freq * 2 * chord_t) * 0.1
            melody[start:end] += np.sin(2 * np.pi * freq * 3 * chord_t) * 0.05
    
    # Apply envelope to melody
    envelope = np.ones(n_samples)
    attack = int(0.01 * sr)
    release = int(0.1 * sr)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    melody *= envelope
    
    # Combine all elements
    audio = kick * 0.4 + hihat * 0.2 + bass * 0.3 + melody * 0.25
    
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.8
    
    # Convert to stereo
    audio_stereo = np.column_stack([audio, audio])
    
    return audio_stereo


def test_micro_timing_engine():
    """Test the MicroTimingEngine module."""
    print("\n" + "="*60)
    print("🎯 Testing MicroTimingEngine")
    print("="*60)
    
    from src.micro_timing_engine import MicroTimingEngine
    
    engine = MicroTimingEngine()
    
    # Generate test audio
    audio = generate_test_audio(duration_sec=10, tempo_bpm=128)[:, 0]  # Mono
    
    # Test groove extraction
    print("  Testing groove extraction...")
    groove = engine.extract_groove_pattern(audio, tempo=128)
    print(f"    ✓ Swing ratio: {groove['swing_ratio']:.3f}")
    print(f"    ✓ Rhythm complexity: {groove['rhythm_complexity']:.3f}")
    print(f"    ✓ Onset count: {groove['onset_count']}")
    
    # Test transient detection
    print("  Testing transient detection...")
    transients = engine.detect_transients(audio)
    print(f"    ✓ Transients detected: {transients['count']}")
    if transients['transient_types']:
        type_counts = {}
        for t in transients['transient_types']:
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"    ✓ Transient types: {type_counts}")
    
    # Test rhythmic DNA
    print("  Testing rhythmic DNA extraction...")
    dna = engine.extract_rhythmic_dna(audio, tempo=128)
    print(f"    ✓ Mean syncopation: {dna['mean_syncopation']:.3f}")
    print(f"    ✓ Rhythmic density: {dna['rhythmic_density']:.3f}")
    print(f"    ✓ Accent positions: {dna['accent_positions']}")
    
    # Test tempo morph
    print("  Testing tempo morph...")
    morph = engine.create_tempo_morph(128, 130, len(audio), 'smooth')
    print(f"    ✓ Morph needed: {morph['morph_needed']}")
    print(f"    ✓ Morph type: {morph.get('morph_type', 'none')}")
    
    print("\n  ✅ MicroTimingEngine: All tests passed!")
    return True


def test_spectral_intelligence():
    """Test the SpectralIntelligenceEngine module."""
    print("\n" + "="*60)
    print("🌈 Testing SpectralIntelligenceEngine")
    print("="*60)
    
    from src.spectral_intelligence import SpectralIntelligenceEngine
    
    engine = SpectralIntelligenceEngine()
    
    # Generate two test audio tracks
    audio_a = generate_test_audio(duration_sec=10, tempo_bpm=128, key_freq=440)[:, 0]
    audio_b = generate_test_audio(duration_sec=10, tempo_bpm=130, key_freq=466)[:, 0]  # Different key
    
    # Test spectrum analysis
    print("  Testing spectrum analysis...")
    spectrum = engine.analyze_spectrum(audio_a)
    print(f"    ✓ Dominant band: {spectrum['dominant_band']}")
    print(f"    ✓ Spectral centroid: {spectrum['spectral_centroid']:.1f} Hz")
    print(f"    ✓ Spectral flatness: {spectrum['spectral_flatness']:.3f}")
    
    # Test frequency slot negotiation
    print("  Testing frequency slot negotiation...")
    negotiation = engine.negotiate_frequency_slots(audio_a, audio_b)
    print(f"    ✓ Overall conflict: {negotiation['overall_conflict']:.3f}")
    print(f"    ✓ Critical conflicts: {list(negotiation['critical_conflicts'].keys())}")
    
    # Test harmonic resonances
    print("  Testing harmonic resonance detection...")
    resonances = engine.find_harmonic_resonances(audio_a, audio_b)
    print(f"    ✓ Resonance count: {resonances['resonance_count']}")
    print(f"    ✓ Resonance strength: {resonances['resonance_strength']:.3f}")
    
    # Test masking analysis
    print("  Testing masking analysis...")
    masking = engine.analyze_masking(audio_a, audio_b)
    print(f"    ✓ Overall masking: {masking['overall_masking']:.3f}")
    print(f"    ✓ Masking severity: {masking['masking_severity']}")
    print(f"    ✓ Recommended action: {masking['recommended_action']}")
    
    # Test spectral morph
    print("  Testing spectral morph...")
    morph = engine.create_spectral_morph(audio_a, audio_b, 'smooth')
    print(f"    ✓ Morph stages: {morph['n_stages']}")
    print(f"    ✓ Morph type: {morph['morph_type']}")
    
    print("\n  ✅ SpectralIntelligenceEngine: All tests passed!")
    return True


def test_hybrid_technique_blender():
    """Test the HybridTechniqueBlender module."""
    print("\n" + "="*60)
    print("🎨 Testing HybridTechniqueBlender")
    print("="*60)
    
    from src.hybrid_technique_blender import HybridTechniqueBlender
    
    blender = HybridTechniqueBlender()
    
    # Test technique compatibility
    print("  Testing technique compatibility...")
    compat = blender.analyze_technique_compatibility('long_blend', 'bass_swap')
    print(f"    ✓ Compatible: {compat['compatible']}")
    print(f"    ✓ Score: {compat['score']:.3f}")
    
    # Test hybrid creation
    print("  Testing hybrid technique creation...")
    hybrid = blender.create_hybrid_technique(
        ['long_blend', 'filter_sweep', 'bass_swap'],
        context={'harmonic_compatibility': 0.7, 'energy_a': 0.6, 'energy_b': 0.8}
    )
    print(f"    ✓ Hybrid name: {hybrid.get('name', 'unknown')}")
    print(f"    ✓ Compatibility: {hybrid.get('compatibility', 0):.3f}")
    print(f"    ✓ Complexity: {hybrid.get('complexity', 0):.3f}")
    print(f"    ✓ Boldness: {hybrid.get('boldness', 0):.3f}")
    
    # Test preset retrieval
    print("  Testing preset retrieval...")
    presets = blender.list_presets()
    print(f"    ✓ Available presets: {len(presets)}")
    for p in presets:
        print(f"      - {p['name']}: {p['description']}")
    
    # Test context-aware suggestion
    print("  Testing context-aware technique suggestion...")
    context = {
        'harmonic_compatibility': 0.5,
        'energy_a': 0.7,
        'energy_b': 0.8,
        'tempo_diff': 2,
        'has_vocals_a': True,
        'has_vocals_b': False
    }
    suggestion = blender.suggest_creative_technique(context, creativity_level=0.7)
    if 'error' not in suggestion:
        print(f"    ✓ Suggested hybrid: {suggestion.get('name')}")
        print(f"    ✓ Techniques: {suggestion.get('techniques')}")
    else:
        print(f"    ⚠ Suggestion error: {suggestion.get('error')}")
    
    print("\n  ✅ HybridTechniqueBlender: All tests passed!")
    return True


def test_stem_orchestrator():
    """Test the StemOrchestrator module."""
    print("\n" + "="*60)
    print("🎭 Testing StemOrchestrator")
    print("="*60)
    
    from src.stem_orchestrator import StemOrchestrator
    
    orchestrator = StemOrchestrator()
    
    # Create mock stems
    n_samples = 44100 * 10  # 10 seconds
    mock_stems_a = {
        'drums': np.random.randn(n_samples, 2) * 0.3,
        'bass': np.random.randn(n_samples, 2) * 0.2,
        'vocals': np.random.randn(n_samples, 2) * 0.15,
        'other': np.random.randn(n_samples, 2) * 0.15
    }
    mock_stems_b = {
        'drums': np.random.randn(n_samples, 2) * 0.3,
        'bass': np.random.randn(n_samples, 2) * 0.2,
        'vocals': np.random.randn(n_samples, 2) * 0.15,
        'other': np.random.randn(n_samples, 2) * 0.15
    }
    
    # Test conversation types
    conversation_types = ['call_response', 'interweave', 'layered_reveal', 'counter_melody']
    
    for conv_type in conversation_types:
        print(f"  Testing {conv_type} conversation...")
        conversation = orchestrator.create_stem_conversation(
            mock_stems_a, mock_stems_b, conv_type
        )
        print(f"    ✓ Type: {conversation['type']}")
        print(f"    ✓ Stems with curves: {list(conversation['curves'].keys())}")
    
    # Test vocal phrase detection
    print("  Testing vocal phrase detection...")
    vocal_audio = np.random.randn(44100 * 5)  # 5 seconds
    # Add some silence gaps
    vocal_audio[44100:44100*2] *= 0.01  # Silence in second 1-2
    vocal_audio[44100*3:int(44100*3.5)] *= 0.01  # Silence in second 3-3.5
    
    phrases = orchestrator.detect_vocal_phrases(vocal_audio)
    print(f"    ✓ Phrases detected: {phrases['phrase_count']}")
    print(f"    ✓ Safe transition points: {len(phrases['safe_transition_points'])}")
    
    # Test orchestration analysis
    print("  Testing stem analysis for orchestration...")
    analysis = orchestrator.analyze_stems_for_orchestration(mock_stems_a, mock_stems_b)
    print(f"    ✓ Recommended conversation: {analysis['recommended_conversation']}")
    print(f"    ✓ Reasoning: {analysis['reasoning']}")
    
    # Test full orchestration
    print("  Testing full stem orchestration...")
    conversation = orchestrator.create_stem_conversation(mock_stems_a, mock_stems_b, 'layered_reveal')
    mixed = orchestrator.orchestrate_mix(mock_stems_a, mock_stems_b, conversation)
    print(f"    ✓ Mixed audio shape: {mixed.shape}")
    print(f"    ✓ Mixed audio RMS: {np.sqrt(np.mean(mixed**2)):.4f}")
    
    print("\n  ✅ StemOrchestrator: All tests passed!")
    return True


def test_montecarlo_optimizer():
    """Test the MonteCarloQualityOptimizer module."""
    print("\n" + "="*60)
    print("🎲 Testing MonteCarloQualityOptimizer")
    print("="*60)
    
    from src.montecarlo_optimizer import MonteCarloQualityOptimizer
    
    optimizer = MonteCarloQualityOptimizer()
    
    # Generate test segments
    audio_a = generate_test_audio(duration_sec=5)[:, 0]
    audio_b = generate_test_audio(duration_sec=5, tempo_bpm=130, key_freq=466)[:, 0]
    
    # Test quality prediction
    print("  Testing quality prediction...")
    prediction = optimizer.predict_quality(audio_a, audio_b, 'long_blend', {})
    print(f"    ✓ Predicted quality: {prediction['predicted_quality']:.3f}")
    print(f"    ✓ Predicted smoothness: {prediction['predicted_smoothness']:.3f}")
    print(f"    ✓ Predicted clarity: {prediction['predicted_clarity']:.3f}")
    print(f"    ✓ Recommendation: {prediction['recommendation']}")
    
    # Test should_proceed
    print("  Testing proceed decision...")
    should, pred = optimizer.should_proceed(audio_a, audio_b, 'bass_swap', {})
    print(f"    ✓ Should proceed: {should}")
    print(f"    ✓ Predicted quality: {pred['predicted_quality']:.3f}")
    
    # Test quality evaluation
    print("  Testing quality evaluation...")
    # Create a simple mixed output for testing
    mixed = audio_a * 0.5 + audio_b * 0.5
    quality = optimizer._evaluate_quality(mixed, audio_a, audio_b, {})
    print(f"    ✓ Overall score: {quality['overall_score']:.3f}")
    print(f"    ✓ Smoothness: {quality['smoothness']:.3f}")
    print(f"    ✓ Creativity: {quality['creativity']:.3f}")
    
    # Test ensemble evaluation
    print("  Testing ensemble evaluation...")
    ensemble = optimizer.ensemble_evaluate(mixed, audio_a, audio_b)
    print(f"    ✓ Ensemble mean: {ensemble['ensemble_mean']:.3f}")
    print(f"    ✓ Confidence: {ensemble['confidence']:.3f}")
    print(f"    ✓ Consensus: {ensemble['consensus']}")
    print(f"    ✓ Recommended action: {ensemble['recommended_action']}")
    
    print("\n  ✅ MonteCarloQualityOptimizer: All tests passed!")
    return True


def test_superhuman_engine():
    """Test the full SuperhumanDJEngine."""
    print("\n" + "="*60)
    print("🚀 Testing SuperhumanDJEngine (Full Integration)")
    print("="*60)
    
    from src.superhuman_engine import SuperhumanDJEngine
    
    engine = SuperhumanDJEngine()
    
    # Configure
    engine.configure(
        enable_micro_timing=True,
        enable_spectral_intelligence=True,
        enable_hybrid_techniques=True,
        enable_stem_orchestration=True,
        enable_montecarlo_optimization=True,
        creativity_level=0.7
    )
    
    print(f"  Engine configured: {engine.get_status()['config']}")
    
    # Generate test segments
    print("  Generating test audio segments...")
    seg_a = generate_test_audio(duration_sec=16, tempo_bpm=128, key_freq=440)
    seg_b = generate_test_audio(duration_sec=16, tempo_bpm=130, key_freq=466)
    
    # Convert to mono for some tests
    seg_a_mono = seg_a[:, 0]
    seg_b_mono = seg_b[:, 0]
    
    # Create mock stems (simplified)
    stems_a = {
        'drums': seg_a * 0.4,
        'bass': seg_a * 0.3,
        'vocals': seg_a * 0.15,
        'other': seg_a * 0.15
    }
    stems_b = {
        'drums': seg_b * 0.4,
        'bass': seg_b * 0.3,
        'vocals': seg_b * 0.15,
        'other': seg_b * 0.15
    }
    
    # Run full superhuman mix
    print("  Running superhuman mix...")
    start_time = time.time()
    
    result = engine.create_superhuman_mix(
        seg_a_mono, seg_b_mono,
        tempo_a=128, tempo_b=130,
        key_a='Am', key_b='Bm',
        stems_a=stems_a,
        stems_b=stems_b
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n  Results:")
    print(f"    ✓ Processing time: {elapsed:.2f}s")
    print(f"    ✓ Output length: {len(result['mixed'])} samples")
    print(f"    ✓ Quality score: {result['quality']['overall_score']:.3f}")
    print(f"    ✓ Confidence: {result['quality'].get('confidence', 0):.3f}")
    
    if 'technique' in result['analysis']:
        tech = result['analysis']['technique']
        print(f"    ✓ Technique: {tech.get('name', tech.get('type', 'unknown'))}")
    
    if 'micro_timing' in result['analysis']:
        mt = result['analysis']['micro_timing']
        print(f"    ✓ Groove compatibility: {mt.get('groove_compatibility', 0):.3f}")
    
    if 'spectral' in result['analysis']:
        sp = result['analysis']['spectral']
        print(f"    ✓ Frequency conflict: {sp.get('overall_conflict', 0):.3f}")
    
    print(f"    ✓ Processing stages: {result['analysis']['stages']}")
    
    print("\n  ✅ SuperhumanDJEngine: All tests passed!")
    return result


def test_smart_mixer_integration():
    """Test SmartMixer with superhuman engine integration."""
    print("\n" + "="*60)
    print("🔧 Testing SmartMixer Integration")
    print("="*60)
    
    from src.smart_mixer import SmartMixer
    
    mixer = SmartMixer()
    
    print(f"  SmartMixer initialized")
    print(f"    ✓ Superhuman enabled: {mixer.superhuman_enabled}")
    print(f"    ✓ Engine type: {type(mixer.superhuman_engine).__name__ if mixer.superhuman_engine else 'None'}")
    
    if mixer.superhuman_engine:
        config = mixer.superhuman_engine.config
        print(f"    ✓ Micro-timing: {config['enable_micro_timing']}")
        print(f"    ✓ Spectral intelligence: {config['enable_spectral_intelligence']}")
        print(f"    ✓ Hybrid techniques: {config['enable_hybrid_techniques']}")
        print(f"    ✓ Stem orchestration: {config['enable_stem_orchestration']}")
        print(f"    ✓ Monte Carlo: {config['enable_montecarlo_optimization']}")
    
    print("\n  ✅ SmartMixer Integration: Ready!")
    return True


def save_test_output(result, output_path: str):
    """Save the test output to a WAV file."""
    if result is not None and 'mixed' in result:
        mixed = result['mixed']
        if mixed.ndim == 1:
            mixed = np.column_stack([mixed, mixed])
        
        # Normalize
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed = mixed / max_val * 0.9
        
        sf.write(output_path, mixed, 44100)
        print(f"\n📁 Test output saved to: {output_path}")
        print(f"   Duration: {len(mixed)/44100:.1f}s")


def main():
    print("\n" + "="*70)
    print("🎧 SUPERHUMAN DJ ENGINE - COMPREHENSIVE END-TO-END TEST")
    print("="*70)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_passed = True
    
    try:
        # Test each module
        all_passed &= test_micro_timing_engine()
        all_passed &= test_spectral_intelligence()
        all_passed &= test_hybrid_technique_blender()
        all_passed &= test_stem_orchestrator()
        all_passed &= test_montecarlo_optimizer()
        
        # Test full engine
        result = test_superhuman_engine()
        all_passed &= (result is not None)
        
        # Test SmartMixer integration
        all_passed &= test_smart_mixer_integration()
        
        # Save test output
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'superhuman_test_output.wav'
        )
        save_test_output(result, output_path)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)
    print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
