#!/usr/bin/env python3
"""
Extract learnable patterns from deep transition analysis.
This creates the actual training data format for AI learning.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_patterns(analysis_path: str):
    """Extract patterns from the deep analysis."""
    
    with open(analysis_path) as f:
        transitions = json.load(f)
    
    print("="*70)
    print("DEEP PATTERN ANALYSIS: Heroes & Villains Mix")
    print("="*70)
    
    # 1. TRANSITION TECHNIQUES BREAKDOWN
    print("\n" + "─"*70)
    print("1. TRANSITION TECHNIQUE PATTERNS")
    print("─"*70)
    
    for i, t in enumerate(transitions, 1):
        print(f"\n  Transition {i}: {t['from_track'][:30]}...")
        print(f"              → {t['to_track'][:30]}...")
        print(f"  ┌─────────────────────────────────────────────────────────────")
        print(f"  │ Type: {t['crossfade_type'].upper()}")
        print(f"  │ Duration: {t['duration_sec']:.1f}s (~{t['bars_duration']} bars)")
        
        # EQ Analysis
        eq_moves = []
        if t['bass_swap_detected'] == 'True' or t['bass_swap_detected'] == True:
            eq_moves.append("BASS SWAP")
        if t['low_cut_on_incoming'] == 'True' or t['low_cut_on_incoming'] == True:
            eq_moves.append("Low cut on incoming")
        if t['high_cut_on_outgoing'] == 'True' or t['high_cut_on_outgoing'] == True:
            eq_moves.append("High cut on outgoing")
        if t['eq_automation_detected'] == 'True' or t['eq_automation_detected'] == True:
            eq_moves.append("EQ automation")
        
        print(f"  │ EQ Moves: {', '.join(eq_moves) if eq_moves else 'None detected'}")
        
        # Energy
        e_before = t['energy_before']
        e_during = t['energy_during']
        e_after = t['energy_after']
        
        if t['energy_build']:
            energy_pattern = "BUILD ↗ (builds energy)"
        elif t['energy_dip']:
            energy_pattern = "DIP ↘↗ (momentary dip)"
        elif e_after > e_before:
            energy_pattern = "RISE ↗"
        elif e_after < e_before:
            energy_pattern = "DROP ↘"
        else:
            energy_pattern = "MAINTAIN →"
        
        print(f"  │ Energy: {energy_pattern}")
        print(f"  │         ({e_before:.3f} → {e_during:.3f} → {e_after:.3f})")
        
        # Harmony
        key_out = t['key_outgoing']
        key_in = t['key_incoming']
        tension = t['harmonic_tension']
        compat = "✓ compatible" if t['key_compatible'] else "✗ tension"
        
        print(f"  │ Key: {key_out} → {key_in} ({compat}, tension={tension:.2f})")
        print(f"  │ Smoothness Score: {t['perceived_smoothness']:.0%}")
        print(f"  └─────────────────────────────────────────────────────────────")
    
    # 2. COMMON PATTERNS
    print("\n" + "─"*70)
    print("2. EXTRACTED PATTERNS (FOR AI TRAINING)")
    print("─"*70)
    
    # Crossfade types
    crossfade_types = [t['crossfade_type'] for t in transitions]
    cut_count = crossfade_types.count('cut')
    print(f"\n  Crossfade Distribution:")
    print(f"    • Quick cuts: {cut_count}/{len(transitions)} ({cut_count/len(transitions)*100:.0f}%)")
    print(f"    • Linear blends: {crossfade_types.count('linear')}/{len(transitions)}")
    print(f"    • INSIGHT: This mix prefers QUICK CUTS over long blends")
    
    # Bass handling
    bass_swaps = sum(1 for t in transitions if t['bass_swap_detected'] == 'True' or t['bass_swap_detected'] == True)
    print(f"\n  Bass Swap Usage: {bass_swaps}/{len(transitions)}")
    print(f"    • INSIGHT: Bass swaps are used strategically, not every transition")
    
    # Energy patterns
    builds = sum(1 for t in transitions if t['energy_build'])
    dips = sum(1 for t in transitions if t['energy_dip'])
    print(f"\n  Energy Management:")
    print(f"    • Energy builds: {builds} transitions")
    print(f"    • Energy dips: {dips} transitions")
    print(f"    • INSIGHT: Mix maintains or builds energy - avoids killing the vibe")
    
    # Key relationships
    compatible = sum(1 for t in transitions if t['key_compatible'])
    print(f"\n  Harmonic Compatibility: {compatible}/{len(transitions)}")
    avg_tension = sum(t['harmonic_tension'] for t in transitions) / len(transitions)
    print(f"    • Average tension: {avg_tension:.2f}")
    print(f"    • INSIGHT: Keeps tension LOW (<0.15) for smooth blends")
    
    # 3. TRAINING DATA STRUCTURE
    print("\n" + "─"*70)
    print("3. TRAINING DATA STRUCTURE (What AI learns)")
    print("─"*70)
    
    training_features = {
        "input_features": {
            "song_A_state": {
                "tempo": "BPM value",
                "key": "Detected key",
                "energy": "0-1 scale",
                "spectral_centroid": "Hz",
                "bass_energy": "normalized",
                "current_section": "verse/chorus/etc"
            },
            "song_B_state": {
                "same as song_A_state": "..."
            },
            "compatibility_metrics": {
                "tempo_diff": "BPM difference",
                "key_compatibility": "0-1 (1=perfect 5th)",
                "energy_match": "0-1",
                "spectral_similarity": "0-1"
            }
        },
        "output_labels": {
            "technique": "cut/blend/bass_swap",
            "duration_bars": "number of bars",
            "crossfade_curve": "linear/exponential/s_curve",
            "eq_moves": ["bass_swap", "high_cut_outgoing"],
            "energy_target": "build/maintain/dip"
        },
        "quality_labels": {
            "smoothness_score": "0-1 (human rated)",
            "effectiveness_factors": ["list of what worked"]
        }
    }
    
    print(json.dumps(training_features, indent=2))
    
    # 4. KEY LEARNINGS
    print("\n" + "─"*70)
    print("4. KEY LEARNINGS FOR AI DJ")
    print("─"*70)
    
    print("""
  ✓ TECHNIQUE SELECTION RULES:
    • Use QUICK CUTS when energy is high and tracks have similar intensity
    • Use BASS SWAP when transitioning between tracks with different bass lines
    • Use LINEAR BLEND when energy needs to be built gradually
    
  ✓ EQ AUTOMATION RULES:
    • Cut HIGH frequencies on OUTGOING track (removes brightness)
    • Cut LOW frequencies on INCOMING track initially (avoids bass clash)
    • Swap bass at the right moment (usually on a downbeat)
    
  ✓ HARMONIC RULES:
    • Same key or compatible keys (5ths, 4ths, relative major/minor)
    • Keep harmonic tension below 0.3 during blend
    • When keys don't match, use quick cuts instead of long blends
    
  ✓ ENERGY RULES:
    • Never let energy drop significantly (kills the vibe)
    • Build energy through transitions when possible
    • If energy must dip, make it quick and recover immediately
    
  ✓ TIMING RULES:
    • Transitions should land on DOWNBEATS (beat 1 of a bar)
    • Duration should be in multiples of 4 bars (4, 8, 16)
    • Match phrase boundaries (verse/chorus transitions)
    """)
    
    # 5. EXPORT SIMPLIFIED TRAINING DATA
    print("\n" + "─"*70)
    print("5. EXPORTED TRAINING SAMPLES")
    print("─"*70)
    
    training_samples = []
    for t in transitions:
        sample = {
            "input": {
                "tempo_A": float(str(t['tempo_outgoing']).strip('[]')),
                "key_A": t['key_outgoing'],
                "energy_A": t['energy_before'],
                "tempo_B": float(str(t['tempo_incoming']).strip('[]')),
                "key_B": t['key_incoming'],
                "energy_B": t['energy_after'],
                "key_compatible": t['key_compatible'],
                "harmonic_tension": t['harmonic_tension']
            },
            "output": {
                "technique": t['crossfade_type'],
                "duration_bars": t['bars_duration'],
                "bass_swap": t['bass_swap_detected'] == 'True' or t['bass_swap_detected'] == True,
                "high_cut_outgoing": t['high_cut_on_outgoing'] == 'True' or t['high_cut_on_outgoing'] == True,
                "low_cut_incoming": t['low_cut_on_incoming'] == 'True' or t['low_cut_on_incoming'] == True,
                "energy_target": "build" if t['energy_build'] else ("dip" if t['energy_dip'] else "maintain")
            },
            "quality": {
                "smoothness": t['perceived_smoothness'],
                "complexity": t['technique_complexity']
            }
        }
        training_samples.append(sample)
    
    # Save training data
    output_path = Path(analysis_path).parent / "training_samples.json"
    with open(output_path, 'w') as f:
        json.dump(training_samples, f, indent=2)
    
    print(f"  Saved {len(training_samples)} training samples to: {output_path}")
    
    # Print one sample
    print("\n  SAMPLE TRAINING DATA:")
    print(json.dumps(training_samples[0], indent=4))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_patterns(sys.argv[1])
    else:
        analyze_patterns('data/deep_heroes_villains_analysis.json')

