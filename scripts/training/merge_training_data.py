#!/usr/bin/env python3
"""
Merge training data from multiple DJ mix analyses.
"""
import json
from pathlib import Path
from datetime import datetime


def merge_all_training_data():
    """Merge training data from all analyzed mixes."""
    
    data_dir = Path("data")
    
    # Find all analysis files
    analysis_files = [
        ("Heroes & Villains", data_dir / "deep_heroes_villains_analysis.json"),
        ("Yusuke Boiler Room Tokyo", data_dir / "yusuke_boiler_room_analysis.json"),
    ]
    
    all_samples = []
    sources = {}
    
    for name, path in analysis_files:
        if path.exists():
            with open(path) as f:
                transitions = json.load(f)
            
            print(f"Loading {name}: {len(transitions)} transitions")
            sources[name] = len(transitions)
            
            for i, t in enumerate(transitions):
                # Extract training sample
                tempo_out = t.get('tempo_outgoing', 0)
                if isinstance(tempo_out, str):
                    tempo_out = float(tempo_out.strip('[]'))
                elif isinstance(tempo_out, list):
                    tempo_out = tempo_out[0]
                
                tempo_in = t.get('tempo_incoming', 0)
                if isinstance(tempo_in, str):
                    tempo_in = float(tempo_in.strip('[]'))
                elif isinstance(tempo_in, list):
                    tempo_in = tempo_in[0]
                
                sample = {
                    "id": f"{name.replace(' ', '_').lower()}_{i+1:02d}",
                    "source_mix": name,
                    "from_track": t.get('from_track', ''),
                    "to_track": t.get('to_track', ''),
                    
                    "input": {
                        "tempo_A": tempo_out,
                        "key_A": t.get('key_outgoing', ''),
                        "energy_A": t.get('energy_before', 0),
                        
                        "tempo_B": tempo_in,
                        "key_B": t.get('key_incoming', ''),
                        "energy_B": t.get('energy_after', 0),
                        
                        "key_compatible": t.get('key_compatible', False),
                        "harmonic_tension": t.get('harmonic_tension', 0),
                        "spectral_smoothness": t.get('spectral_smoothness', 0),
                        "frequency_masking": t.get('frequency_masking', 0),
                    },
                    
                    "output": {
                        "technique": t.get('crossfade_type', 'cut'),
                        "duration_bars": t.get('bars_duration', 4),
                        "duration_sec": t.get('duration_sec', 10),
                        
                        "bass_swap": (t.get('bass_swap_detected') == 'True' or 
                                     t.get('bass_swap_detected') == True),
                        "low_cut_incoming": (t.get('low_cut_on_incoming') == 'True' or 
                                            t.get('low_cut_on_incoming') == True),
                        "high_cut_outgoing": (t.get('high_cut_on_outgoing') == 'True' or 
                                             t.get('high_cut_on_outgoing') == True),
                        
                        "energy_build": t.get('energy_build', False),
                        "energy_dip": t.get('energy_dip', False),
                    },
                    
                    "quality": {
                        "perceived_smoothness": t.get('perceived_smoothness', 0),
                        "technique_complexity": t.get('technique_complexity', 0),
                        "beat_aligned": (t.get('beat_aligned') == 'True' or 
                                        t.get('beat_aligned') == True),
                        "effectiveness_factors": t.get('effectiveness_factors', []),
                        "techniques_used": t.get('techniques_used', []),
                    },
                    
                    "curves": {
                        "volume_outgoing": t.get('volume_curve_outgoing', []),
                        "volume_incoming": t.get('volume_curve_incoming', []),
                        "bass_energy": t.get('bass_energy_curve', []),
                        "mid_energy": t.get('mid_energy_curve', []),
                        "high_energy": t.get('high_energy_curve', []),
                    }
                }
                
                all_samples.append(sample)
    
    # Save merged training data
    output = {
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "total_samples": len(all_samples),
        "sources": sources,
        "samples": all_samples
    }
    
    output_path = data_dir / "merged_training_data.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print("MERGED TRAINING DATA SUMMARY")
    print('='*60)
    print(f"\nTotal samples: {len(all_samples)}")
    print(f"\nBy source:")
    for source, count in sources.items():
        print(f"  • {source}: {count} transitions")
    
    print(f"\nSaved to: {output_path}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print('='*60)
    
    # Technique distribution
    techniques = {}
    for s in all_samples:
        t = s['output']['technique']
        techniques[t] = techniques.get(t, 0) + 1
    
    print("\nTechnique Distribution:")
    for t, c in sorted(techniques.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({c/len(all_samples)*100:.1f}%)")
    
    # EQ moves
    bass_swaps = sum(1 for s in all_samples if s['output']['bass_swap'])
    low_cuts = sum(1 for s in all_samples if s['output']['low_cut_incoming'])
    high_cuts = sum(1 for s in all_samples if s['output']['high_cut_outgoing'])
    
    print("\nEQ Move Frequency:")
    print(f"  Bass swap: {bass_swaps} ({bass_swaps/len(all_samples)*100:.1f}%)")
    print(f"  Low cut incoming: {low_cuts} ({low_cuts/len(all_samples)*100:.1f}%)")
    print(f"  High cut outgoing: {high_cuts} ({high_cuts/len(all_samples)*100:.1f}%)")
    
    # Energy patterns
    builds = sum(1 for s in all_samples if s['output']['energy_build'])
    dips = sum(1 for s in all_samples if s['output']['energy_dip'])
    
    print("\nEnergy Management:")
    print(f"  Energy builds: {builds} ({builds/len(all_samples)*100:.1f}%)")
    print(f"  Energy dips: {dips} ({dips/len(all_samples)*100:.1f}%)")
    
    # Quality scores
    smoothness = [s['quality']['perceived_smoothness'] for s in all_samples]
    print(f"\nQuality Scores:")
    print(f"  Avg smoothness: {sum(smoothness)/len(smoothness):.1%}")
    print(f"  Min smoothness: {min(smoothness):.1%}")
    print(f"  Max smoothness: {max(smoothness):.1%}")
    
    # Key compatibility
    compat = sum(1 for s in all_samples if s['input']['key_compatible'])
    print(f"\nHarmonic Matching:")
    print(f"  Key compatible: {compat}/{len(all_samples)} ({compat/len(all_samples)*100:.1f}%)")
    
    tensions = [s['input']['harmonic_tension'] for s in all_samples]
    print(f"  Avg tension: {sum(tensions)/len(tensions):.3f}")
    
    return output


if __name__ == '__main__':
    merge_all_training_data()

