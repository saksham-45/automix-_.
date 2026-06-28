#!/usr/bin/env python3
"""
Compare DJ styles between different mixes.
"""
import json
import sys
from pathlib import Path


def load_analysis(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def analyze_style(name: str, transitions: list) -> dict:
    """Extract style characteristics from transitions."""
    
    n = len(transitions)
    
    # Crossfade types
    types = {}
    for t in transitions:
        ctype = t.get('crossfade_type', 'unknown')
        types[ctype] = types.get(ctype, 0) + 1
    
    # Energy patterns
    builds = sum(1 for t in transitions if t.get('energy_build'))
    dips = sum(1 for t in transitions if t.get('energy_dip'))
    
    # EQ usage
    bass_swaps = sum(1 for t in transitions if 
                    t.get('bass_swap_detected') == 'True' or 
                    t.get('bass_swap_detected') == True)
    
    low_cuts = sum(1 for t in transitions if 
                  t.get('low_cut_on_incoming') == 'True' or 
                  t.get('low_cut_on_incoming') == True)
    
    high_cuts = sum(1 for t in transitions if 
                   t.get('high_cut_on_outgoing') == 'True' or 
                   t.get('high_cut_on_outgoing') == True)
    
    # Harmonic
    key_compat = sum(1 for t in transitions if t.get('key_compatible'))
    tensions = [t.get('harmonic_tension', 0) for t in transitions]
    avg_tension = sum(tensions) / len(tensions) if tensions else 0
    
    # Beat alignment
    beat_aligned = sum(1 for t in transitions if 
                      t.get('beat_aligned') == 'True' or 
                      t.get('beat_aligned') == True or
                      t.get('beat_aligned') == "True")
    
    # Smoothness
    smoothness = [t.get('perceived_smoothness', 0) for t in transitions]
    avg_smoothness = sum(smoothness) / len(smoothness) if smoothness else 0
    
    # Tempo range
    tempos = []
    for t in transitions:
        tempo = t.get('tempo_outgoing', 0)
        if isinstance(tempo, str):
            tempo = float(tempo.strip('[]'))
        elif isinstance(tempo, list):
            tempo = tempo[0]
        tempos.append(tempo)
    
    return {
        'name': name,
        'total_transitions': n,
        'crossfade_types': types,
        'energy_builds': builds,
        'energy_dips': dips,
        'bass_swaps': bass_swaps,
        'low_cuts': low_cuts,
        'high_cuts': high_cuts,
        'key_compatible': key_compat,
        'avg_tension': avg_tension,
        'beat_aligned': beat_aligned,
        'avg_smoothness': avg_smoothness,
        'tempo_range': (min(tempos), max(tempos)) if tempos else (0, 0)
    }


def compare_styles(style1: dict, style2: dict):
    """Compare two DJ styles."""
    
    print("="*80)
    print("DJ STYLE COMPARISON")
    print("="*80)
    
    print(f"\n{'':30} | {style1['name'][:20]:^20} | {style2['name'][:20]:^20}")
    print("-"*80)
    
    print(f"{'Total Transitions':30} | {style1['total_transitions']:^20} | {style2['total_transitions']:^20}")
    
    # Crossfade types
    print(f"\n{'CROSSFADE STYLE':30}")
    all_types = set(style1['crossfade_types'].keys()) | set(style2['crossfade_types'].keys())
    for ctype in all_types:
        v1 = style1['crossfade_types'].get(ctype, 0)
        v2 = style2['crossfade_types'].get(ctype, 0)
        p1 = f"{v1}/{style1['total_transitions']} ({v1/style1['total_transitions']*100:.0f}%)"
        p2 = f"{v2}/{style2['total_transitions']} ({v2/style2['total_transitions']*100:.0f}%)"
        print(f"  {ctype:28} | {p1:^20} | {p2:^20}")
    
    # Energy management
    print(f"\n{'ENERGY MANAGEMENT':30}")
    print(f"  {'Energy Builds':28} | {style1['energy_builds']:^20} | {style2['energy_builds']:^20}")
    print(f"  {'Energy Dips':28} | {style1['energy_dips']:^20} | {style2['energy_dips']:^20}")
    
    # EQ Techniques
    print(f"\n{'EQ TECHNIQUES':30}")
    print(f"  {'Bass Swaps':28} | {style1['bass_swaps']:^20} | {style2['bass_swaps']:^20}")
    print(f"  {'Low Cuts (incoming)':28} | {style1['low_cuts']:^20} | {style2['low_cuts']:^20}")
    print(f"  {'High Cuts (outgoing)':28} | {style1['high_cuts']:^20} | {style2['high_cuts']:^20}")
    
    # Harmonic
    print(f"\n{'HARMONIC MATCHING':30}")
    print(f"  {'Key Compatible':28} | {style1['key_compatible']}/{style1['total_transitions']:^16} | {style2['key_compatible']}/{style2['total_transitions']:^16}")
    t1 = f"{style1['avg_tension']:.3f}"
    t2 = f"{style2['avg_tension']:.3f}"
    print(f"  {'Avg Tension':28} | {t1:^20} | {t2:^20}")
    
    # Technical execution
    print(f"\n{'TECHNICAL EXECUTION':30}")
    print(f"  {'Beat Aligned':28} | {style1['beat_aligned']}/{style1['total_transitions']:^16} | {style2['beat_aligned']}/{style2['total_transitions']:^16}")
    s1 = f"{style1['avg_smoothness']:.0%}"
    s2 = f"{style2['avg_smoothness']:.0%}"
    print(f"  {'Avg Smoothness':28} | {s1:^20} | {s2:^20}")
    
    # Tempo
    print(f"\n{'TEMPO RANGE':30}")
    t1 = f"{style1['tempo_range'][0]:.0f}-{style1['tempo_range'][1]:.0f} BPM"
    t2 = f"{style2['tempo_range'][0]:.0f}-{style2['tempo_range'][1]:.0f} BPM"
    print(f"  {'Range':28} | {t1:^20} | {t2:^20}")
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    insights = []
    
    # Compare crossfade preferences
    s1_cut = style1['crossfade_types'].get('cut', 0) / style1['total_transitions']
    s2_cut = style2['crossfade_types'].get('cut', 0) / style2['total_transitions']
    
    if abs(s1_cut - s2_cut) > 0.2:
        if s1_cut > s2_cut:
            insights.append(f"• {style1['name']} uses MORE quick cuts ({s1_cut:.0%} vs {s2_cut:.0%})")
        else:
            insights.append(f"• {style2['name']} uses MORE quick cuts ({s2_cut:.0%} vs {s1_cut:.0%})")
    else:
        insights.append(f"• Both DJs prefer QUICK CUTS ({s1_cut:.0%} / {s2_cut:.0%})")
    
    # Compare bass swap usage
    s1_bass = style1['bass_swaps'] / style1['total_transitions']
    s2_bass = style2['bass_swaps'] / style2['total_transitions']
    
    if s1_bass > s2_bass + 0.1:
        insights.append(f"• {style1['name']} uses BASS SWAPS more frequently")
    elif s2_bass > s1_bass + 0.1:
        insights.append(f"• {style2['name']} uses BASS SWAPS more frequently")
    else:
        insights.append(f"• Both DJs use bass swaps sparingly (strategic use)")
    
    # Compare harmonic matching
    if style1['avg_tension'] < style2['avg_tension'] - 0.02:
        insights.append(f"• {style1['name']} maintains LOWER harmonic tension (cleaner blends)")
    elif style2['avg_tension'] < style1['avg_tension'] - 0.02:
        insights.append(f"• {style2['name']} maintains LOWER harmonic tension (cleaner blends)")
    else:
        insights.append(f"• Both DJs keep harmonic tension very LOW (avg {(style1['avg_tension']+style2['avg_tension'])/2:.2f})")
    
    # Energy management
    s1_builds = style1['energy_builds'] / style1['total_transitions']
    s2_builds = style2['energy_builds'] / style2['total_transitions']
    
    if s1_builds > s2_builds + 0.1:
        insights.append(f"• {style1['name']} focuses more on BUILDING energy")
    elif s2_builds > s1_builds + 0.1:
        insights.append(f"• {style2['name']} focuses more on BUILDING energy")
    
    # Tempo range
    s1_range = style1['tempo_range'][1] - style1['tempo_range'][0]
    s2_range = style2['tempo_range'][1] - style2['tempo_range'][0]
    
    if s1_range < 10:
        insights.append(f"• {style1['name']} maintains CONSISTENT tempo (~{style1['tempo_range'][0]:.0f} BPM)")
    if s2_range < 10:
        insights.append(f"• {style2['name']} maintains CONSISTENT tempo (~{style2['tempo_range'][0]:.0f} BPM)")
    elif s2_range > 20:
        insights.append(f"• {style2['name']} uses VARIED tempos ({style2['tempo_range'][0]:.0f}-{style2['tempo_range'][1]:.0f} BPM)")
    
    for insight in insights:
        print(insight)
    
    # Combined training value
    print("\n" + "="*80)
    print("COMBINED TRAINING DATA VALUE")
    print("="*80)
    
    total = style1['total_transitions'] + style2['total_transitions']
    print(f"\n  Total training samples: {total}")
    print(f"  Genres covered: Hip-hop/Trap + Electronic/Techno")
    print(f"  Tempo coverage: {min(style1['tempo_range'][0], style2['tempo_range'][0]):.0f} - {max(style1['tempo_range'][1], style2['tempo_range'][1]):.0f} BPM")
    print(f"  Average quality: {(style1['avg_smoothness'] + style2['avg_smoothness'])/2:.0%}")


def main():
    analyses = [
        ("Heroes & Villains (Hip-hop)", "data/deep_heroes_villains_analysis.json"),
        ("Yusuke Boiler Room (Electronic)", "data/yusuke_boiler_room_analysis.json")
    ]
    
    styles = []
    for name, path in analyses:
        if Path(path).exists():
            transitions = load_analysis(path)
            style = analyze_style(name, transitions)
            styles.append(style)
    
    if len(styles) >= 2:
        compare_styles(styles[0], styles[1])
    elif len(styles) == 1:
        print(f"Only one analysis found: {styles[0]['name']}")
        print(json.dumps(styles[0], indent=2))


if __name__ == '__main__':
    main()

