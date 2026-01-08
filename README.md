# AI DJ Transition Analysis System

A comprehensive system for analyzing songs and DJ transitions to train AI models for automatic DJ mixing.

## Overview

This system extracts **everything a human DJ perceives** from audio and stores it as structured data. No audio files are stored - only the extracted knowledge.

### What It Does

1. **Song Analysis**: Extracts 100+ features from any song (tempo, key, energy, spectrum, structure, etc.)
2. **Mix Analysis**: Identifies tracks in DJ mixes and detects transition points
3. **Transition Analysis**: Analyzes HOW transitions were executed (techniques, EQ, volume curves)
4. **Knowledge Storage**: Stores all data in a structured database for AI training

## Project Structure

```
.
├── src/
│   ├── song_analyzer.py          # Complete song feature extraction
│   ├── transition_detector.py    # Detect transitions in mixes
│   ├── transition_analyzer.py    # Analyze transition techniques
│   ├── mix_analyzer.py           # Complete mix analysis pipeline
│   ├── database.py               # Database schema and operations
│   ├── schema_validator.py       # JSON schema validation
│   └── utils.py                  # Utility functions
├── scripts/
│   ├── analyze_song.py           # Analyze a single song
│   ├── analyze_mix.py            # Analyze a DJ mix
│   └── batch_analyze.py          # Batch process multiple files
├── examples/
│   └── example_usage.py          # Usage examples
├── config/
│   └── config.yaml               # Configuration
├── data/
│   ├── songs/                    # Song analysis outputs
│   ├── mixes/                    # Mix analysis outputs
│   ├── transitions/              # Transition analysis outputs
│   ├── features/                 # Feature JSON files
│   └── audio_snippets/           # Transition audio snippets
└── database/
    └── transitions.db            # SQLite database (created on first run)

```

## Installation

```bash
pip install -r requirements.txt
```

**Note**: Some packages may require additional setup:
- `essentia`: May need compilation or use pre-built wheels
- `demucs`: Requires PyTorch
- `madmom`: May need system dependencies

## Quick Start

### Analyze a Single Song

```python
from src.analyzers.song_analyzer import SongAnalyzer

analyzer = SongAnalyzer()
features = analyzer.analyze("path/to/song.wav")
# Features is a complete dict with all extracted features
```

### Analyze a DJ Mix

```python
from src.analyzers.transition_detector import TransitionDetector
from src.analyzers.transition_analyzer import TransitionAnalyzer

detector = TransitionDetector()
transitions = detector.detect_transitions("path/to/mix.wav")

analyzer = TransitionAnalyzer()
for transition in transitions:
    analysis = analyzer.analyze_transition(
        mix_audio=transition['audio_segment'],
        transition_start=transition['start'],
        transition_end=transition['end']
    )
```

## Features Extracted

### Song Features
- Tempo & rhythm (BPM, beat grid, microtiming)
- Harmony (key, chords, pitch distribution)
- Energy & dynamics (loudness, RMS curves)
- Spectral features (frequency bands, brightness, warmth)
- Timbre & texture (MFCC, descriptors)
- Song structure (sections, phrases, mix points)
- Vocal analysis (presence, segments, spectral signature)
- Stereo & spatial features
- Learned embeddings (semantic understanding)

### Transition Features
- Technique classification (long blend, quick cut, bass swap, etc.)
- Volume curves (fade in/out)
- EQ automation (bass swaps, filter sweeps)
- Beat alignment quality
- Energy flow during transition
- Spectral balance maintenance
- Quality assessment scores

## Database Schema

The system uses SQLite with JSON storage for complex nested data. See `src/database/models.py` for the complete schema.

## Output Format

All analysis outputs are JSON files following the schemas in `src/schemas/`. Each song analysis is ~50-500KB depending on detail level.

## License

MIT

