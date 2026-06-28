# How to Start Extraction - Quick Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
pip install --no-deps 'git+https://github.com/adefossez/demucs.git'
```

The second command installs **demucs** from the [adefossez/demucs](https://github.com/adefossez/demucs) fork (avoids lameenc on Python 3.14; stem separation uses WAV only).

**Note**: If you don't have `requirements.txt`, install these packages:
```bash
pip install librosa numpy scipy
```

## Step 2: Prepare Your Premixed Album

Have your premixed album audio file ready. Supported formats:
- `.wav` (recommended)
- `.mp3`
- `.flac`
- `.m4a`

## Step 3: Run the Analysis

### Basic Usage (Minimum Required)

```bash
python scripts/analyze_premixed_album.py path/to/your/mix.wav
```

### With Metadata (Recommended)

```bash
python scripts/analyze_premixed_album.py path/to/your/mix.wav \
    --title "Essential Mix 2024" \
    --dj "Charlotte de Witte" \
    --genre "techno"
```

### Full Example

```bash
python scripts/analyze_premixed_album.py ~/Music/boiler_room_mix.wav \
    --title "Boiler Room Set" \
    --dj "Fred Again.." \
    --genre "electronic" \
    --output-dir data/my_analyses
```

## What Happens

1. **Loads the mix** - Reads your audio file
2. **Detects transitions** - Finds where tracks transition
3. **Analyzes each transition** - Extracts volume curves, EQ automation, techniques
4. **Saves analysis** - Creates JSON file with all data
5. **Saves to database** - Stores in SQLite database (optional)
6. **Exports training data** - Creates ML-ready training examples (optional)

## Output Locations

All outputs go to `data/premixed_albums/` by default:

```
data/premixed_albums/
├── {mix_id}_analysis.json          # Complete analysis
└── {mix_id}_training_data/         # Training examples
    ├── transition_001.json
    ├── transition_002.json
    ├── ...
    └── summary.json
```

## Example Output

After running, you'll see:

```
======================================================================
ANALYZING PREMIXED ALBUM
======================================================================
Mix: ~/Music/mix.wav
Title: Essential Mix 2024
DJ: Charlotte de Witte

Loading mix audio...
✓ Loaded: 120.5 minutes

Step 1: Detecting transitions...
✓ Detected 15 transitions

Step 2: Analyzing transitions in detail...
  Analyzing transition 1/15 (218.5s - 252.3s)...
  Analyzing transition 2/15 (465.0s - 512.3s)...
  ...
✓ Analyzed 15 transitions

Step 3: Estimating track boundaries...
✓ Estimated 16 tracks

Step 4: Saving analysis...
  Output: data/premixed_albums/Essential_Mix_2024_analysis.json
✓ Saved analysis

Step 5: Saving to database...
✓ Saved to database: data/music_analysis.db

Step 6: Extracting training data...
✓ Exported 15 training examples
  Output: data/premixed_albums/Essential_Mix_2024_training_data

======================================================================
ANALYSIS COMPLETE
======================================================================
Mix ID: Essential_Mix_2024
Duration: 120.5 minutes
Transitions detected: 15
Tracks estimated: 16
Analysis saved: data/premixed_albums/Essential_Mix_2024_analysis.json
Database: data/music_analysis.db
Training data: data/premixed_albums/Essential_Mix_2024_training_data
```

## Command Line Options

```bash
python scripts/analyze_premixed_album.py mix.wav [OPTIONS]

Required:
  mix_path              Path to premixed album audio file

Optional:
  --title TITLE         Title of the mix/album
  --dj DJ_NAME          Name of the DJ
  --genre GENRE         Genre of the mix
  --track-library PATH  JSON file mapping song_id -> audio_path
  --no-db               Skip saving to database
  --no-training          Skip exporting training data
  --db-path PATH         Database path (default: data/music_analysis.db)
  --output-dir PATH      Output directory (default: data/premixed_albums)
```

## Troubleshooting

### "Module not found" errors
```bash
pip install librosa numpy scipy
```

### "File not found" error
Make sure you're running from the project root directory:
```bash
cd /path/to/project
python scripts/analyze_premixed_album.py mix.wav
```

### Processing is slow
- Analysis takes time! A 2-hour mix can take 10-30 minutes
- This is normal - it's analyzing every transition in detail

### Out of memory
- Try processing shorter mixes first
- Close other applications

## Next Steps

After extraction:
1. Check the analysis JSON file to see what was extracted
2. Review training data in the `_training_data/` folder
3. Use the training data to train your AI model
4. Query the database for compatible tracks

## Quick Test

Test with a short mix first:
```bash
python scripts/analyze_premixed_album.py test_mix.wav --title "Test Mix"
```

