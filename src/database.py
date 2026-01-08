"""
Database schema and utilities for storing song and transition analysis data.
"""
import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class MusicDatabase:
    """SQLite database for storing song analyses and transitions."""
    
    def __init__(self, db_path: str = "data/music_analysis.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Songs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                song_id TEXT PRIMARY KEY,
                title TEXT,
                artist TEXT,
                duration_sec REAL,
                sample_rate INTEGER,
                analyzed_at TIMESTAMP,
                analysis_json_path TEXT,
                bpm REAL,
                key TEXT,
                camelot TEXT,
                energy_mean REAL,
                has_vocals BOOLEAN,
                genre TEXT,
                embedding_vector BLOB
            )
        """)
        
        # Mixes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mixes (
                mix_id TEXT PRIMARY KEY,
                title TEXT,
                dj TEXT,
                genre TEXT,
                duration_sec REAL,
                source_url TEXT,
                analyzed_at TIMESTAMP
            )
        """)
        
        # Transitions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transitions (
                transition_id TEXT PRIMARY KEY,
                mix_id TEXT REFERENCES mixes(mix_id),
                track_a_id TEXT REFERENCES songs(song_id),
                track_b_id TEXT REFERENCES songs(song_id),
                
                start_time_sec REAL,
                end_time_sec REAL,
                duration_sec REAL,
                duration_bars INTEGER,
                
                technique TEXT,
                technique_confidence REAL,
                
                bpm_a REAL,
                bpm_b REAL,
                key_a TEXT,
                key_b TEXT,
                camelot_distance INTEGER,
                
                has_bass_swap BOOLEAN,
                has_filter_sweep BOOLEAN,
                has_echo_out BOOLEAN,
                
                beat_alignment_quality REAL,
                energy_delta REAL,
                
                volume_curves_json TEXT,
                eq_automation_json TEXT,
                transition_analysis_json_path TEXT,
                
                quality_score REAL,
                analyzed_at TIMESTAMP
            )
        """)
        
        # Create indexes for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_bpm ON songs(bpm)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_key ON songs(key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_camelot ON songs(camelot)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_energy ON songs(energy_mean)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_genre ON songs(genre)")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transitions_technique ON transitions(technique)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transitions_bpm ON transitions(bpm_a, bpm_b)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transitions_key ON transitions(key_a, key_b)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transitions_duration ON transitions(duration_bars)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transitions_quality ON transitions(quality_score)")
        
        conn.commit()
        conn.close()
    
    def save_song_analysis(self, song_id: str, analysis: Dict[str, Any], 
                          analysis_json_path: Optional[str] = None):
        """Save song analysis to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract key fields for quick queries
        tempo = analysis.get('tempo', {})
        harmony = analysis.get('harmony', {})
        energy = analysis.get('energy', {})
        vocals = analysis.get('vocals', {})
        embeddings = analysis.get('embeddings', {})
        
        # Get embedding vector if available
        embedding_vector = None
        if 'clap_embedding' in embeddings:
            import numpy as np
            embedding_vector = np.array(embeddings['clap_embedding']).tobytes()
        
        cursor.execute("""
            INSERT OR REPLACE INTO songs (
                song_id, title, artist, duration_sec, sample_rate,
                analyzed_at, analysis_json_path, bpm, key, camelot,
                energy_mean, has_vocals, genre, embedding_vector
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            song_id,
            analysis.get('metadata', {}).get('title'),
            analysis.get('metadata', {}).get('artist'),
            analysis.get('metadata', {}).get('duration_sec'),
            analysis.get('metadata', {}).get('sample_rate'),
            datetime.now().isoformat(),
            analysis_json_path,
            tempo.get('bpm'),
            harmony.get('key', {}).get('estimated_key'),
            harmony.get('key', {}).get('camelot'),
            energy.get('energy_statistics', {}).get('mean'),
            vocals.get('has_vocals', False),
            analysis.get('embeddings', {}).get('semantic_tags', {}).get('genre', [None])[0] if isinstance(analysis.get('embeddings', {}).get('semantic_tags', {}).get('genre'), list) else analysis.get('embeddings', {}).get('semantic_tags', {}).get('genre'),
            embedding_vector
        ))
        
        conn.commit()
        conn.close()
    
    def save_transition(self, transition_id: str, transition_data: Dict[str, Any],
                       transition_json_path: Optional[str] = None):
        """Save transition analysis to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        track_a = transition_data.get('track_a_features', {})
        track_b = transition_data.get('track_b_features', {})
        transition = transition_data.get('transition_execution', {})
        compatibility = transition_data.get('compatibility_metrics', {})
        quality = transition_data.get('quality_assessment', {})
        
        cursor.execute("""
            INSERT OR REPLACE INTO transitions (
                transition_id, mix_id, track_a_id, track_b_id,
                start_time_sec, end_time_sec, duration_sec, duration_bars,
                technique, technique_confidence,
                bpm_a, bpm_b, key_a, key_b, camelot_distance,
                has_bass_swap, has_filter_sweep, has_echo_out,
                beat_alignment_quality, energy_delta,
                volume_curves_json, eq_automation_json,
                transition_analysis_json_path, quality_score, analyzed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transition_id,
            transition_data.get('context', {}).get('source_mix_id'),
            track_a.get('song_id'),
            track_b.get('song_id'),
            transition_data.get('transition', {}).get('start_time_sec'),
            transition_data.get('transition', {}).get('end_time_sec'),
            transition.get('duration_sec'),
            transition.get('duration_bars'),
            transition.get('technique_primary'),
            transition.get('technique_confidence', 0.0),
            track_a.get('bpm'),
            track_b.get('bpm'),
            track_a.get('key'),
            track_b.get('key'),
            compatibility.get('camelot_distance'),
            'bass_swap' in transition.get('technique_secondary', []),
            'filter_sweep' in transition.get('technique_secondary', []),
            'echo_out' in transition.get('technique_secondary', []),
            quality.get('beat_match_quality'),
            compatibility.get('energy_delta'),
            json.dumps(transition.get('volume_curves', {})),
            json.dumps(transition.get('eq_automation', {})),
            transition_json_path,
            quality.get('overall_transition_quality'),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def query_compatible_songs(self, bpm: float, key: str, camelot: str,
                               energy: float, limit: int = 20) -> List[Dict]:
        """Query songs compatible for mixing."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Camelot wheel compatibility
        camelot_num = int(camelot[0]) if camelot and camelot[0].isdigit() else None
        compatible_camelots = []
        if camelot_num:
            compatible_camelots = [
                f"{camelot_num}{camelot[1]}",  # Same
                f"{(camelot_num % 12) + 1}{camelot[1]}",  # +1
                f"{((camelot_num - 2) % 12) + 1}{camelot[1]}",  # -1
                f"{camelot_num}{'B' if camelot[1] == 'A' else 'A'}"  # Relative major/minor
            ]
        
        cursor.execute("""
            SELECT * FROM songs
            WHERE bpm BETWEEN ? AND ?
            AND (camelot IN ({}) OR key = ?)
            AND energy_mean BETWEEN ? AND ?
            ORDER BY ABS(bpm - ?) ASC
            LIMIT ?
        """.format(','.join(['?'] * len(compatible_camelots))), 
        [bpm * 0.94, bpm * 1.06] + compatible_camelots + [key, energy - 0.2, energy + 0.2, bpm, limit])
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def query_transitions_by_technique(self, technique: str, limit: int = 100) -> List[Dict]:
        """Query transitions by technique."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM transitions
            WHERE technique = ?
            ORDER BY quality_score DESC
            LIMIT ?
        """, (technique, limit))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_song_analysis(self, song_id: str) -> Optional[Dict]:
        """Retrieve full song analysis from JSON file."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT analysis_json_path FROM songs WHERE song_id = ?", (song_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            with open(row[0], 'r') as f:
                return json.load(f)
        return None
    
    def get_transition_analysis(self, transition_id: str) -> Optional[Dict]:
        """Retrieve full transition analysis from JSON file."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT transition_analysis_json_path FROM transitions WHERE transition_id = ?", 
                      (transition_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            with open(row[0], 'r') as f:
                return json.load(f)
        return None


def compute_song_id(audio_path: str) -> str:
    """Compute unique song ID from audio file."""
    # Use file hash + metadata
    import hashlib
    with open(audio_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    return file_hash[:16]  # Use first 16 chars as ID

