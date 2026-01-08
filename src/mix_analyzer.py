"""
Analyze DJ mixes to identify tracks and extract transitions.
"""
import librosa
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from .song_analyzer import SongAnalyzer
from .transition_detector import TransitionDetector
from .transition_analyzer import TransitionAnalyzer


class MixAnalyzer:
    """Analyze DJ mixes to extract track information and transitions."""
    
    def __init__(self, song_analyzer: Optional[SongAnalyzer] = None):
        self.song_analyzer = song_analyzer or SongAnalyzer()
        self.transition_detector = TransitionDetector()
        self.transition_analyzer = TransitionAnalyzer()
    
    def analyze_mix(self, mix_audio_path: str, 
                    track_library: Optional[Dict[str, str]] = None,
                    mix_id: Optional[str] = None) -> Dict:
        """
        Analyze a DJ mix to identify tracks and transitions.
        
        Args:
            mix_audio_path: Path to the mix audio file
            track_library: Optional dict mapping song_id -> audio_path for matching
            mix_id: Optional mix identifier
        
        Returns:
            Complete mix analysis with identified tracks and transitions
        """
        print(f"Loading mix: {mix_audio_path}")
        y, sr = librosa.load(mix_audio_path, sr=44100)
        duration = len(y) / sr
        
        if mix_id is None:
            mix_id = Path(mix_audio_path).stem
        
        # Step 1: Detect transition points
        print("Detecting transitions...")
        transitions = self.transition_detector.detect_transitions(y, sr)
        
        # Step 2: Identify tracks (if library provided)
        identified_tracks = []
        if track_library:
            print("Identifying tracks from library...")
            identified_tracks = self._identify_tracks(y, sr, track_library, transitions)
        else:
            # If no library, estimate track boundaries from transitions
            identified_tracks = self._estimate_track_boundaries(transitions, duration)
        
        # Step 3: Analyze each transition in detail
        print(f"Analyzing {len(transitions)} transitions...")
        analyzed_transitions = []
        for i, transition in enumerate(transitions):
            print(f"  Analyzing transition {i+1}/{len(transitions)}...")
            
            # Get track info if available
            track_a_info = self._get_track_info(identified_tracks, transition['start_time_sec'])
            track_b_info = self._get_track_info(identified_tracks, transition['end_time_sec'])
            
            # Extract transition segment
            start_idx = int(transition['start_time_sec'] * sr)
            end_idx = int(transition['end_time_sec'] * sr)
            transition_audio = y[start_idx:end_idx]
            
            # Analyze transition
            transition_analysis = self.transition_analyzer.analyze_transition(
                transition_audio, sr,
                track_a_info=track_a_info,
                track_b_info=track_b_info,
                transition_start_sec=transition['start_time_sec']
            )
            
            analyzed_transitions.append({
                'transition': transition,
                'analysis': transition_analysis,
                'track_a_info': track_a_info,
                'track_b_info': track_b_info
            })
        
        return {
            'mix_id': mix_id,
            'mix_path': mix_audio_path,
            'duration_sec': duration,
            'sample_rate': sr,
            'detected_transitions': transitions,
            'identified_tracks': identified_tracks,
            'analyzed_transitions': analyzed_transitions,
            'total_tracks': len(identified_tracks),
            'total_transitions': len(transitions)
        }
    
    def _identify_tracks(self, mix_audio: np.ndarray, sr: int,
                        track_library: Dict[str, str],
                        transitions: List[Dict]) -> List[Dict]:
        """
        Attempt to identify tracks in mix using audio matching.
        This is a simplified version - in production you'd use proper fingerprinting.
        """
        identified = []
        
        # For each transition, try to match segments
        for i, transition in enumerate(transitions):
            # Extract segment before transition (likely track A)
            if i == 0:
                start_sec = 0
            else:
                start_sec = transitions[i-1]['end_time_sec']
            
            end_sec = transition['start_time_sec']
            segment = mix_audio[int(start_sec*sr):int(end_sec*sr)]
            
            # Try to match against library
            best_match = self._match_segment(segment, sr, track_library)
            if best_match:
                identified.append({
                    'song_id': best_match['song_id'],
                    'start_sec': start_sec,
                    'end_sec': end_sec,
                    'confidence': best_match['confidence']
                })
        
        return identified
    
    def _match_segment(self, segment: np.ndarray, sr: int,
                      track_library: Dict[str, str]) -> Optional[Dict]:
        """
        Match audio segment against library.
        Simplified version using chroma similarity.
        """
        if len(segment) < sr:  # Need at least 1 second
            return None
        
        # Compute chroma for segment
        segment_chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        segment_chroma_mean = np.mean(segment_chroma, axis=1)
        
        best_match = None
        best_score = 0.0
        
        # Compare against each track in library
        for song_id, audio_path in track_library.items():
            try:
                y_ref, sr_ref = librosa.load(audio_path, sr=sr, duration=30)
                ref_chroma = librosa.feature.chroma_stft(y=y_ref, sr=sr_ref)
                ref_chroma_mean = np.mean(ref_chroma, axis=1)
                
                # Cosine similarity
                similarity = np.dot(segment_chroma_mean, ref_chroma_mean) / (
                    np.linalg.norm(segment_chroma_mean) * np.linalg.norm(ref_chroma_mean)
                )
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = {
                        'song_id': song_id,
                        'confidence': float(similarity)
                    }
            except Exception as e:
                print(f"Error matching {song_id}: {e}")
                continue
        
        return best_match if best_score > 0.7 else None
    
    def _estimate_track_boundaries(self, transitions: List[Dict], 
                                   mix_duration: float) -> List[Dict]:
        """Estimate track boundaries from transitions."""
        tracks = []
        
        for i, transition in enumerate(transitions):
            start_sec = transitions[i-1]['end_time_sec'] if i > 0 else 0.0
            end_sec = transition['start_time_sec']
            
            tracks.append({
                'track_index': i,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'duration_sec': end_sec - start_sec
            })
        
        # Add final track
        if transitions:
            last_transition = transitions[-1]
            tracks.append({
                'track_index': len(transitions),
                'start_sec': last_transition['end_time_sec'],
                'end_sec': mix_duration,
                'duration_sec': mix_duration - last_transition['end_time_sec']
            })
        
        return tracks
    
    def _get_track_info(self, identified_tracks: List[Dict], 
                       time_sec: float) -> Optional[Dict]:
        """Get track info at given time."""
        for track in identified_tracks:
            if track['start_sec'] <= time_sec <= track.get('end_sec', float('inf')):
                return track
        return None

