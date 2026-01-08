"""
JSON Schema validator for song analysis and transition data.
Ensures data consistency for AI training.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path


class SchemaValidator:
    """Validates song analysis and transition data against expected schemas."""
    
    @staticmethod
    def validate_song_analysis(data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a complete song analysis JSON.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Required top-level fields
        required_fields = [
            'song_id', 'duration_sec', 'sample_rate',
            'tempo', 'key', 'energy', 'spectrum', 'structure'
        ]
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate tempo structure
        if 'tempo' in data:
            tempo = data['tempo']
            if 'bpm' not in tempo:
                errors.append("tempo.bpm is required")
            if 'beat_grid' in tempo:
                if 'beat_positions_sec' not in tempo['beat_grid']:
                    errors.append("tempo.beat_grid.beat_positions_sec is required")
        
        # Validate key structure
        if 'key' in data:
            key = data['key']
            if 'estimated_key' not in key and 'key' not in key:
                errors.append("key.estimated_key or key.key is required")
            if 'mode' not in key:
                errors.append("key.mode is required")
            if 'camelot' not in key:
                errors.append("key.camelot is required")
        
        # Validate energy structure
        if 'energy' in data:
            energy = data['energy']
            if 'energy_curve' in energy:
                curve = energy['energy_curve']
                if 'times_sec' not in curve or 'values' not in curve:
                    errors.append("energy.energy_curve requires times_sec and values")
        
        # Validate spectrum structure
        if 'spectrum' in data:
            spectrum = data['spectrum']
            if 'frequency_bands' not in spectrum:
                errors.append("spectrum.frequency_bands is required")
        
        # Validate structure
        if 'structure' in data:
            structure = data['structure']
            if 'sections' not in structure:
                errors.append("structure.sections is required")
            else:
                for i, section in enumerate(structure['sections']):
                    if 'type' not in section:
                        errors.append(f"structure.sections[{i}].type is required")
                    if 'start_sec' not in section:
                        errors.append(f"structure.sections[{i}].start_sec is required")
                    if 'end_sec' not in section:
                        errors.append(f"structure.sections[{i}].end_sec is required")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_transition_analysis(data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a transition analysis JSON.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Required top-level fields
        required_fields = [
            'transition_id', 'track_a_features', 'track_b_features',
            'transition_execution'
        ]
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate track features
        for track in ['track_a_features', 'track_b_features']:
            if track in data:
                track_data = data[track]
                required_track_fields = ['song_id', 'bpm', 'key', 'camelot']
                for field in required_track_fields:
                    if field not in track_data:
                        errors.append(f"{track}.{field} is required")
        
        # Validate transition execution
        if 'transition_execution' in data:
            exec_data = data['transition_execution']
            if 'duration_sec' not in exec_data:
                errors.append("transition_execution.duration_sec is required")
            if 'technique_primary' not in exec_data:
                errors.append("transition_execution.technique_primary is required")
            if 'volume_curves' in exec_data:
                curves = exec_data['volume_curves']
                if 'track_a_gain_db' not in curves or 'track_b_gain_db' not in curves:
                    errors.append("volume_curves requires track_a_gain_db and track_b_gain_db")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_file(file_path: Path, is_transition: bool = False) -> tuple[bool, List[str]]:
        """
        Validate a JSON file.
        
        Args:
            file_path: Path to JSON file
            is_transition: If True, validate as transition; else validate as song
        
        Returns:
            (is_valid, list_of_errors)
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if is_transition:
                return SchemaValidator.validate_transition_analysis(data)
            else:
                return SchemaValidator.validate_song_analysis(data)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {str(e)}"]
        except Exception as e:
            return False, [f"Error reading file: {str(e)}"]


def validate_directory(directory: Path, is_transition: bool = False) -> Dict[str, tuple[bool, List[str]]]:
    """
    Validate all JSON files in a directory.
    
    Returns:
        Dict mapping file paths to (is_valid, errors) tuples
    """
    results = {}
    
    json_files = list(directory.glob("*.json"))
    
    for json_file in json_files:
        is_valid, errors = SchemaValidator.validate_file(json_file, is_transition)
        results[str(json_file)] = (is_valid, errors)
    
    return results

