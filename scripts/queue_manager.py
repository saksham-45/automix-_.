#!/usr/bin/env python3
"""
Queue management system for continuous mixing.
Simple JSON-based queue storage for YouTube URLs and local files.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class QueueManager:
    """Manages a queue of songs for continuous mixing."""
    
    def __init__(self, queue_path: str = "data/queue.json"):
        self.queue_path = Path(queue_path)
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_queue_file()
    
    def _ensure_queue_file(self):
        """Ensure queue file exists with proper structure."""
        if not self.queue_path.exists():
            self._write_queue({"songs": []})
    
    def _read_queue(self) -> Dict:
        """Read queue from JSON file."""
        try:
            with open(self.queue_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Reset to empty queue if corrupted
            return {"songs": []}
    
    def _write_queue(self, queue: Dict):
        """Write queue to JSON file."""
        with open(self.queue_path, 'w') as f:
            json.dump(queue, f, indent=2)
    
    def add_song(self, url: str, source: str = "youtube", 
                 metadata: Optional[Dict] = None) -> int:
        """
        Add a song to the queue.
        
        Args:
            url: YouTube URL or local file path
            source: "youtube" or "local"
            metadata: Optional metadata dict (title, artist, etc.)
            
        Returns:
            Queue ID of the added song
        """
        queue = self._read_queue()
        
        # Find next ID
        if queue["songs"]:
            next_id = max(song["id"] for song in queue["songs"]) + 1
        else:
            next_id = 0
        
        song_entry = {
            "id": next_id,
            "url": url,
            "source": source,
            "status": "queued",
            "added_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        queue["songs"].append(song_entry)
        self._write_queue(queue)
        
        return next_id
    
    def list_queue(self) -> List[Dict]:
        """List all songs in the queue."""
        queue = self._read_queue()
        return queue["songs"]
    
    def remove_song(self, song_id: int) -> bool:
        """
        Remove a song from the queue by ID.
        
        Args:
            song_id: Queue ID of the song to remove
            
        Returns:
            True if removed, False if not found
        """
        queue = self._read_queue()
        original_len = len(queue["songs"])
        queue["songs"] = [s for s in queue["songs"] if s["id"] != song_id]
        
        if len(queue["songs"]) < original_len:
            self._write_queue(queue)
            return True
        return False
    
    def clear_queue(self):
        """Clear all songs from the queue."""
        self._write_queue({"songs": []})
    
    def update_song_status(self, song_id: int, status: str, 
                       metadata: Optional[Dict] = None):
        """
        Update status of a song in the queue.
        
        Args:
            song_id: Queue ID of the song
            status: New status (e.g., "downloading", "analyzing", "mixed")
            metadata: Optional metadata to update
        """
        queue = self._read_queue()
        for song in queue["songs"]:
            if song["id"] == song_id:
                song["status"] = status
                if metadata:
                    song["metadata"].update(metadata)
                self._write_queue(queue)
                return
    
    def get_song(self, song_id: int) -> Optional[Dict]:
        """Get a specific song by ID."""
        queue = self._read_queue()
        for song in queue["songs"]:
            if song["id"] == song_id:
                return song
        return None
