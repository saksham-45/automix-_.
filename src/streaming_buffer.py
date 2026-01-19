#!/usr/bin/env python3
"""
Triple buffer system for streaming continuous mixes.
Manages: current (playing), transition (ready), next (preparing)

This enables seamless transitions by maintaining three buffers:
1. Current: What's playing now
2. Transition: Mixed transition ready to play
3. Next: Next segment being prepared (downloading/analyzing/mixing)

As audio plays, buffers rotate: next → transition → current
"""
from typing import Optional, Dict, Tuple
import numpy as np
from concurrent.futures import Future, ThreadPoolExecutor


class StreamingBuffer:
    """
    Triple buffer for streaming continuous mixing.
    
    Architecture:
    - Buffer 1 (current): Currently playing audio
    - Buffer 2 (transition): Mixed transition ready to play
    - Buffer 3 (next): Next segment being prepared
    
    As audio plays, buffers rotate: next → transition → current
    """
    
    def __init__(self, buffer_size_seconds: int = 60, sr: int = 44100):
        """
        Initialize streaming buffer.
        
        Args:
            buffer_size_seconds: Target buffer size (for future use)
            sr: Sample rate
        """
        self.buffer_size_samples = int(buffer_size_seconds * sr)
        self.sr = sr
        
        self.buffers = {
            'current': None,      # Currently playing
            'transition': None,   # Mixed transition ready
            'next': None          # Next segment preparing
        }
        
        self.buffer_metadata = {
            'current': None,
            'transition': None,
            'next': None
        }
        
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.futures = {}
        
    def set_current(self, audio: np.ndarray, metadata: Dict = None):
        """Set current buffer (what's playing now)."""
        self.buffers['current'] = audio
        self.buffer_metadata['current'] = metadata or {}
    
    def set_transition(self, audio: np.ndarray, metadata: Dict = None):
        """Set transition buffer (ready to play)."""
        self.buffers['transition'] = audio
        self.buffer_metadata['transition'] = metadata or {}
    
    def set_next_future(self, future: Future):
        """Set future for next buffer preparation."""
        self.futures['next'] = future
    
    def advance(self):
        """
        Rotate buffers: next → transition → current.
        
        This is called when current buffer finishes playing.
        """
        # Move transition to current
        self.buffers['current'] = self.buffers['transition']
        self.buffer_metadata['current'] = self.buffer_metadata['transition']
        
        # Move next to transition (if ready)
        if 'next' in self.futures:
            future = self.futures.pop('next')
            if future.done():
                try:
                    next_audio, next_meta = future.result()
                    self.buffers['transition'] = next_audio
                    self.buffer_metadata['transition'] = next_meta
                except Exception as e:
                    print(f"  ⚠ Next buffer failed: {e}")
                    self.buffers['transition'] = None
            else:
                # Not ready yet, transition stays None
                self.buffers['transition'] = None
        else:
            # No next buffer prepared
            self.buffers['transition'] = None
        
        # Next buffer will be prepared separately
    
    def get_current(self) -> Optional[np.ndarray]:
        """Get current playing buffer."""
        return self.buffers['current']
    
    def get_transition(self) -> Optional[np.ndarray]:
        """Get transition buffer."""
        return self.buffers['transition']
    
    def is_transition_ready(self) -> bool:
        """Check if transition buffer is ready."""
        return self.buffers['transition'] is not None
    
    def is_next_ready(self) -> bool:
        """Check if next buffer is ready."""
        if 'next' not in self.futures:
            return False
        return self.futures['next'].done()
    
    def clear(self):
        """Clear all buffers."""
        self.buffers = {
            'current': None,
            'transition': None,
            'next': None
        }
        self.buffer_metadata = {
            'current': None,
            'transition': None,
            'next': None
        }
        self.futures = {}
    
    def shutdown(self):
        """Shutdown executor."""
        self.executor.shutdown(wait=True)
