"""
DJ Transition Analysis System
Extracts comprehensive audio features for AI training

Advanced Superhuman Modules:
- MicroTimingEngine: Sub-millisecond groove and transient matching
- SpectralIntelligenceEngine: Surgical frequency management
- HybridTechniqueBlender: Creative technique combinations
- StemOrchestrator: Musical stem conversations
- MonteCarloQualityOptimizer: Simulation-based quality optimization
- SuperhumanDJEngine: Unified engine coordinating all modules
"""

__version__ = "0.2.0"

# Core exports
from src.smart_mixer import SmartMixer

# Superhuman engine exports
try:
    from src.superhuman_engine import SuperhumanDJEngine
    from src.micro_timing_engine import MicroTimingEngine
    from src.spectral_intelligence import SpectralIntelligenceEngine
    from src.hybrid_technique_blender import HybridTechniqueBlender
    from src.stem_orchestrator import StemOrchestrator
    from src.montecarlo_optimizer import MonteCarloQualityOptimizer
except ImportError:
    pass  # Modules may not be available in all environments
