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

# Best-effort eager exports. Kept optional so lightweight entry points (the
# stem-free club mixer + streaming server) can import this package on a torch-free
# image without dragging in demucs/torch. Anything that actually needs these
# symbols imports them explicitly and will surface the real ImportError at use.
try:
    from src.smart_mixer import SmartMixer  # noqa: F401
except Exception:
    pass

# Superhuman engine exports
try:
    from src.superhuman_engine import SuperhumanDJEngine  # noqa: F401
    from src.micro_timing_engine import MicroTimingEngine  # noqa: F401
    from src.spectral_intelligence import SpectralIntelligenceEngine  # noqa: F401
    from src.hybrid_technique_blender import HybridTechniqueBlender  # noqa: F401
    from src.stem_orchestrator import StemOrchestrator  # noqa: F401
    from src.montecarlo_optimizer import MonteCarloQualityOptimizer  # noqa: F401
except Exception:
    pass  # Modules may not be available in all environments
