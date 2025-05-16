# Expose the main entry point for initiating training runs
from .run import run

# Import submodules to ensure their registrations (e.g., tools, rewards) take effect
import retrain.reward # Already implicitly imported by other modules, but good to be explicit if needed
import retrain.verifier # Ensure verifiers are registered

__all__ = ['run']

# Optionally expose other core components if needed for advanced use
# from .trainer.base import BaseTrainer
# from .model.loader import load_model
# ...

# Version information
__version__ = "0.1.0" # Example version
