# Lazy imports to avoid hanging during package import
def run(*args, **kwargs):
    """Lazy import and call the run function."""
    from .run import run as _run
    return _run(*args, **kwargs)

def ensure_registrations():
    """Ensure reward and verifier registrations are loaded when needed."""
    import retrain.reward
    import retrain.verifier

__all__ = ['run', 'ensure_registrations']

# Optionally expose other core components if needed for advanced use
# from .trainer.base import BaseTrainer
# from .model.loader import load_model
# ...

# Version information
__version__ = "0.1.0" # Example version
