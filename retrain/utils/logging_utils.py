from loguru import logger
import sys

DEFAULT_LOG_LEVEL = "INFO"

def setup_logging(level: str = DEFAULT_LOG_LEVEL, sink = sys.stderr):
    """
    Configures the Loguru logger with a single handler.

    Removes any existing handlers and adds a new one with the specified
    level and a standard format.

    Args:
        level: The minimum logging level (e.g., "DEBUG", "INFO", "WARNING").
        sink: The sink for the logs (e.g., sys.stderr, file path).
    """
    logger.remove() # Remove all existing handlers to ensure a clean setup
    logger.add(
        sink,
        level=level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True # Enable colorization for stderr if it's a TTY
    )
    logger.info(f"Loguru logger (re)configured. Level: {level.upper()}, Sink: {sink}")

def get_logger(name: str):
    """
    Returns a logger instance, potentially bound with the given name.
    Relies on `setup_logging` having been called previously to configure the global logger.

    Args:
        name: The name to associate with the logger (e.g., module name).
              This can be used by Loguru's formatter or for specific bindings.

    Returns:
        The Loguru logger instance, possibly bound with the name.
    """
    # The `name` can be used in the format string if logger.bind(name=name) is used, 
    # or if the format string includes {name}.
    # For basic usage, returning the global logger is fine. 
    # If different modules need truly separate logger objects or configs beyond binding,
    # Loguru's approach is generally to use the single `logger` object and filter/format at the sink level.
    # Here, we ensure the name is available for formatters like the one in setup_logging.
    return logger.bind(name=name) # Binding allows {name} in format to work correctly.


__all__ = [
    "setup_logging",
    "get_logger"
] 