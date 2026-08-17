import logging
import sys
from pathlib import Path
from src.utils.config import load_config

def setup_logging(name="two-tower"):
    """Configure logging to file and stdout."""
    config = load_config()
    log_level = getattr(logging, config["log_level"].upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setLevel(log_level)

    # Stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
