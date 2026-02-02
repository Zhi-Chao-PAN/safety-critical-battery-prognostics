import logging
import sys
from pathlib import Path

def setup_logger(name: str = "BatteryPrognostics", log_file: str = "project.log", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a structured logger.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file
        level: Logging level (default: logging.INFO)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # improved formatting with timestamp and module name
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
