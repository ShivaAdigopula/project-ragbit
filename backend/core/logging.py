import logging
import sys
import time
import os
from logging.handlers import RotatingFileHandler
from typing import Any
from backend.core.config import settings

# Configure logging format
class StructuredFormatter(logging.Formatter):
    """Custom formatter to output structured logs or clean console logs."""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Include correlation ID or execution context if available
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = getattr(record, "correlation_id")
            
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        # For local development console, output readable text. E.g. in JSON or structured pattern.
        # Here we construct a clean structured-like output.
        correlation_str = f" [CorrelationID: {log_data['correlation_id']}]" if "correlation_id" in log_data else ""
        exc_str = f"\n{log_data['exception']}" if "exception" in log_data else ""
        return f"{log_data['timestamp']} | {log_data['level']:<7} | {log_data['logger']} | {log_data['message']}{correlation_str}{exc_str}"

def setup_logging() -> None:
    """Sets up global application logging hierarchy."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    root_logger.addHandler(console_handler)
    
    # File Handler (logs/app.log)
    try:
        os.makedirs("logs", exist_ok=True)
        file_handler = RotatingFileHandler(
            "logs/app.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(StructuredFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
        root_logger.addHandler(file_handler)
    except Exception as e:
        # Fallback if logs dir cannot be written
        sys.stderr.write(f"Warning: Failed to initialize file logging handler: {str(e)}\n")
    
    # Adjust specific noisy dependencies
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("motor").setLevel(logging.WARNING)

# Initialize logging setup
setup_logging()
logger = logging.getLogger("app")
