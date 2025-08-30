"""
Logging utilities for A2A agents.
Health Universe compatible structured logging.
"""

import os
import sys
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for A2A agent logs.
    Provides consistent logging format for Health Universe deployment.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        # Base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'message', 'exc_info', 
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        # Add Health Universe context if available
        hu_app_url = os.getenv("HU_APP_URL")
        if hu_app_url:
            log_entry["deployment"] = {
                "platform": "health_universe",
                "app_url": hu_app_url
            }
        
        return json.dumps(log_entry)


class PlainFormatter(logging.Formatter):
    """Plain text formatter for local development."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


def get_logger(
    name: str,
    level: Optional[str] = None,
    structured: Optional[bool] = None
) -> logging.Logger:
    """
    Get configured logger for A2A agents.
    
    Args:
        name: Logger name (typically class name)
        level: Log level override
        structured: Force structured/plain logging
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Determine log level
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    try:
        log_level_int = getattr(logging, log_level)
    except AttributeError:
        log_level_int = logging.INFO
    
    logger.setLevel(log_level_int)
    
    # Determine formatter type
    if structured is None:
        # Auto-detect: use structured for Health Universe, plain for local
        structured = "healthuniverse.com" in os.getenv("HU_APP_URL", "")
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Set formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = PlainFormatter()
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def log_agent_start(
    logger: logging.Logger,
    agent_name: str,
    version: str,
    port: Optional[int] = None
) -> None:
    """
    Log agent startup information.
    
    Args:
        logger: Logger instance
        agent_name: Name of the agent
        version: Agent version
        port: Port number if applicable
    """
    startup_info = {
        "event": "agent_start",
        "agent_name": agent_name,
        "version": version,
        "platform": "health_universe" if os.getenv("HU_APP_URL") else "local"
    }
    
    if port:
        startup_info["port"] = port
    
    # Add environment info
    startup_info["environment"] = {
        "python_version": sys.version.split()[0],
        "has_google_api": bool(os.getenv("GOOGLE_API_KEY")),
        "has_openai_api": bool(os.getenv("OPENAI_API_KEY")),
        "has_anthropic_api": bool(os.getenv("ANTHROPIC_API_KEY"))
    }
    
    logger.info("Agent starting", extra=startup_info)


def log_agent_call(
    logger: logging.Logger,
    source_agent: str,
    target_agent: str,
    message_type: str,
    success: bool,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None
) -> None:
    """
    Log inter-agent communication.
    
    Args:
        logger: Logger instance
        source_agent: Calling agent name
        target_agent: Target agent name
        message_type: Type of message/request
        success: Whether call succeeded
        duration_ms: Request duration in milliseconds
        error: Error message if failed
    """
    call_info = {
        "event": "agent_call",
        "source_agent": source_agent,
        "target_agent": target_agent,
        "message_type": message_type,
        "success": success
    }
    
    if duration_ms is not None:
        call_info["duration_ms"] = duration_ms
    
    if error:
        call_info["error"] = error
    
    level = logging.INFO if success else logging.ERROR
    message = f"Agent call: {source_agent} -> {target_agent}"
    
    logger.log(level, message, extra=call_info)


def log_processing_start(
    logger: logging.Logger,
    agent_name: str,
    task_id: str,
    message_length: int
) -> None:
    """
    Log start of message processing.
    
    Args:
        logger: Logger instance
        agent_name: Agent name
        task_id: Task identifier
        message_length: Length of input message
    """
    process_info = {
        "event": "processing_start",
        "agent_name": agent_name,
        "task_id": task_id,
        "message_length": message_length
    }
    
    logger.info("Starting message processing", extra=process_info)


def log_processing_complete(
    logger: logging.Logger,
    agent_name: str,
    task_id: str,
    duration_ms: float,
    response_type: str,
    success: bool = True
) -> None:
    """
    Log completion of message processing.
    
    Args:
        logger: Logger instance
        agent_name: Agent name
        task_id: Task identifier
        duration_ms: Processing duration in milliseconds
        response_type: Type of response (text, data, etc.)
        success: Whether processing succeeded
    """
    process_info = {
        "event": "processing_complete",
        "agent_name": agent_name,
        "task_id": task_id,
        "duration_ms": duration_ms,
        "response_type": response_type,
        "success": success
    }
    
    level = logging.INFO if success else logging.ERROR
    message = f"Message processing {'completed' if success else 'failed'}"
    
    logger.log(level, message, extra=process_info)


def log_health_check(
    logger: logging.Logger,
    agent_name: str,
    status: str,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log health check results.
    
    Args:
        logger: Logger instance
        agent_name: Agent name
        status: Health status (healthy, degraded, unhealthy)
        details: Additional health details
    """
    health_info = {
        "event": "health_check",
        "agent_name": agent_name,
        "status": status
    }
    
    if details:
        health_info["details"] = details
    
    level = logging.INFO if status == "healthy" else logging.WARNING
    
    logger.log(level, f"Health check: {status}", extra=health_info)


def setup_root_logger(structured: Optional[bool] = None) -> None:
    """
    Setup root logger configuration.
    
    Args:
        structured: Force structured/plain logging
    """
    # Get root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure root logger
    get_logger("root", structured=structured)


def create_request_logger(request_id: str) -> logging.LoggerAdapter:
    """
    Create a logger adapter with request ID context.
    
    Args:
        request_id: Unique request identifier
        
    Returns:
        Logger adapter with request context
    """
    logger = get_logger("request")
    
    class RequestAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            return f"[{request_id}] {msg}", kwargs
    
    return RequestAdapter(logger, {"request_id": request_id})


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        """
        Initialize performance timer.
        
        Args:
            logger: Logger instance
            operation: Operation name
            **context: Additional context for logging
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log results."""
        if self.start_time is not None:
            import time
            duration_ms = (time.time() - self.start_time) * 1000
            
            timing_info = {
                "event": "performance_timing",
                "operation": self.operation,
                "duration_ms": duration_ms,
                "success": exc_type is None,
                **self.context
            }
            
            if exc_type:
                timing_info["error"] = str(exc_val)
            
            level = logging.INFO if exc_type is None else logging.ERROR
            message = f"Operation '{self.operation}' took {duration_ms:.2f}ms"
            
            self.logger.log(level, message, extra=timing_info)


def setup_logging(level: str = "INFO", structured: bool = False) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        structured: Whether to use structured JSON logging
    """
    # Setup root logger
    setup_root_logger(structured=structured)
    
    # Set log level
    root_logger = logging.getLogger()
    try:
        log_level = getattr(logging, level.upper())
        root_logger.setLevel(log_level)
    except AttributeError:
        root_logger.setLevel(logging.INFO)


def reset_logging() -> None:
    """Reset logging configuration by clearing all handlers."""
    root_logger = logging.getLogger()
    
    # Clear all handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Reset level
    root_logger.setLevel(logging.WARNING)