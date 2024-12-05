import logging
import os
from datetime import datetime

# Configure logging
def setup_logging(name: str, **kwargs):
    level_name = logging.DEBUG if kwargs.get('debug', False) else logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(level_name)
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # File handler with timestamp in filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Use custom file name pattern if provided, otherwise use default
    file_pattern = kwargs.get('file_pattern', 'test_case_{test_case_name}_{timestamp}.log' if type == 'test-case' else 'test_session_{timestamp}.log')
    file_name = f"logs/{file_pattern.format(timestamp=timestamp, test_case_name=kwargs.get('test_case_name', ''))}"
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(level_name)
    
    # Stream handler for console output
    if kwargs.get('console_output', True):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level_name)
    
    # Create formatter with custom format if provided
    format_string = kwargs.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    date_format = kwargs.get('date_format', '%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter(format_string, date_format)
    
    file_handler.setFormatter(formatter)
    if kwargs.get('console_output', True):
        stream_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    if kwargs.get('console_output', True):
        logger.addHandler(stream_handler)
    
    # Add any additional handlers from kwargs
    extra_handlers = kwargs.get('handlers', [])
    for handler in extra_handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger