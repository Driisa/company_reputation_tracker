import logging
import os
import sys
from datetime import datetime
from functools import wraps

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configure the main logger
log_file = os.path.join(logs_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log')

# Create a custom formatter that includes timestamp, level, module information, and filename
class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Add the timestamp in a readable format
        record.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract filename from pathname if available
        if hasattr(record, 'pathname') and record.pathname:
            # Get the absolute path of the project root directory
            project_root = os.path.dirname(os.path.abspath(__file__))
            
            # Calculate relative path if the pathname starts with project root
            if record.pathname.startswith(project_root):
                record.rel_pathname = os.path.relpath(record.pathname, project_root)
            else:
                record.rel_pathname = record.pathname
                
            record.filename = os.path.basename(record.pathname)
        else:
            record.filename = 'unknown'
            record.rel_pathname = 'unknown'
            
        return super().format(record)

# Configure the logger
logger = logging.getLogger('company_tracker')
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = CustomFormatter('%(timestamp)s [%(levelname)s] [%(rel_pathname)s] %(module)s.%(funcName)s: %(message)s')

# Add formatter to handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Decorator for logging function calls
def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        module_name = func.__module__
        
        # Get the file name and path from the module
        try:
            import inspect
            file_path = inspect.getfile(func)
            file_name = os.path.basename(file_path)
            
            # Get relative path from project root
            project_root = os.path.dirname(os.path.abspath(__file__))
            if file_path.startswith(project_root):
                rel_path = os.path.relpath(file_path, project_root)
            else:
                rel_path = file_path
        except:
            file_name = 'unknown'
            rel_path = 'unknown'
        
        # Log function entry
        logger.info(f"Starting {func_name} in {rel_path}")
        
        try:
            # Call the original function
            result = func(*args, **kwargs)
            
            # Log function exit
            logger.info(f"Completed {func_name} in {rel_path} successfully")
            
            return result
        except Exception as e:
            # Log the exception
            logger.error(f"Error in {func_name} ({rel_path}): {str(e)}")
            raise
    
    return wrapper

# Function to get the logger
def get_logger():
    return logger

# Log application startup
def log_startup(app_name=None):
    if app_name:
        logger.info(f"=== {app_name} Started ===")
    else:
        logger.info("=== Application Started ===")
    logger.info(f"Log file: {log_file}")

# Log application shutdown
def log_shutdown(app_name=None):
    if app_name:
        logger.info(f"=== {app_name} Shutdown ===")
    else:
        logger.info("=== Application Shutdown ===")

# Log a general message
def log_info(message):
    logger.info(message)

# Log a warning message
def log_warning(message):
    logger.warning(message)

# Log an error message
def log_error(message, exc_info=None):
    if exc_info:
        logger.error(message, exc_info=exc_info)
    else:
        logger.error(message)

# Log a debug message
def log_debug(message):
    logger.debug(message)

# For testing
if __name__ == "__main__":
    log_startup("Logger Test")
    log_info("This is a test info message")
    log_warning("This is a test warning message")
    log_error("This is a test error message")
    log_shutdown("Logger Test")