import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format='%(asctime)s - %(levelname)s - %(message)s',  # Format for the log messages
        handlers=[
            logging.StreamHandler()  # Log to the console
        ]
    )