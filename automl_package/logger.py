import logging

# Configure a logger for the automl_package
logger = logging.getLogger('automl_package')
logger.setLevel(logging.INFO) # Default level to INFO

# Add a console handler if not already added
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)