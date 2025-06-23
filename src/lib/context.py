from typing import List
from .data import Data

import logging
import sys

class Context:
    def __init__(self, data: List[Data]):
        self.data: List[Data] = data
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with proper configuration"""
        logger = logging.getLogger("pipeline")
        
        # Avoid duplicate handlers if logger already configured
        if logger.handlers:
            return logger
            
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger