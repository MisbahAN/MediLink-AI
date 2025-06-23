import logging
import os

from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient


logger = logging.getLogger(__name__)


class OpenAI:
    def __init__(self):
        try:
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                logger.error("OPENAI_API_KEY environment variable not found")
                raise ValueError("OPENAI_API_KEY environment variable is required")
            
            self.client = OpenAIClient(api_key=api_key)
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
    
    def log_api_call(self, model: str, prompt_length: int, response_length: int = None, tokens_used: int = None):
        """Log API call statistics"""
        log_msg = f"OpenAI API call - Model: {model}, Prompt length: {prompt_length} chars"
        
        if response_length is not None:
            log_msg += f", Response length: {response_length} chars"
        
        if tokens_used is not None:
            log_msg += f", Tokens used: {tokens_used}"
            
        logger.info(log_msg)