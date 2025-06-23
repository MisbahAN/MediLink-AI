from mistralai import Mistral as MistralClient
import os
from dotenv import load_dotenv
import base64
from ..lib.util import is_base64, parse_json
from typing import Any, Dict, List


class Mistral:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found")
        self.client = MistralClient(api_key=self.api_key)

   