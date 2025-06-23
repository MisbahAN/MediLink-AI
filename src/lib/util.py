import base64
import binascii
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def is_base64(s: str) -> bool:
    """
    Check if a string is a valid base64 encoded string
    @param s: The string to check
    @return: True if the string is a valid base64 encoded string, False otherwise
    """
    try:
        base64.b64decode(s, validate=True)
        return True
    except binascii.Error:
        return False


def read_file(path: str) -> str:
    """Read content from file
    @param path: Path to the file
    @return: Content
    """
    try:
        with open(path, "r", encoding='utf-8') as file:
            content = file.read()
        return content
        
    except Exception as e:
        logger.error(f"Error reading file {path}: {str(e)}")
        raise


def write_file(path: str, content: str):
    """Write content to file
    @param path: Path to the file
    @param content: Content to write
    """
    try:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding='utf-8') as file:
            file.write(content)
            
    except Exception as e:
        logger.error(f"Error writing file {path}: {str(e)}")
        raise


def write_json(path: str, content: dict):
    """Write JSON content to file
    @param path: Path to the JSON file
    @param content: JSON content
    """
    try:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding='utf-8') as file:
            json.dump(content, file, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Error writing JSON file {path}: {str(e)}")
        raise


def read_json(path: str) -> dict:
    """Read JSON file
    @param path: Path to the JSON file
    @return: JSON content
    """
    try:
        with open(path, "r", encoding='utf-8') as file:
            content = json.load(file)
        return content
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error reading JSON file {path}: {str(e)}")
        raise


def parse_json(content: str) -> dict:
    """Parse JSON content
    @param content: JSON content
    @return: Parsed JSON
    """
    try:
        return json.loads(content)
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON content: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error parsing JSON content: {str(e)}")
        raise


def read_pdf_as_base64(path: str) -> str:
    """Read PDF file and return base64 encoded data
    @param path: Path to the PDF file
    @return base64_data: Base64 encoded PDF data
    """
    try:
        with open(path, "rb") as f:
            pdf_data = f.read()
            base64_data = base64.b64encode(pdf_data).decode('utf-8')
        return base64_data
        
    except Exception as e:
        logger.error(f"Error reading PDF as base64 {path}: {str(e)}")
        raise
