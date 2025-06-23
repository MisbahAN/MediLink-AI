#!/usr/bin/env python3
"""
OCR Module for Referral Package Processing
Extracts text from PDFs using Mistral API with caching
"""

import base64
import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from ..ai.mistral import Mistral
from ..lib.context import Context
from ..lib.data import Data
from ..lib.step import Step
from ..lib.util import is_base64, parse_json, read_pdf_as_base64, write_file

logger = logging.getLogger(__name__)


class OCR(Step):
    def __init__(self):
        super().__init__("OCR", "Extract text from PDF files using Mistral API with caching")

    def run(self, context: Context):
        """Extract text from PDF using OCR with caching"""
        logger.info(f"Starting OCR processing for {len(context.data)} data entries")
        
        for i, data in enumerate(context.data, 1):
            logger.info(f"Processing OCR for data entry {i}/{len(context.data)}: {data.name}")
            
            try:
                self._process_prior_authorization(data)
                self._process_referral_package(data)
                
            except Exception as e:
                logger.error(f"  Error during OCR processing for {data.name}: {str(e)}")
                raise
        
        logger.info("OCR processing completed for all data entries")

    def _clean_text(self, text: str) -> str:
        """Apply comprehensive text sanitization and markdown formatting"""
        if not text:
            return ""

        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Fix LaTeX/math notation artifacts
        text = re.sub(r'\$\s*\\?square\s*\$', '☐', text)  # Empty checkbox
        text = re.sub(r'\$\s*\\?qquad\s*\$', '_' * 20, text)  # Long blank line
        text = re.sub(r'\$\s*\\?quad\s*\$', '_' * 10, text)  # Medium blank line
        text = re.sub(r'\$\s*\\?geq\s*\$', '≥', text)  # Greater than or equal
        text = re.sub(r'\$\s*\\?leq\s*\$', '≤', text)  # Less than or equal
        text = re.sub(r'\$\s*\\?circ\s*\$', '®', text)  # Registered trademark
        text = re.sub(r'\$\s*\\?circledR\s*\$', '®', text)  # Registered trademark
        text = re.sub(r'\$\s*\\?text\s*\{\s*TM\s*\}\s*\$', '™', text)  # Trademark
        text = re.sub(r'\$\s*\{\s*\}\s*\^\s*\{\s*\\?circ\s*\}\s*\$', '®', text)  # Complex trademark
        
        # Clean up spacing around symbols
        text = re.sub(r'\s+([®™©])', r'\1', text)
        
        # Fix broken words (common OCR issue)
        text = re.sub(r'Antimigra\s+ine', 'Antimigraine', text)
        text = re.sub(r'Authoriza\s+tion', 'Authorization', text)
        text = re.sub(r'Eptinezuma\s+b-jmmr', 'Eptinezumab-jmmr', text)
        
        # Add proper line breaks for form sections
        text = re.sub(r'(Member Information)', r'\n## \1\n', text)
        text = re.sub(r'(Prescriber information)', r'\n## \1\n', text)
        text = re.sub(r'(Drug information)', r'\n## \1\n', text)
        
        # Format field labels properly
        text = re.sub(r'(Last name|First name|Medicaid ID number|Date of birth|Weight in kilograms|NPI number|Phone number|Fax number|Drug name|Drug form|Drug strength|Dosing frequency|Length of therapy|Quantity):\s*', r'\n**\1:** ', text)
        
        # Format questions with proper numbering
        text = re.sub(r'(\d+\.)\s*([A-Z][^?]+\?)', r'\n\1 \2', text)
        
        # Format checkboxes properly
        text = re.sub(r'☐\s*(Yes|No)', r'☐ \1', text)
        
        # Clean up table formatting
        text = re.sub(r'\s*---\s*---\s*', '\n\n| | |\n|---|---|\n', text)
        
        # Add spacing around headers
        text = re.sub(r'(#{1,3}\s+[^\n]+)', r'\n\1\n', text)
        
        # Clean up multiple spaces and normalize line breaks
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove repeated punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Clean up common OCR artifacts
        text = re.sub(r'\s*[|\\/_]\s*', ' ', text)
        
        return text.strip()
    
    def _ocr(self, data: str, document_type: str = "PDF") -> Dict[str, Any] | List[Dict[str, Any]]:
        """
        @param data: Base64 encoded PDF data
        @param document_type: Type of document being processed
        @return ocr_result: OCR result
        """
        try:
            if (not is_base64(data)):
                data = base64.b64encode(data).decode('utf-8')
            
            mistral = Mistral()
            
            response = mistral.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{data}"
                },
                include_image_base64=True
            )
            
            result = parse_json(response.model_dump_json())
            return result
            
        except Exception as e:
            logger.error(f"    Error processing OCR for {document_type}: {str(e)}")
            return {"pages": []}

    def _process_referral_package(self, data: Data):
        """Process referral package"""
        try:
            input_path = Path(data.input_dir).joinpath("referral_package.pdf")
            output_path = Path(data.output_dir).joinpath("referral_package.md")
            
            if output_path.exists():
                return
            
            if not input_path.exists():
                logger.warning(f"Referral package PDF not found: {input_path}")
                return
            
            base64_data = read_pdf_as_base64(input_path)
            response = self._ocr(base64_data, "Referral Package")
            
            # Extract markdown from all pages
            if "pages" in response:
                markdown_pages = []
                for page in response["pages"]:
                    if "markdown" in page:
                        page_md = page["markdown"]
                        cleaned_md = self._clean_text(page_md)
                        markdown_pages.append(cleaned_md)
                
                markdown = "\n".join(markdown_pages)
            else:
                markdown = ""
                logger.warning("  No pages found in OCR response")
            
            write_file(output_path, markdown)
            
        except Exception as e:
            logger.error(f"  Error processing referral package for {data.name}: {str(e)}")
            raise

    def _process_prior_authorization(self, data: Data):
        """Process prior authorization"""
        try:
            input_path = Path(data.input_dir).joinpath("prior_authorization.pdf")
            output_path = Path(data.output_dir).joinpath("prior_authorization.md")
            
            if output_path.exists():
                return
            
            if not input_path.exists():
                logger.warning(f"  Prior authorization PDF not found: {input_path}")
                return
            
            base64_data = read_pdf_as_base64(input_path)
            response = self._ocr(base64_data, "Prior Authorization")
            
            # Extract markdown from all pages
            if "pages" in response:
                markdown_pages = []
                for page in response["pages"]:
                    if "markdown" in page:
                        page_md = page["markdown"]
                        cleaned_md = self._clean_text(page_md)
                        markdown_pages.append(cleaned_md)
                
                markdown = "\n".join(markdown_pages)
            else:
                markdown = ""
                logger.warning("  No pages found in OCR response")
            
            write_file(output_path, markdown)
            
        except Exception as e:
            logger.error(f"  Error processing prior authorization for {data.name}: {str(e)}")
            raise
