# backend/app/services/mistral_service.py
"""
Mistral OCR service for PDF extraction and document processing.

This service provides AI-powered OCR capabilities using Mistral's dedicated OCR API,
optimized for medical document processing with high accuracy and cost efficiency.
Primary service for document extraction with Gemini as fallback.
"""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import json
import time
import re

try:
    from mistralai import Mistral
except ImportError:
    Mistral = None

try:
    import spacy
    # Try to load medical NLP model, fallback to general model
    try:
        nlp = spacy.load("en_core_sci_md")  # Medical model if available
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm")  # General model
        except OSError:
            nlp = None
except ImportError:
    spacy = None
    nlp = None

from app.models.schemas import ExtractedField, ConfidenceLevel, PatientInfo, ClinicalData
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MistralService:
    """
    Service for AI-powered OCR using Mistral's dedicated OCR API.
    
    Provides high-accuracy document extraction optimized for medical forms,
    with cost-effective processing at $0.001 per page and structured output.
    """
    
    def __init__(self):
        """Initialize Mistral OCR service with API configuration."""
        self.client = None
        self.model_name = settings.MISTRAL_OCR_MODEL
        self.api_key = settings.MISTRAL_API_KEY
        self.timeout_seconds = settings.PROCESSING_TIMEOUT_SECONDS
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay_base = 1.0  # Base delay in seconds
        self.retry_backoff_factor = 2.0
        
        # Confidence thresholds for Mistral OCR
        self.confidence_thresholds = {
            "high": 0.90,       # Excellent OCR quality
            "medium": 0.75,     # Good OCR quality with minor issues
            "low": 0.60,        # Acceptable but may need verification
            "very_low": 0.40    # Poor quality, recommend fallback
        }
        
        # Cost tracking and configuration
        self.use_batch_processing = settings.MISTRAL_BATCH_PROCESSING
        self.confidence_threshold = settings.MISTRAL_CONFIDENCE_THRESHOLD
        self.max_pages_per_request = settings.MISTRAL_MAX_PAGES_PER_REQUEST
        self.cost_per_page = 0.001  # $0.001 per page regular
        self.batch_cost_per_page = 0.0005  # $0.0005 per page with batch
        
        logger.info("Mistral OCR service initialized as PRIMARY extraction method")
    
    def initialize_client(self) -> bool:
        """
        Initialize the Mistral API client.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not Mistral:
            logger.error("mistralai package not installed")
            return False
        
        if not self.api_key:
            logger.error("MISTRAL_API_KEY not configured")
            return False
        
        try:
            self.client = Mistral(api_key=self.api_key)
            
            # Test connection with a simple request
            # Note: Mistral OCR doesn't have a simple test endpoint
            # We'll validate by checking if client is properly configured
            if self.client:
                logger.info(f"Mistral OCR client initialized successfully")
                return True
            else:
                logger.error("Failed to initialize Mistral OCR client")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Mistral client: {e}")
            return False
    
    async def extract_from_pdf(
        self, 
        pdf_path: Union[str, Path], 
        extraction_type: str = "medical_referral",
        include_images: bool = True
    ) -> Dict[str, Any]:
        """
        Extract structured data from PDF using Mistral OCR API.
        
        Args:
            pdf_path: Path to the PDF file
            extraction_type: Type of extraction ("medical_referral", "pa_form", "general")
            include_images: Whether to include image base64 in response
            
        Returns:
            Dictionary containing extracted data with high confidence scores
        """
        try:
            # Initialize client if not already done
            if not self.client and not self.initialize_client():
                raise RuntimeError("Failed to initialize Mistral client")
            
            logger.info(f"Starting Mistral OCR extraction for {Path(pdf_path).name}")
            
            # Process PDF with retry logic
            raw_result = await self._process_pdf_with_retry(
                str(pdf_path), extraction_type, include_images
            )
            
            # Structure the results into standardized format
            structured_result = self._structure_ocr_results(raw_result, extraction_type)
            
            logger.info(f"Mistral OCR extraction completed with {structured_result['overall_confidence']:.2%} confidence")
            return structured_result
            
        except Exception as e:
            logger.error(f"Mistral OCR extraction failed: {e}")
            raise
    
    async def _process_pdf_with_retry(
        self, 
        pdf_path: str, 
        extraction_type: str,
        include_images: bool
    ) -> Dict[str, Any]:
        """
        Process PDF with Mistral OCR API with retry logic.
        
        Args:
            pdf_path: Path to PDF file
            extraction_type: Type of extraction to perform
            include_images: Whether to include images
            
        Returns:
            Raw OCR results from Mistral API
        """
        for attempt in range(self.max_retries):
            try:
                result = await self._call_mistral_ocr_api(
                    pdf_path, extraction_type, include_images
                )
                return result
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Mistral OCR processing failed after {self.max_retries} attempts: {e}"
                    )
                    raise
                
                # Calculate retry delay with exponential backoff
                delay = self.retry_delay_base * (self.retry_backoff_factor ** attempt)
                logger.warning(
                    f"Mistral OCR attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.1f} seconds..."
                )
                await asyncio.sleep(delay)
    
    async def _call_mistral_ocr_api(
        self, 
        pdf_path: str, 
        extraction_type: str,
        include_images: bool
    ) -> Dict[str, Any]:
        """
        Make the actual API call to Mistral OCR.
        
        Args:
            pdf_path: Path to PDF file
            extraction_type: Type of extraction
            include_images: Whether to include images
            
        Returns:
            OCR results from Mistral API
        """
        try:
            # Read PDF file and encode to base64
            with open(pdf_path, 'rb') as pdf_file:
                pdf_content = pdf_file.read()
                pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
            
            logger.info(f"Making Mistral OCR request: PDF size {len(pdf_content)} bytes")
            
            # Make OCR request with corrected API format
            start_time = time.time()
            ocr_response = self.client.ocr.process(
                model=self.model_name,
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{pdf_base64}"
                },
                include_image_base64=include_images
            )
            processing_time = time.time() - start_time
            
            logger.info(f"Mistral OCR completed in {processing_time:.2f} seconds")
            
            # Parse response structure
            if not hasattr(ocr_response, 'pages') or not ocr_response.pages:
                raise ValueError("No pages in Mistral OCR response")
            
            # Extract all text from pages
            extracted_text = ""
            page_data = []
            
            for page in ocr_response.pages:
                if hasattr(page, 'markdown') and page.markdown:
                    extracted_text += page.markdown + "\n\n"
                    page_data.append({
                        "index": page.index,
                        "markdown": page.markdown,
                        "dimensions": page.dimensions.__dict__ if hasattr(page, 'dimensions') else {},
                        "images": [img.__dict__ for img in page.images] if hasattr(page, 'images') else []
                    })
            
            if not extracted_text.strip():
                raise ValueError("No text extracted from PDF")
            
            return {
                "extracted_text": extracted_text.strip(),
                "page_data": page_data,
                "raw_response": ocr_response,
                "processing_time": processing_time,
                "pages_processed": len(ocr_response.pages),
                "usage_info": ocr_response.usage_info.__dict__ if hasattr(ocr_response, 'usage_info') else {}
            }
            
        except Exception as e:
            logger.error(f"Mistral OCR API call failed: {e}")
            raise
    
    def _structure_ocr_results(
        self, 
        raw_result: Dict[str, Any], 
        extraction_type: str
    ) -> Dict[str, Any]:
        """
        Structure OCR results into standardized format.
        
        Args:
            raw_result: Raw results from Mistral OCR API
            extraction_type: Type of extraction performed
            
        Returns:
            Structured extraction result
        """
        extracted_text = raw_result.get("extracted_text", "")
        pages_processed = raw_result.get("pages_processed", 1)
        processing_time = raw_result.get("processing_time", 0)
        
        # Parse extracted text based on type
        if extraction_type == "medical_referral":
            parsed_data = self._parse_medical_referral(extracted_text)
        elif extraction_type == "pa_form":
            parsed_data = self._parse_pa_form(extracted_text)
        else:
            parsed_data = self._parse_general_document(extracted_text)
        
        # Calculate confidence based on text quality and structure
        confidence = self._calculate_confidence(extracted_text, parsed_data)
        
        # Calculate cost
        cost_info = self._calculate_cost(pages_processed)
        
        return {
            "patient_info": parsed_data.get("patient_info", {}),
            "clinical_data": parsed_data.get("clinical_data", {}),
            "raw_text": extracted_text,
            "raw_extracted_pages": {f"page_{i+1}": page for i, page in enumerate(raw_result.get("page_data", []))},
            "overall_confidence": confidence,
            "pages_processed": pages_processed,
            "successful_extraction": True,
            "text_length": len(extracted_text),
            "processing_time_seconds": processing_time,
            "parsing_details": parsed_data.get("parsing_details", {}),
            "cost_info": cost_info,
            "usage_info": raw_result.get("usage_info", {})
        }
    
    def _parse_medical_referral(self, text: str) -> Dict[str, Any]:
        """
        Parse medical referral text to extract structured data using NLP.
        
        Args:
            text: Extracted text from OCR
            
        Returns:
            Parsed medical data with confidence scores
        """
        patient_info = {}
        clinical_data = {}
        parsing_details = {"extraction_method": "nlp_enhanced", "patterns_found": []}
        
        # Try NLP-based extraction first
        if nlp is not None:
            try:
                # Extract using spaCy NLP
                nlp_results = self._extract_with_nlp(text)
                patient_info.update(nlp_results.get("patient_info", {}))
                clinical_data.update(nlp_results.get("clinical_data", {}))
                parsing_details["patterns_found"].extend(nlp_results.get("patterns_found", []))
            except Exception as e:
                logger.warning(f"NLP extraction failed: {e}, falling back to patterns")
        
        # Fallback to pattern-based extraction for missing fields
        pattern_results = self._extract_with_patterns(text)
        
        # Merge results, preferring NLP results when available
        for field, data in pattern_results.get("patient_info", {}).items():
            if field not in patient_info:
                patient_info[field] = data
                
        for field, data in pattern_results.get("clinical_data", {}).items():
            if field not in clinical_data:
                clinical_data[field] = data
                
        parsing_details["patterns_found"].extend(pattern_results.get("patterns_found", []))
        
        return {
            "patient_info": patient_info,
            "clinical_data": clinical_data,
            "parsing_details": parsing_details
        }
    
    def _extract_with_nlp(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy NLP."""
        patient_info = {}
        clinical_data = {}
        patterns_found = []
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Extract person names
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if persons:
            # Find most likely patient name (often appears multiple times)
            name_counts = {}
            for person in persons:
                # Clean up the name
                clean_name = re.sub(r'\s+', ' ', person.strip())
                if len(clean_name.split()) >= 2 and not re.search(r'\d', clean_name):
                    name_counts[clean_name] = name_counts.get(clean_name, 0) + 1
            
            if name_counts:
                # Use the most frequent valid name
                patient_name = max(name_counts, key=name_counts.get)
                patient_info["name"] = {
                    "value": patient_name,
                    "confidence": 0.85,
                    "confidence_level": "medium",
                    "source_page": 1,
                    "extraction_method": "mistral_ocr"
                }
                patterns_found.append("patient_name_nlp")
        
        # Extract dates for DOB
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        dob_pattern = re.compile(r'\b\d{1,2}[\/\-]\d{1,2}[\/\-](?:\d{2}|\d{4})\b')
        for date in dates:
            if dob_pattern.match(date.strip()):
                patient_info["date_of_birth"] = {
                    "value": date.strip(),
                    "confidence": 0.90,
                    "confidence_level": "high",
                    "source_page": 1,
                    "extraction_method": "mistral_ocr"
                }
                patterns_found.append("date_of_birth_nlp")
                break
        
        # Extract medical conditions using pattern matching on sentences
        sentences = [sent.text for sent in doc.sents]
        medical_keywords = [
            "multiple sclerosis", "ms", "diabetes", "hypertension", "cancer", 
            "arthritis", "depression", "anxiety", "asthma", "copd"
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in medical_keywords:
                if keyword in sentence_lower:
                    # Extract the full diagnosis from the sentence
                    diagnosis_match = re.search(
                        rf'{keyword}[^.]*(?:\([^)]*\))?\s*(?:\[[^\]]*\])?',
                        sentence,
                        re.IGNORECASE
                    )
                    if diagnosis_match:
                        diagnosis = diagnosis_match.group(0).strip()
                        clinical_data["primary_diagnosis"] = {
                            "value": diagnosis,
                            "confidence": 0.80,
                            "confidence_level": "medium",
                            "source_page": 1,
                            "extraction_method": "mistral_ocr"
                        }
                        patterns_found.append("diagnosis_nlp")
                        break
            if "primary_diagnosis" in clinical_data:
                break
        
        return {
            "patient_info": patient_info,
            "clinical_data": clinical_data,
            "patterns_found": patterns_found
        }
    
    def _extract_with_patterns(self, text: str) -> Dict[str, Any]:
        """Fallback pattern-based extraction."""
        patient_info = {}
        clinical_data = {}
        patterns_found = []
        
        # Generic date pattern for DOB
        dob_pattern = re.compile(r'(?:DOB|Date.of.Birth|Birth.Date)\s*:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', re.IGNORECASE)
        match = dob_pattern.search(text)
        if match:
            patient_info["date_of_birth"] = {
                "value": match.group(1),
                "confidence": 0.85,
                "confidence_level": "medium",
                "source_page": 1,
                "extraction_method": "mistral_ocr"
            }
            patterns_found.append("date_of_birth_pattern")
        
        # Generic insurance ID pattern
        insurance_pattern = re.compile(r'(?:Member.ID|Insurance.ID|Policy)\s*:?\s*([A-Za-z0-9]+)', re.IGNORECASE)
        match = insurance_pattern.search(text)
        if match:
            patient_info["insurance_id"] = {
                "value": match.group(1),
                "confidence": 0.75,
                "confidence_level": "medium",
                "source_page": 1,
                "extraction_method": "mistral_ocr"
            }
            patterns_found.append("insurance_id_pattern")
        
        return {
            "patient_info": patient_info,
            "clinical_data": clinical_data,
            "patterns_found": patterns_found
        }
    
    def _parse_pa_form(self, text: str) -> Dict[str, Any]:
        """Parse PA form text for form fields."""
        # Similar parsing logic for PA forms
        return {
            "patient_info": {},
            "clinical_data": {},
            "parsing_details": {"type": "pa_form"}
        }
    
    def _parse_general_document(self, text: str) -> Dict[str, Any]:
        """Parse general document text."""
        return {
            "patient_info": {},
            "clinical_data": {},
            "parsing_details": {"type": "general"}
        }
    
    def _calculate_confidence(self, text: str, parsed_data: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score for extracted data.
        
        Args:
            text: Extracted text
            parsed_data: Parsed structured data
            
        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        if not text or len(text) < 100:
            return 0.20  # Very short text, low confidence
        
        # Count successfully parsed fields
        patient_fields = len(parsed_data.get("patient_info", {}))
        clinical_fields = len(parsed_data.get("clinical_data", {}))
        total_fields = patient_fields + clinical_fields
        
        # Base confidence from text length and structure
        text_length_score = min(len(text) / 5000, 1.0)  # Up to 5000 chars = max score
        
        # Structure score based on parsed fields
        structure_score = min(total_fields / 10, 1.0)  # Up to 10 fields = max score
        
        # Text quality indicators
        import re
        quality_indicators = [
            bool(re.search(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}', text)),  # Has dates
            bool(re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', text)),  # Has proper names
            bool(re.search(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', text)),  # Has SSN-like numbers
            bool(re.search(r'[A-Za-z0-9]{8,}', text))  # Has ID-like strings
        ]
        
        quality_score = sum(quality_indicators) / len(quality_indicators)
        
        # Weighted average
        confidence = (0.4 * text_length_score + 0.4 * structure_score + 0.2 * quality_score)
        
        # Mistral OCR typically provides high quality, so boost confidence
        confidence = min(confidence + 0.15, 0.95)  # Boost but cap at 95%
        
        return confidence
    
    def _calculate_cost(self, pages_processed: int) -> Dict[str, float]:
        """
        Calculate processing cost.
        
        Args:
            pages_processed: Number of pages processed
            
        Returns:
            Cost breakdown dictionary
        """
        regular_cost = pages_processed * self.cost_per_page
        batch_cost = pages_processed * self.batch_cost_per_page
        
        return {
            "pages_processed": pages_processed,
            "regular_cost_usd": regular_cost,
            "batch_cost_usd": batch_cost,
            "cost_per_page": self.cost_per_page
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get current service status and health information.
        
        Returns:
            Service status dictionary
        """
        return {
            "service_name": "mistral_ocr_service",
            "service_role": "primary_extraction",
            "client_initialized": self.client is not None,
            "model_name": self.model_name,
            "api_key_configured": bool(self.api_key),
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "confidence_thresholds": self.confidence_thresholds,
            "cost_per_page": self.cost_per_page,
            "cost_effectiveness": "high"
        }


# Global service instance
mistral_service = MistralService()


def get_mistral_service() -> MistralService:
    """
    Get the global Mistral service instance.
    
    Returns:
        MistralService instance for dependency injection
    """
    return mistral_service