# backend/app/services/gemini_service_fallback.py
"""
Gemini AI service for FALLBACK PDF extraction and vision-based document processing.

This service provides AI-powered extraction capabilities using Google's Gemini API,
serving as a FALLBACK when Mistral OCR fails or provides low confidence results.
Secondary service for vision-based processing of complex medical forms.
"""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import json
import time

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    genai = None
    HarmCategory = None
    HarmBlockThreshold = None

from app.models.schemas import ExtractedField, ConfidenceLevel, PatientInfo, ClinicalData
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class GeminiService:
    """
    FALLBACK service for AI-powered document extraction using Google Gemini API.
    
    Provides vision-based PDF processing when Mistral OCR fails or returns
    low confidence results. Secondary extraction method with vision capabilities.
    """
    
    def __init__(self):
        """Initialize Gemini service with API configuration."""
        self.client = None
        self.model_name = settings.GEMINI_MODEL
        self.api_key = settings.GEMINI_API_KEY
        self.context_window = settings.GEMINI_CONTEXT_WINDOW
        self.max_output_tokens = 8192  # Reasonable limit for output
        self.timeout_seconds = settings.PROCESSING_TIMEOUT_SECONDS
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay_base = 1.0  # Base delay in seconds
        self.retry_backoff_factor = 2.0
        
        # Confidence thresholds for Gemini extraction
        self.confidence_thresholds = {
            "high": 0.85,      # Well-structured, clear text
            "medium": 0.70,    # Some ambiguity but likely correct
            "low": 0.50,       # Significant uncertainty
            "very_low": 0.30   # High uncertainty, manual review needed
        }
        
        # Safety settings for medical document processing
        if HarmCategory and HarmBlockThreshold:
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        else:
            self.safety_settings = {}
        
        logger.info("Gemini service initialized as FALLBACK extraction method")
    
    def initialize_client(self) -> bool:
        """
        Initialize the Gemini API client.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not genai:
            logger.error("google-generativeai package not installed")
            return False
        
        if not self.api_key:
            logger.error("GEMINI_API_KEY not configured")
            return False
        
        try:
            genai.configure(api_key=self.api_key)
            
            # Test connection with a simple request
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                "Test connection. Respond with 'OK'.",
                safety_settings=self.safety_settings
            )
            
            if response and response.text:
                self.client = model
                logger.info(f"Gemini client initialized successfully with model {self.model_name}")
                return True
            else:
                logger.error("Failed to get response from Gemini API")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            return False
    
    async def extract_from_pdf(
        self, 
        pdf_path: Union[str, Path], 
        extraction_type: str = "medical_referral",
        page_numbers: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from PDF using Gemini vision capabilities.
        
        Args:
            pdf_path: Path to the PDF file
            extraction_type: Type of extraction ("medical_referral", "pa_form", "general")
            page_numbers: Optional list of specific pages to process
            
        Returns:
            Dictionary containing extracted data with confidence scores
        """
        if not self.client and not self.initialize_client():
            raise RuntimeError("Gemini client not initialized")
        
        try:
            # Convert PDF to images for vision processing
            pdf_images = await self._pdf_to_images(pdf_path, page_numbers)
            
            if not pdf_images:
                raise ValueError("No images extracted from PDF")
            
            # Process each page with Gemini vision
            extracted_pages = []
            for page_data in pdf_images:
                page_result = await self._process_page_with_retry(
                    page_data, extraction_type
                )
                extracted_pages.append(page_result)
            
            # Combine and structure results
            combined_result = self._combine_page_results(extracted_pages, extraction_type)
            
            # Add metadata
            combined_result.update({
                "extraction_method": "gemini_vision",
                "model_used": self.model_name,
                "pages_processed": len(extracted_pages),
                "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                "extraction_type": extraction_type
            })
            
            logger.info(
                f"Gemini extraction completed: {len(extracted_pages)} pages, "
                f"type: {extraction_type}"
            )
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Gemini PDF extraction failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}, Details: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    async def _pdf_to_images(
        self, 
        pdf_path: Union[str, Path], 
        page_numbers: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert PDF pages to images for vision processing.
        
        Args:
            pdf_path: Path to PDF file
            page_numbers: Optional specific pages to convert
            
        Returns:
            List of page data with base64-encoded images
        """
        try:
            import fitz  # PyMuPDF for PDF to image conversion
        except ImportError:
            logger.error("PyMuPDF not installed - required for PDF to image conversion")
            raise ImportError("PyMuPDF (fitz) required for Gemini vision processing")
        
        pdf_document = fitz.open(pdf_path)
        pages_data = []
        
        try:
            total_pages = len(pdf_document)
            pages_to_process = page_numbers or list(range(total_pages))
            
            for page_num in pages_to_process:
                if page_num >= total_pages:
                    logger.warning(f"Page {page_num} does not exist in PDF")
                    continue
                
                page = pdf_document[page_num]
                
                # Convert page to image (PNG format)
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x scaling for better quality
                img_data = pix.tobytes("png")
                
                # Encode to base64 for API
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                pages_data.append({
                    "page_number": page_num + 1,  # 1-indexed
                    "image_base64": img_base64,
                    "image_format": "png",
                    "width": pix.width,
                    "height": pix.height
                })
                
        finally:
            pdf_document.close()
        
        logger.info(f"Converted {len(pages_data)} PDF pages to images")
        return pages_data
    
    async def _process_page_with_retry(
        self, 
        page_data: Dict[str, Any], 
        extraction_type: str
    ) -> Dict[str, Any]:
        """
        Process a single page with Gemini vision API with retry logic.
        
        Args:
            page_data: Page data with base64 image
            extraction_type: Type of extraction to perform
            
        Returns:
            Extracted data from the page
        """
        for attempt in range(self.max_retries):
            try:
                result = await self._process_single_page(page_data, extraction_type)
                return result
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Gemini processing failed after {self.max_retries} attempts: {e}"
                    )
                    return {
                        "page_number": page_data["page_number"],
                        "success": False,
                        "error": str(e),
                        "extracted_data": {},
                        "confidence": 0.0
                    }
                
                # Calculate retry delay with exponential backoff
                delay = self.retry_delay_base * (self.retry_backoff_factor ** attempt)
                logger.warning(
                    f"Gemini processing attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.1f} seconds..."
                )
                await asyncio.sleep(delay)
    
    async def _process_single_page(
        self, 
        page_data: Dict[str, Any], 
        extraction_type: str
    ) -> Dict[str, Any]:
        """
        Process a single page with Gemini vision API.
        
        Args:
            page_data: Page data with base64 image
            extraction_type: Type of extraction to perform
            
        Returns:
            Extracted data from the page
        """
        # Create prompt based on extraction type
        prompt = self._create_extraction_prompt(extraction_type)
        
        # Prepare image for Gemini
        image_data = {
            "mime_type": "image/png",
            "data": page_data["image_base64"]
        }
        
        # Make API request
        try:
            response = self.client.generate_content(
                [prompt, image_data],
                safety_settings=self.safety_settings,
                generation_config={
                    "max_output_tokens": self.max_output_tokens,
                    "temperature": 0.1,  # Low temperature for consistent extraction
                }
            )
            
            if not response or not response.text:
                raise ValueError("Empty response from Gemini API")
            
            # Parse JSON response
            extracted_data = self._parse_gemini_response(response.text)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(extracted_data, page_data)
            
            return {
                "page_number": page_data["page_number"],
                "success": True,
                "extracted_data": extracted_data,
                "confidence": confidence,
                "raw_response": response.text[:1000],  # First 1000 chars for debugging
                "processing_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"Gemini API error on page {page_data['page_number']}: {e}")
            raise
    
    def _create_extraction_prompt(self, extraction_type: str) -> str:
        """
        Create extraction prompt based on document type.
        
        Args:
            extraction_type: Type of document to extract from
            
        Returns:
            Formatted prompt string
        """
        base_prompt = """
You are a medical document processing AI. Extract structured information from this document image.
Return your response as a valid JSON object with the following structure:

"""
        
        if extraction_type == "medical_referral":
            schema = {
                "patient_info": {
                    "name": {"value": "string", "confidence": "float"},
                    "date_of_birth": {"value": "string", "confidence": "float"},
                    "gender": {"value": "string", "confidence": "float"},
                    "insurance_id": {"value": "string", "confidence": "float"},
                    "phone": {"value": "string", "confidence": "float"},
                    "address": {"value": "string", "confidence": "float"}
                },
                "clinical_info": {
                    "primary_diagnosis": {"value": "string", "confidence": "float"},
                    "diagnosis_codes": [{"value": "string", "confidence": "float"}],
                    "provider_name": {"value": "string", "confidence": "float"},
                    "provider_npi": {"value": "string", "confidence": "float"},
                    "referral_date": {"value": "string", "confidence": "float"},
                    "treatment_plan": {"value": "string", "confidence": "float"}
                }
            }
        
        elif extraction_type == "pa_form":
            schema = {
                "form_fields": {
                    "patient_name": {"value": "string", "confidence": "float"},
                    "member_id": {"value": "string", "confidence": "float"},
                    "prescriber_name": {"value": "string", "confidence": "float"},
                    "medication_requested": {"value": "string", "confidence": "float"},
                    "diagnosis": {"value": "string", "confidence": "float"}
                }
            }
        
        else:  # general
            schema = {
                "text_content": "string",
                "key_fields": [{"field_name": "string", "value": "string", "confidence": "float"}]
            }
        
        return f"{base_prompt}{json.dumps(schema, indent=2)}\n\nImportant:\n- Set confidence values between 0.0 and 1.0 based on text clarity\n- Use 'null' for missing information\n- Extract dates in MM/DD/YYYY format when possible\n- Be precise and conservative with confidence scores"
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate Gemini API response.
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Parsed JSON data
        """
        try:
            # Clean up response text (remove markdown formatting if present)
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]  # Remove ```json
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]  # Remove trailing ```
            
            # Parse JSON
            parsed_data = json.loads(cleaned_text)
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.debug(f"Raw response: {response_text[:500]}")
            
            # Return fallback structure
            return {
                "error": "json_parse_failed",
                "raw_text": response_text[:1000],
                "confidence": 0.1
            }
    
    def _calculate_confidence(
        self, 
        extracted_data: Dict[str, Any], 
        page_data: Dict[str, Any]
    ) -> float:
        """
        Calculate overall confidence score for extracted data.
        
        Args:
            extracted_data: Parsed extraction results
            page_data: Original page data
            
        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        if "error" in extracted_data:
            return 0.1
        
        # Collect individual confidence scores
        confidence_scores = []
        
        def collect_confidences(data):
            if isinstance(data, dict):
                if "confidence" in data:
                    conf_val = data["confidence"]
                    # Handle string confidence values that can be converted to float
                    try:
                        if isinstance(conf_val, (int, float)):
                            confidence_scores.append(float(conf_val))
                        elif isinstance(conf_val, str) and conf_val.replace('.', '', 1).isdigit():
                            confidence_scores.append(float(conf_val))
                    except (ValueError, TypeError):
                        # Skip invalid confidence values
                        pass
                else:
                    for value in data.values():
                        collect_confidences(value)
            elif isinstance(data, list):
                for item in data:
                    collect_confidences(item)
        
        collect_confidences(extracted_data)
        
        if not confidence_scores:
            # No explicit confidence scores, estimate based on data completeness
            non_null_fields = self._count_non_null_fields(extracted_data)
            if non_null_fields > 10:
                return 0.75
            elif non_null_fields > 5:
                return 0.60
            elif non_null_fields > 0:
                return 0.45
            else:
                return 0.20
        
        # Calculate weighted average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Apply adjustment based on data completeness
        completeness_factor = min(len(confidence_scores) / 10, 1.0)  # Expect ~10 fields
        adjusted_confidence = avg_confidence * (0.7 + 0.3 * completeness_factor)
        
        return min(max(adjusted_confidence, 0.0), 1.0)
    
    def _count_non_null_fields(self, data: Any) -> int:
        """Count non-null fields in extracted data."""
        count = 0
        
        if isinstance(data, dict):
            for value in data.values():
                if value is not None and value != "":
                    if isinstance(value, dict) and "value" in value:
                        if value["value"] is not None and value["value"] != "":
                            count += 1
                    else:
                        count += self._count_non_null_fields(value)
        elif isinstance(data, list):
            for item in data:
                count += self._count_non_null_fields(item)
        
        return count
    
    def _combine_page_results(
        self, 
        page_results: List[Dict[str, Any]], 
        extraction_type: str
    ) -> Dict[str, Any]:
        """
        Combine results from multiple pages into a single structured result.
        
        Args:
            page_results: List of individual page results
            extraction_type: Type of extraction performed
            
        Returns:
            Combined extraction result
        """
        combined_data = {
            "patient_info": {},
            "clinical_data": {},
            "pages": page_results,
            "overall_confidence": 0.0,
            "successful_pages": 0,
            "failed_pages": 0
        }
        
        # Track successful vs failed pages
        successful_results = []
        for result in page_results:
            if result.get("success", False):
                successful_results.append(result)
                combined_data["successful_pages"] += 1
            else:
                combined_data["failed_pages"] += 1
        
        if not successful_results:
            combined_data["overall_confidence"] = 0.0
            return combined_data
        
        # Merge patient info across pages (take highest confidence values)
        patient_info_merged = {}
        clinical_data_merged = {}
        
        for result in successful_results:
            extracted = result.get("extracted_data", {})
            
            # Merge patient info
            page_patient_info = extracted.get("patient_info", {})
            for field, data in page_patient_info.items():
                if isinstance(data, dict) and "value" in data and "confidence" in data:
                    current_conf_raw = patient_info_merged.get(field, {}).get("confidence", 0.0)
                    # Convert current confidence to float
                    try:
                        if isinstance(current_conf_raw, (int, float)):
                            current_confidence = float(current_conf_raw)
                        elif isinstance(current_conf_raw, str):
                            current_confidence = float(current_conf_raw) if current_conf_raw.replace('.', '', 1).isdigit() else 0.0
                        else:
                            current_confidence = 0.0
                    except (ValueError, TypeError):
                        current_confidence = 0.0
                        
                    # Handle None, string, and numeric confidence values for new data
                    try:
                        conf_val = data["confidence"]
                        if conf_val is None:
                            data_confidence = 0.0
                        elif isinstance(conf_val, (int, float)):
                            data_confidence = float(conf_val)
                        elif isinstance(conf_val, str):
                            data_confidence = float(conf_val) if conf_val.replace('.', '', 1).isdigit() else 0.0
                        else:
                            data_confidence = 0.0
                    except (ValueError, TypeError):
                        data_confidence = 0.0
                    
                    if data_confidence > current_confidence:
                        patient_info_merged[field] = data
            
            # Merge clinical data
            page_clinical_data = extracted.get("clinical_info", {})
            for field, data in page_clinical_data.items():
                if isinstance(data, dict) and "value" in data and "confidence" in data:
                    current_conf_raw = clinical_data_merged.get(field, {}).get("confidence", 0.0)
                    # Convert current confidence to float
                    try:
                        if isinstance(current_conf_raw, (int, float)):
                            current_confidence = float(current_conf_raw)
                        elif isinstance(current_conf_raw, str):
                            current_confidence = float(current_conf_raw) if current_conf_raw.replace('.', '', 1).isdigit() else 0.0
                        else:
                            current_confidence = 0.0
                    except (ValueError, TypeError):
                        current_confidence = 0.0
                        
                    # Handle None, string, and numeric confidence values for new data
                    try:
                        conf_val = data["confidence"]
                        if conf_val is None:
                            data_confidence = 0.0
                        elif isinstance(conf_val, (int, float)):
                            data_confidence = float(conf_val)
                        elif isinstance(conf_val, str):
                            data_confidence = float(conf_val) if conf_val.replace('.', '', 1).isdigit() else 0.0
                        else:
                            data_confidence = 0.0
                    except (ValueError, TypeError):
                        data_confidence = 0.0
                    
                    if data_confidence > current_confidence:
                        clinical_data_merged[field] = data
                elif isinstance(data, list):
                    # Handle lists (e.g., diagnosis_codes)
                    if field not in clinical_data_merged:
                        clinical_data_merged[field] = []
                    clinical_data_merged[field].extend(data)
        
        combined_data["patient_info"] = patient_info_merged
        combined_data["clinical_data"] = clinical_data_merged
        
        # Calculate overall confidence
        page_confidences = [r["confidence"] for r in successful_results]
        combined_data["overall_confidence"] = sum(page_confidences) / len(page_confidences)
        
        return combined_data
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get current service status and health information.
        
        Returns:
            Service status dictionary
        """
        return {
            "service_name": "gemini_service",
            "service_role": "fallback_extraction",
            "client_initialized": self.client is not None,
            "model_name": self.model_name,
            "api_key_configured": bool(self.api_key),
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "confidence_thresholds": self.confidence_thresholds
        }


# Global service instance
gemini_service = GeminiService()


def get_gemini_service() -> GeminiService:
    """
    Get the global Gemini service instance.
    
    Returns:
        GeminiService instance for dependency injection
    """
    return gemini_service