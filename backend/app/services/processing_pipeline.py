"""
Processing Pipeline for Prior Authorization document automation.

This service orchestrates the complete workflow from document upload to filled PA forms,
integrating OCR, field detection, mapping, and form filling with comprehensive
error handling and progress tracking.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone
from enum import Enum

from models.schemas import (
    ProcessingStatusEnum, ExtractedData, PAFormField, MissingField, 
    ProcessingResult, ConfidenceLevel
)
from core.config import get_settings
from .mistral_service import get_mistral_service
from .gemini_service_fallback import get_gemini_service
from .openai_service import get_openai_service
from .widget_detector import get_widget_detector
from .field_mapper import get_field_mapper
from .concurrent_processor import get_async_processor, create_progress_callback

logger = logging.getLogger(__name__)
settings = get_settings()


class ProcessingStage(str, Enum):
    """Stages of document processing pipeline."""
    INITIALIZING = "initializing"
    VALIDATING_FILES = "validating_files"
    EXTRACTING_REFERRAL = "extracting_referral"
    DETECTING_PA_FIELDS = "detecting_pa_fields"
    MAPPING_FIELDS = "mapping_fields"
    APPLYING_THRESHOLDS = "applying_thresholds"
    GENERATING_OUTPUT = "generating_output"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingPipeline:
    """
    Comprehensive processing pipeline for PA form automation.
    
    Orchestrates the complete workflow from document upload to filled forms,
    managing OCR extraction, field detection, intelligent mapping, and
    confidence-based validation with detailed progress tracking.
    """
    
    def __init__(self):
        """Initialize the processing pipeline with service dependencies."""
        self.mistral_service = get_mistral_service()
        self.gemini_service = get_gemini_service()
        self.openai_service = get_openai_service()
        self.widget_detector = get_widget_detector()
        self.field_mapper = get_field_mapper()
        self.async_processor = get_async_processor()
        
        # Processing configuration
        self.confidence_thresholds = {
            "auto_fill": 0.90,      # High confidence - auto-fill
            "flag_review": 0.70,    # Medium confidence - fill with flag
            "manual_entry": 0.50,   # Low confidence - mark as missing
            "reject": 0.30          # Very low confidence - reject value
        }
        
        # Processing timeouts
        self.timeout_config = {
            "referral_extraction": 300,  # 5 minutes for large referrals
            "pa_field_detection": 60,    # 1 minute for PA form analysis
            "field_mapping": 120,        # 2 minutes for AI mapping
            "total_processing": 600      # 10 minutes total timeout
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "min_extracted_fields": 3,      # Minimum fields to proceed
            "min_pa_fields": 5,              # Minimum PA fields required
            "min_mapping_success": 0.6,      # 60% of fields must map
            "min_critical_fields": 0.8      # 80% of critical fields required
        }
        
        logger.info("Processing pipeline initialized with comprehensive workflow")
    
    async def process_documents(
        self,
        session_id: str,
        referral_pdf_path: Union[str, Path],
        pa_form_pdf_path: Union[str, Path],
        progress_callback: Optional[callable] = None
    ) -> ProcessingResult:
        """
        Process referral packet and PA form to generate filled form.
        
        Args:
            session_id: Unique session identifier
            referral_pdf_path: Path to referral packet PDF
            pa_form_pdf_path: Path to PA form PDF
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete processing result with filled form and reports
        """
        start_time = datetime.now(timezone.utc)
        
        # Initialize progress tracking
        current_stage = ProcessingStage.INITIALIZING
        progress_data = {
            "session_id": session_id,
            "stage": current_stage,
            "progress": 0,
            "message": "Initializing document processing..."
        }
        
        if progress_callback:
            await progress_callback(progress_data)
        
        try:
            logger.info(f"Starting document processing for session {session_id}")
            
            # Stage 1: Validate input files
            current_stage = ProcessingStage.VALIDATING_FILES
            progress_data.update({"stage": current_stage, "progress": 10, 
                                "message": "Validating input files..."})
            if progress_callback:
                await progress_callback(progress_data)
            
            validation_result = await self._validate_input_files(
                referral_pdf_path, pa_form_pdf_path
            )
            if not validation_result["valid"]:
                raise ValueError(f"File validation failed: {validation_result['error']}")
            
            # Stage 2: Extract referral data
            current_stage = ProcessingStage.EXTRACTING_REFERRAL
            progress_data.update({"stage": current_stage, "progress": 25, 
                                "message": "Extracting data from referral packet..."})
            if progress_callback:
                await progress_callback(progress_data)
            
            referral_data = await self.extract_referral_data(
                referral_pdf_path, session_id, progress_callback
            )
            
            # Stage 3: Detect PA form fields
            current_stage = ProcessingStage.DETECTING_PA_FIELDS
            progress_data.update({"stage": current_stage, "progress": 50, 
                                "message": "Analyzing PA form structure..."})
            if progress_callback:
                await progress_callback(progress_data)
            
            pa_fields = await self.detect_pa_fields(
                pa_form_pdf_path, session_id
            )
            
            # Stage 4: Map data to fields
            current_stage = ProcessingStage.MAPPING_FIELDS
            progress_data.update({"stage": current_stage, "progress": 70, 
                                "message": "Mapping referral data to PA form fields..."})
            if progress_callback:
                await progress_callback(progress_data)
            
            field_mappings = await self.map_data_to_fields(
                referral_data, pa_fields, session_id
            )
            
            # Stage 5: Apply confidence thresholds
            current_stage = ProcessingStage.APPLYING_THRESHOLDS
            progress_data.update({"stage": current_stage, "progress": 85, 
                                "message": "Applying confidence thresholds and validation..."})
            if progress_callback:
                await progress_callback(progress_data)
            
            final_mappings, missing_fields = await self.apply_confidence_thresholds(
                field_mappings, pa_fields
            )
            
            # Stage 6: Generate final result
            current_stage = ProcessingStage.GENERATING_OUTPUT
            progress_data.update({"stage": current_stage, "progress": 95, 
                                "message": "Generating final output and reports..."})
            if progress_callback:
                await progress_callback(progress_data)
            
            # Create processing result
            processing_result = await self._create_processing_result(
                session_id=session_id,
                referral_data=referral_data,
                pa_fields=pa_fields,
                field_mappings=final_mappings,
                missing_fields=missing_fields,
                start_time=start_time
            )
            
            # Final stage: Completed
            current_stage = ProcessingStage.COMPLETED
            progress_data.update({"stage": current_stage, "progress": 100, 
                                "message": "Processing completed successfully!"})
            if progress_callback:
                await progress_callback(progress_data)
            
            logger.info(f"Document processing completed for session {session_id}")
            return processing_result
            
        except Exception as e:
            logger.error(f"Document processing failed for session {session_id}: {e}")
            
            # Update progress with error
            progress_data.update({
                "stage": ProcessingStage.FAILED, 
                "progress": 0, 
                "message": f"Processing failed: {str(e)}"
            })
            if progress_callback:
                await progress_callback(progress_data)
            
            # Create failed result
            return await self._create_failed_result(
                session_id, str(e), start_time
            )
    
    async def extract_referral_data(
        self,
        referral_pdf_path: Union[str, Path],
        session_id: str,
        progress_callback: Optional[callable] = None
    ) -> ExtractedData:
        """
        Extract structured data from referral packet using primary and fallback OCR.
        
        Args:
            referral_pdf_path: Path to referral packet PDF
            session_id: Session identifier for tracking
            progress_callback: Optional progress callback
            
        Returns:
            Structured extracted data with confidence scores
        """
        referral_path = Path(referral_pdf_path)
        
        try:
            logger.info(f"Extracting referral data from {referral_path.name}")
            
            # Try primary extraction with Mistral OCR
            try:
                logger.info("Attempting primary extraction with Mistral OCR")
                extraction_result = await self.mistral_service.extract_from_pdf(
                    referral_path, 
                    extraction_type="medical_referral",
                    include_images=True
                )
                
                # Check extraction quality
                if self._validate_extraction_quality(extraction_result):
                    logger.info("Mistral extraction successful - high quality results")
                    return await self._structure_extracted_data(
                        extraction_result, session_id, "mistral_ocr"
                    )
                else:
                    logger.warning("Mistral extraction quality below threshold - trying fallback")
                    
            except Exception as e:
                logger.warning(f"Mistral extraction failed: {e} - trying fallback")
            
            # Fallback to Gemini Vision
            try:
                logger.info("Attempting fallback extraction with Gemini Vision")
                extraction_result = await self.gemini_service.extract_from_pdf(
                    referral_path,
                    extraction_type="medical_referral"
                )
                
                if self._validate_extraction_quality(extraction_result):
                    logger.info("Gemini extraction successful")
                    return await self._structure_extracted_data(
                        extraction_result, session_id, "gemini_vision"
                    )
                else:
                    logger.error("Both primary and fallback extraction failed quality checks")
                    raise RuntimeError("Extraction quality insufficient from all methods")
                    
            except Exception as e:
                logger.error(f"Gemini fallback extraction failed: {e}")
                raise RuntimeError("All extraction methods failed")
        
        except Exception as e:
            logger.error(f"Referral data extraction failed: {e}")
            raise
    
    async def detect_pa_fields(
        self,
        pa_form_pdf_path: Union[str, Path],
        session_id: str
    ) -> Dict[str, PAFormField]:
        """
        Detect and analyze PA form fields using widget detection.
        
        Args:
            pa_form_pdf_path: Path to PA form PDF
            session_id: Session identifier for tracking
            
        Returns:
            Dictionary of detected PA form fields with properties
        """
        pa_form_path = Path(pa_form_pdf_path)
        
        try:
            logger.info(f"Detecting PA form fields in {pa_form_path.name}")
            
            # Detect form fields using widget detector
            detection_result = self.widget_detector.detect_form_fields(
                pa_form_path,
                detection_method="hybrid_detection"
            )
            
            if not detection_result.get("success", False):
                raise RuntimeError(f"Field detection failed: {detection_result.get('error')}")
            
            detected_fields = detection_result.get("fields", {})
            
            # Validate minimum field requirements
            if len(detected_fields) < self.quality_thresholds["min_pa_fields"]:
                logger.warning(f"Only {len(detected_fields)} fields detected, below minimum threshold")
            
            # Convert to PAFormField objects
            pa_form_fields = {}
            for field_id, field_data in detected_fields.items():
                try:
                    pa_field = self._create_pa_form_field(field_id, field_data)
                    pa_form_fields[field_id] = pa_field
                except Exception as e:
                    logger.warning(f"Failed to create PA field {field_id}: {e}")
            
            logger.info(f"Successfully detected {len(pa_form_fields)} PA form fields")
            return pa_form_fields
            
        except Exception as e:
            logger.error(f"PA field detection failed: {e}")
            raise
    
    async def map_data_to_fields(
        self,
        referral_data: ExtractedData,
        pa_fields: Dict[str, PAFormField],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Map extracted referral data to PA form fields using AI-powered analysis.
        
        Args:
            referral_data: Extracted data from referral packet
            pa_fields: Detected PA form fields
            session_id: Session identifier for tracking
            
        Returns:
            Field mappings with confidence scores
        """
        try:
            logger.info(f"Mapping referral data to {len(pa_fields)} PA form fields")
            
            # Prepare data for OpenAI mapping
            referral_dict = self._prepare_referral_data_for_mapping(referral_data)
            pa_fields_dict = self._prepare_pa_fields_for_mapping(pa_fields)
            
            # Use OpenAI service for intelligent field mapping
            mapping_result = await self.openai_service.extract_and_map_fields(
                referral_dict, pa_fields_dict
            )
            
            if not mapping_result:
                raise RuntimeError("OpenAI field mapping returned empty result")
            
            # Enhance mappings with field mapper utilities
            enhanced_mappings = await self._enhance_field_mappings(
                mapping_result, referral_data, pa_fields
            )
            
            # Validate mapping quality
            mapping_quality = self._validate_mapping_quality(enhanced_mappings, pa_fields)
            logger.info(f"Field mapping completed with {mapping_quality:.2%} success rate")
            
            return enhanced_mappings
            
        except Exception as e:
            logger.error(f"Data mapping failed: {e}")
            raise
    
    async def apply_confidence_thresholds(
        self,
        field_mappings: Dict[str, Any],
        pa_fields: Dict[str, PAFormField]
    ) -> Tuple[Dict[str, Any], List[MissingField]]:
        """
        Apply confidence thresholds to determine final field values and missing fields.
        
        Args:
            field_mappings: Raw field mappings with confidence scores
            pa_fields: PA form field definitions
            
        Returns:
            Tuple of (final_mappings, missing_fields)
        """
        try:
            logger.info("Applying confidence thresholds to field mappings")
            
            final_mappings = {}
            missing_fields = []
            
            # Process each mapping
            for field_name, mapping_data in field_mappings.get("field_mappings", {}).items():
                confidence = mapping_data.get("confidence", 0.0)
                mapped_value = mapping_data.get("mapped_value", "")
                
                # Determine action based on confidence
                if confidence >= self.confidence_thresholds["auto_fill"]:
                    # High confidence - auto-fill
                    final_mappings[field_name] = {
                        **mapping_data,
                        "action": "auto_fill",
                        "validation_status": "approved"
                    }
                    
                elif confidence >= self.confidence_thresholds["flag_review"]:
                    # Medium confidence - fill with review flag
                    final_mappings[field_name] = {
                        **mapping_data,
                        "action": "fill_with_flag",
                        "validation_status": "review_required"
                    }
                    
                elif confidence >= self.confidence_thresholds["manual_entry"]:
                    # Low confidence - suggest value but mark for manual entry
                    missing_field = MissingField(
                        field_name=field_name,
                        display_label=pa_fields.get(field_name, {}).get("display_label", field_name),
                        required=pa_fields.get(field_name, {}).get("required", False),
                        reason=f"Low confidence mapping ({confidence:.2f})",
                        confidence=confidence,
                        suggested_value=mapped_value,
                        manual_review_required=True,
                        priority="medium"
                    )
                    missing_fields.append(missing_field)
                    
                else:
                    # Very low confidence - reject and mark as missing
                    missing_field = MissingField(
                        field_name=field_name,
                        display_label=pa_fields.get(field_name, {}).get("display_label", field_name),
                        required=pa_fields.get(field_name, {}).get("required", False),
                        reason=f"Insufficient confidence ({confidence:.2f})",
                        confidence=None,
                        suggested_value=None,
                        manual_review_required=True,
                        priority="high" if pa_fields.get(field_name, {}).get("required", False) else "low"
                    )
                    missing_fields.append(missing_field)
            
            # Check for unmapped required fields
            for field_name, pa_field in pa_fields.items():
                if field_name not in field_mappings.get("field_mappings", {}) and pa_field.required:
                    missing_field = MissingField(
                        field_name=field_name,
                        display_label=pa_field.display_label,
                        required=True,
                        reason="No mapping found in referral data",
                        confidence=None,
                        suggested_value=None,
                        manual_review_required=True,
                        priority="high"
                    )
                    missing_fields.append(missing_field)
            
            logger.info(f"Applied thresholds: {len(final_mappings)} approved, {len(missing_fields)} missing")
            return final_mappings, missing_fields
            
        except Exception as e:
            logger.error(f"Confidence threshold application failed: {e}")
            raise
    
    async def _validate_input_files(
        self, 
        referral_path: Union[str, Path], 
        pa_form_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Validate input PDF files exist and are readable."""
        try:
            referral_path = Path(referral_path)
            pa_form_path = Path(pa_form_path)
            
            if not referral_path.exists():
                return {"valid": False, "error": f"Referral file not found: {referral_path}"}
            
            if not pa_form_path.exists():
                return {"valid": False, "error": f"PA form file not found: {pa_form_path}"}
            
            # Basic file size checks
            if referral_path.stat().st_size == 0:
                return {"valid": False, "error": "Referral file is empty"}
            
            if pa_form_path.stat().st_size == 0:
                return {"valid": False, "error": "PA form file is empty"}
            
            return {
                "valid": True,
                "referral_size": referral_path.stat().st_size,
                "pa_form_size": pa_form_path.stat().st_size
            }
            
        except Exception as e:
            return {"valid": False, "error": f"File validation error: {e}"}
    
    def _validate_extraction_quality(self, extraction_result: Dict[str, Any]) -> bool:
        """Validate quality of extraction results."""
        try:
            # Check for successful extraction
            if not extraction_result.get("successful_extraction", False):
                return False
            
            # Check confidence score
            overall_confidence = extraction_result.get("overall_confidence", 0.0)
            if overall_confidence < 0.6:  # Minimum 60% confidence
                return False
            
            # Check for extracted content
            patient_info = extraction_result.get("patient_info", {})
            clinical_data = extraction_result.get("clinical_data", {})
            
            total_fields = len(patient_info) + len(clinical_data)
            if total_fields < self.quality_thresholds["min_extracted_fields"]:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Quality validation error: {e}")
            return False
    
    async def _structure_extracted_data(
        self,
        extraction_result: Dict[str, Any],
        session_id: str,
        extraction_method: str
    ) -> ExtractedData:
        """Structure raw extraction results into ExtractedData model."""
        try:
            # Create structured extracted data
            extracted_data = ExtractedData(
                session_id=session_id,
                patient_info=extraction_result.get("patient_info", {}),
                clinical_data=extraction_result.get("clinical_data", {}),
                raw_extracted_pages=extraction_result.get("raw_extracted_pages", {}),
                extraction_summary={
                    "extraction_method": extraction_method,
                    "overall_confidence": extraction_result.get("overall_confidence", 0.0),
                    "pages_processed": extraction_result.get("pages_processed", 1),
                    "text_length": extraction_result.get("text_length", 0)
                },
                total_pages_processed=extraction_result.get("pages_processed", 1),
                extraction_timestamp=datetime.now(timezone.utc)
            )
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Failed to structure extracted data: {e}")
            raise
    
    def _create_pa_form_field(self, field_id: str, field_data: Dict[str, Any]) -> PAFormField:
        """Create PAFormField object from detected field data."""
        try:
            return PAFormField(
                field_name=field_data.get("field_name", field_id),
                field_type=field_data.get("field_type", "text"),
                display_label=field_data.get("display_name", field_id),
                required=field_data.get("required", False),
                coordinates=field_data.get("coordinates", {}),
                validation_rules=field_data.get("validation_rules", {})
            )
        except Exception as e:
            logger.error(f"Failed to create PA form field {field_id}: {e}")
            raise
    
    def _prepare_referral_data_for_mapping(self, referral_data: ExtractedData) -> Dict[str, Any]:
        """Prepare referral data for OpenAI mapping."""
        return {
            "patient_info": referral_data.patient_info,
            "clinical_data": referral_data.clinical_data,
            "extraction_summary": referral_data.extraction_summary
        }
    
    def _prepare_pa_fields_for_mapping(self, pa_fields: Dict[str, PAFormField]) -> Dict[str, Any]:
        """Prepare PA fields for OpenAI mapping."""
        prepared_fields = {}
        for field_id, pa_field in pa_fields.items():
            prepared_fields[field_id] = {
                "name": pa_field.field_name,
                "type": pa_field.field_type,
                "required": pa_field.required,
                "display_label": pa_field.display_label
            }
        return prepared_fields
    
    async def _enhance_field_mappings(
        self,
        mapping_result: Dict[str, Any],
        referral_data: ExtractedData,
        pa_fields: Dict[str, PAFormField]
    ) -> Dict[str, Any]:
        """Enhance field mappings with additional validation and normalization."""
        try:
            enhanced_mappings = mapping_result.copy()
            
            # Apply field mapper utilities for additional validation
            for field_name, mapping_data in mapping_result.get("field_mappings", {}).items():
                mapped_value = mapping_data.get("mapped_value", "")
                
                # Apply field-specific normalization
                if pa_fields.get(field_name, {}).field_type == "date":
                    normalized_value, confidence = self.field_mapper.normalize_date_format(mapped_value)
                    if confidence > mapping_data.get("confidence", 0.0):
                        enhanced_mappings["field_mappings"][field_name]["mapped_value"] = normalized_value
                        enhanced_mappings["field_mappings"][field_name]["confidence"] = confidence
            
            return enhanced_mappings
            
        except Exception as e:
            logger.error(f"Failed to enhance field mappings: {e}")
            return mapping_result
    
    def _validate_mapping_quality(
        self, 
        mappings: Dict[str, Any], 
        pa_fields: Dict[str, PAFormField]
    ) -> float:
        """Calculate mapping success rate."""
        try:
            total_fields = len(pa_fields)
            mapped_fields = len(mappings.get("field_mappings", {}))
            
            if total_fields == 0:
                return 0.0
            
            return mapped_fields / total_fields
            
        except Exception as e:
            logger.error(f"Mapping quality validation error: {e}")
            return 0.0
    
    async def _create_processing_result(
        self,
        session_id: str,
        referral_data: ExtractedData,
        pa_fields: Dict[str, PAFormField],
        field_mappings: Dict[str, Any],
        missing_fields: List[MissingField],
        start_time: datetime
    ) -> ProcessingResult:
        """Create comprehensive processing result."""
        try:
            completion_time = datetime.now(timezone.utc)
            duration = (completion_time - start_time).total_seconds()
            
            # Calculate summary statistics
            total_fields = len(pa_fields)
            mapped_fields = len(field_mappings)
            missing_count = len(missing_fields)
            completion_rate = mapped_fields / total_fields if total_fields > 0 else 0.0
            
            processing_summary = {
                "total_pa_fields": total_fields,
                "successfully_mapped": mapped_fields,
                "missing_fields_count": missing_count,
                "completion_rate": completion_rate,
                "processing_duration_seconds": duration,
                "extraction_method": referral_data.extraction_summary.get("extraction_method", "unknown"),
                "overall_extraction_confidence": referral_data.extraction_summary.get("overall_confidence", 0.0)
            }
            
            return ProcessingResult(
                session_id=session_id,
                processing_status=ProcessingStatusEnum.COMPLETED,
                extracted_data=referral_data,
                pa_form_fields=pa_fields,
                missing_fields=missing_fields,
                processing_summary=processing_summary,
                started_at=start_time,
                completed_at=completion_time,
                processing_duration=duration
            )
            
        except Exception as e:
            logger.error(f"Failed to create processing result: {e}")
            raise
    
    async def _create_failed_result(
        self,
        session_id: str,
        error_message: str,
        start_time: datetime
    ) -> ProcessingResult:
        """Create processing result for failed processing."""
        try:
            completion_time = datetime.now(timezone.utc)
            duration = (completion_time - start_time).total_seconds()
            
            return ProcessingResult(
                session_id=session_id,
                processing_status=ProcessingStatusEnum.FAILED,
                extracted_data=ExtractedData(
                    session_id=session_id,
                    patient_info={},
                    clinical_data={},
                    total_pages_processed=0,
                    extraction_timestamp=start_time
                ),
                pa_form_fields={},
                missing_fields=[],
                processing_summary={
                    "error": error_message,
                    "processing_duration_seconds": duration,
                    "failed_at_stage": "unknown"
                },
                started_at=start_time,
                completed_at=completion_time,
                processing_duration=duration
            )
            
        except Exception as e:
            logger.error(f"Failed to create failed result: {e}")
            raise


# Global processing pipeline instance
processing_pipeline = ProcessingPipeline()


def get_processing_pipeline() -> ProcessingPipeline:
    """
    Get the global processing pipeline instance.
    
    Returns:
        ProcessingPipeline instance for dependency injection
    """
    return processing_pipeline