"""
Pydantic models for API data validation and serialization.

This module defines all data models used for request/response validation,
internal data structures, and API serialization in the Prior Authorization
automation system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class ProcessingStatusEnum(str, Enum):
    """Processing status enumeration for tracking document processing stages."""
    PENDING = "pending"
    UPLOADING = "uploading"
    EXTRACTING = "extracting"
    MAPPING = "mapping"
    FILLING = "filling"
    COMPLETED = "completed"
    FAILED = "failed"


class ConfidenceLevel(str, Enum):
    """Confidence level categories for extracted data."""
    HIGH = "high"      # >90% confidence
    MEDIUM = "medium"  # 70-90% confidence
    LOW = "low"        # <70% confidence


class FieldType(str, Enum):
    """PDF form field types."""
    TEXT = "text"
    DATE = "date"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    DROPDOWN = "dropdown"
    SIGNATURE = "signature"


class UploadResponse(BaseModel):
    """Response model for file upload endpoint."""
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="Upload status message")
    files_received: List[str] = Field(..., description="List of uploaded filenames")
    referral_file: Optional[str] = Field(None, description="Referral packet filename")
    pa_form_file: Optional[str] = Field(None, description="PA form filename")
    upload_timestamp: datetime = Field(..., description="Upload completion time")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "pa_session_123",
                "message": "Files uploaded successfully",
                "files_received": ["referral_packet.pdf", "pa_form.pdf"],
                "referral_file": "referral_packet.pdf",
                "pa_form_file": "pa_form.pdf",
                "upload_timestamp": "2025-06-11T12:00:00Z"
            }
        }


class ProcessingStatus(BaseModel):
    """Model for tracking processing status and progress."""
    session_id: str = Field(..., description="Session identifier")
    status: ProcessingStatusEnum = Field(..., description="Current processing status")
    stage_description: str = Field(..., description="Human-readable stage description")
    progress_percentage: int = Field(..., ge=0, le=100, description="Processing progress (0-100)")
    current_step: str = Field(..., description="Current processing step")
    total_steps: int = Field(..., description="Total number of processing steps")
    estimated_time_remaining: Optional[int] = Field(None, description="Estimated seconds remaining")
    error_message: Optional[str] = Field(None, description="Error message if status is failed")
    last_updated: datetime = Field(..., description="Last status update timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "pa_session_123",
                "status": "extracting",
                "stage_description": "Extracting patient data from referral packet",
                "progress_percentage": 45,
                "current_step": "Processing page 3 of 8",
                "total_steps": 8,
                "estimated_time_remaining": 120,
                "error_message": None,
                "last_updated": "2025-06-11T12:05:30Z"
            }
        }


class ExtractedField(BaseModel):
    """Model for individual extracted data fields with confidence scoring."""
    value: str = Field(..., description="Extracted field value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence category")
    source_page: int = Field(..., ge=1, description="Source page number")
    source_coordinates: Optional[Dict[str, float]] = Field(None, description="Bounding box coordinates")
    extraction_method: str = Field(..., description="Method used for extraction (gemini/mistral/ocr)")
    
    @field_validator('confidence_level', mode='before')
    @classmethod
    def set_confidence_level(cls, v, info):
        """Automatically set confidence level based on confidence score."""
        if hasattr(info, 'data') and 'confidence' in info.data:
            confidence = info.data['confidence']
            if confidence >= 0.9:
                return ConfidenceLevel.HIGH
            elif confidence >= 0.7:
                return ConfidenceLevel.MEDIUM
            else:
                return ConfidenceLevel.LOW
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "value": "John Doe",
                "confidence": 0.95,
                "confidence_level": "high",
                "source_page": 1,
                "source_coordinates": {"x": 150.5, "y": 200.3, "width": 120.0, "height": 18.0},
                "extraction_method": "gemini"
            }
        }


class PatientInfo(BaseModel):
    """Patient demographic information."""
    name: Optional[ExtractedField] = Field(None, description="Patient full name")
    date_of_birth: Optional[ExtractedField] = Field(None, description="Patient date of birth")
    gender: Optional[ExtractedField] = Field(None, description="Patient gender")
    insurance_id: Optional[ExtractedField] = Field(None, description="Insurance member ID")
    group_number: Optional[ExtractedField] = Field(None, description="Insurance group number")
    phone_number: Optional[ExtractedField] = Field(None, description="Patient phone number")
    address: Optional[ExtractedField] = Field(None, description="Patient address")
    ssn: Optional[ExtractedField] = Field(None, description="Patient SSN (last 4 digits)")


class ClinicalData(BaseModel):
    """Clinical information extracted from referral."""
    primary_diagnosis: Optional[ExtractedField] = Field(None, description="Primary diagnosis")
    secondary_diagnoses: List[ExtractedField] = Field(default_factory=list, description="Additional diagnoses")
    treatment_plan: Optional[ExtractedField] = Field(None, description="Proposed treatment plan")
    procedure_codes: List[ExtractedField] = Field(default_factory=list, description="CPT/procedure codes")
    diagnosis_codes: List[ExtractedField] = Field(default_factory=list, description="ICD-10 diagnosis codes")
    provider_name: Optional[ExtractedField] = Field(None, description="Referring provider name")
    provider_npi: Optional[ExtractedField] = Field(None, description="Provider NPI number")
    facility_name: Optional[ExtractedField] = Field(None, description="Healthcare facility name")
    referral_date: Optional[ExtractedField] = Field(None, description="Referral date")


class ExtractedData(BaseModel):
    """Complete extracted data structure from referral packet."""
    session_id: str = Field(..., description="Session identifier")
    patient_info: PatientInfo = Field(..., description="Patient demographic data")
    clinical_data: ClinicalData = Field(..., description="Clinical information")
    raw_extracted_pages: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Raw extracted text and metadata by page"
    )
    extraction_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics and metadata"
    )
    total_pages_processed: int = Field(..., ge=1, description="Number of pages processed")
    extraction_timestamp: datetime = Field(..., description="Extraction completion time")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "pa_session_123",
                "patient_info": {
                    "name": {
                        "value": "John Doe",
                        "confidence": 0.95,
                        "confidence_level": "high",
                        "source_page": 1,
                        "extraction_method": "gemini"
                    }
                },
                "clinical_data": {
                    "primary_diagnosis": {
                        "value": "Type 2 Diabetes Mellitus",
                        "confidence": 0.88,
                        "confidence_level": "medium",
                        "source_page": 3,
                        "extraction_method": "gemini"
                    }
                },
                "raw_extracted_pages": {},
                "extraction_summary": {
                    "high_confidence_fields": 15,
                    "medium_confidence_fields": 8,
                    "low_confidence_fields": 3
                },
                "total_pages_processed": 12,
                "extraction_timestamp": "2025-06-11T12:10:00Z"
            }
        }


class PAFormField(BaseModel):
    """Model for PA form field definitions and mapping."""
    field_name: str = Field(..., description="Form field identifier")
    field_type: FieldType = Field(..., description="Type of form field")
    display_label: str = Field(..., description="Human-readable field label")
    required: bool = Field(..., description="Whether field is required")
    coordinates: Dict[str, float] = Field(..., description="Field position coordinates")
    mapped_value: Optional[str] = Field(None, description="Mapped value from extracted data")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Mapping confidence")
    confidence_level: Optional[ConfidenceLevel] = Field(None, description="Confidence category")
    source_field: Optional[str] = Field(None, description="Source field from extracted data")
    validation_rules: Optional[Dict[str, Any]] = Field(None, description="Field validation rules")
    
    @field_validator('confidence_level', mode='before')
    @classmethod
    def set_confidence_level_from_confidence(cls, v, info):
        """Set confidence level based on confidence score."""
        if v is None and hasattr(info, 'data') and 'confidence' in info.data and info.data['confidence'] is not None:
            confidence = info.data['confidence']
            if confidence >= 0.9:
                return ConfidenceLevel.HIGH
            elif confidence >= 0.7:
                return ConfidenceLevel.MEDIUM
            else:
                return ConfidenceLevel.LOW
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "field_name": "patient_name",
                "field_type": "text",
                "display_label": "Patient Name",
                "required": True,
                "coordinates": {"x": 100.0, "y": 200.0, "width": 200.0, "height": 20.0},
                "mapped_value": "John Doe",
                "confidence": 0.95,
                "confidence_level": "high",
                "source_field": "patient_info.name",
                "validation_rules": {"max_length": 100}
            }
        }


class MissingField(BaseModel):
    """Model for tracking missing or low-confidence fields."""
    field_name: str = Field(..., description="Name of the missing field")
    display_label: str = Field(..., description="Human-readable field label")
    required: bool = Field(..., description="Whether field is required for PA form")
    reason: str = Field(..., description="Reason why field is missing or flagged")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence if extracted but low")
    suggested_value: Optional[str] = Field(None, description="Best guess value if available")
    manual_review_required: bool = Field(..., description="Whether manual review is needed")
    priority: str = Field(..., description="Priority level (high/medium/low)")
    
    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v):
        """Validate priority field values."""
        if v not in ['high', 'medium', 'low']:
            raise ValueError('Priority must be high, medium, or low')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "field_name": "provider_npi",
                "display_label": "Provider NPI Number",
                "required": True,
                "reason": "Not found in referral documents",
                "confidence": None,
                "suggested_value": None,
                "manual_review_required": True,
                "priority": "high"
            }
        }


class ProcessingResult(BaseModel):
    """Complete processing result with all extracted and mapped data."""
    session_id: str = Field(..., description="Session identifier")
    processing_status: ProcessingStatusEnum = Field(..., description="Final processing status")
    extracted_data: ExtractedData = Field(..., description="Raw extracted data")
    pa_form_fields: Dict[str, PAFormField] = Field(..., description="PA form field mappings")
    missing_fields: List[MissingField] = Field(..., description="Missing or low-confidence fields")
    filled_form_path: Optional[str] = Field(None, description="Path to filled PDF form")
    report_path: Optional[str] = Field(None, description="Path to missing fields report")
    processing_summary: Dict[str, Any] = Field(..., description="Processing statistics and metadata")
    started_at: datetime = Field(..., description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    processing_duration: Optional[float] = Field(None, description="Processing duration in seconds")
    
    @field_validator('processing_duration', mode='before')
    @classmethod
    def calculate_duration(cls, v, info):
        """Calculate processing duration if start and end times are available."""
        if v is None and hasattr(info, 'data') and 'started_at' in info.data and 'completed_at' in info.data:
            started = info.data['started_at']
            completed = info.data['completed_at']
            if started and completed:
                return (completed - started).total_seconds()
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "pa_session_123",
                "processing_status": "completed",
                "extracted_data": {},
                "pa_form_fields": {},
                "missing_fields": [],
                "filled_form_path": "/tmp/pa_session_123/filled_form.pdf",
                "report_path": "/tmp/pa_session_123/missing_fields_report.md",
                "processing_summary": {
                    "total_fields_mapped": 45,
                    "high_confidence_mappings": 38,
                    "missing_fields_count": 7,
                    "overall_completion_rate": 0.84
                },
                "started_at": "2025-06-11T12:00:00Z",
                "completed_at": "2025-06-11T12:15:30Z",
                "processing_duration": 930.0
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    session_id: Optional[str] = Field(None, description="Session ID if applicable")
    timestamp: datetime = Field(..., description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ProcessingError",
                "message": "Failed to extract data from PDF",
                "details": {"page": 5, "reason": "OCR confidence too low"},
                "session_id": "pa_session_123",
                "timestamp": "2025-06-11T12:05:00Z"
            }
        }


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(..., description="Dependent service statuses")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-06-11T12:00:00Z",
                "version": "1.0.0",
                "services": {
                    "gemini_api": "connected",
                    "redis": "connected",
                    "storage": "available"
                }
            }
        }


class ConfidenceScore(BaseModel):
    """Model for confidence scoring with detailed breakdown."""
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score (0.0-1.0)")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence category")
    scoring_breakdown: Dict[str, float] = Field(
        default_factory=dict, 
        description="Breakdown of confidence factors"
    )
    extraction_method: str = Field(..., description="Method used for extraction")
    validation_checks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Results of validation checks"
    )
    quality_indicators: Dict[str, Any] = Field(
        default_factory=dict,
        description="Quality metrics and indicators"
    )
    
    @field_validator('confidence_level', mode='before')
    @classmethod
    def set_confidence_level_from_score(cls, v, info):
        """Set confidence level based on overall confidence score."""
        if v is None and hasattr(info, 'data') and 'overall_confidence' in info.data:
            confidence = info.data['overall_confidence']
            if confidence >= 0.9:
                return ConfidenceLevel.HIGH
            elif confidence >= 0.7:
                return ConfidenceLevel.MEDIUM
            else:
                return ConfidenceLevel.LOW
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "overall_confidence": 0.88,
                "confidence_level": "medium",
                "scoring_breakdown": {
                    "text_quality": 0.92,
                    "pattern_match": 0.85,
                    "context_validation": 0.87
                },
                "extraction_method": "mistral_ocr",
                "validation_checks": {
                    "format_valid": True,
                    "length_appropriate": True,
                    "pattern_match": True
                },
                "quality_indicators": {
                    "text_length": 1250,
                    "character_confidence": 0.91,
                    "structure_score": 0.84
                }
            }
        }


class FieldMapping(BaseModel):
    """Model for field mapping between extracted data and PA form fields."""
    source_field: str = Field(..., description="Source field from extracted data")
    target_field: str = Field(..., description="Target field in PA form")
    mapped_value: str = Field(..., description="Mapped value after transformation")
    original_value: Optional[str] = Field(None, description="Original extracted value")
    confidence_score: ConfidenceScore = Field(..., description="Confidence scoring details")
    transformation_applied: Optional[str] = Field(None, description="Transformation method applied")
    validation_status: str = Field(..., description="Validation status (approved/flagged/rejected)")
    mapping_source: str = Field(..., description="Source of mapping (ai/pattern/manual)")
    source_page: Optional[int] = Field(None, ge=1, description="Source page number")
    source_coordinates: Optional[Dict[str, float]] = Field(None, description="Source text coordinates")
    mapping_notes: Optional[str] = Field(None, description="Additional mapping notes")
    requires_review: bool = Field(default=False, description="Whether mapping requires manual review")
    
    @field_validator('validation_status')
    @classmethod
    def validate_status(cls, v):
        """Validate mapping status values."""
        valid_statuses = ['approved', 'flagged', 'rejected', 'pending_review']
        if v not in valid_statuses:
            raise ValueError(f'Validation status must be one of: {", ".join(valid_statuses)}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "source_field": "patient_info.name",
                "target_field": "patient_name",
                "mapped_value": "John Smith",
                "original_value": "Smith, John",
                "confidence_score": {
                    "overall_confidence": 0.95,
                    "confidence_level": "high",
                    "scoring_breakdown": {
                        "exact_match": 0.90,
                        "format_conversion": 0.95,
                        "context_validation": 1.0
                    },
                    "extraction_method": "mistral_ocr",
                    "validation_checks": {
                        "format_valid": True,
                        "length_appropriate": True,
                        "pattern_match": True
                    }
                },
                "transformation_applied": "name_format_conversion",
                "validation_status": "approved",
                "mapping_source": "ai",
                "source_page": 1,
                "source_coordinates": {"x": 150.5, "y": 200.3, "width": 120.0, "height": 18.0},
                "mapping_notes": "Converted from Last, First to First Last format",
                "requires_review": False
            }
        }


class ExtractionResult(BaseModel):
    """Comprehensive model for extraction results with detailed metadata."""
    session_id: str = Field(..., description="Session identifier")
    extraction_id: str = Field(..., description="Unique extraction identifier")
    document_type: str = Field(..., description="Type of document processed")
    document_path: str = Field(..., description="Path to processed document")
    extraction_method: str = Field(..., description="Primary extraction method used")
    fallback_methods: List[str] = Field(default_factory=list, description="Fallback methods attempted")
    
    # Extraction results
    extracted_fields: Dict[str, ExtractedField] = Field(
        default_factory=dict,
        description="All extracted fields with metadata"
    )
    field_mappings: List[FieldMapping] = Field(
        default_factory=list,
        description="Field mappings to target form"
    )
    confidence_summary: ConfidenceScore = Field(..., description="Overall extraction confidence")
    
    # Processing metadata
    pages_processed: int = Field(..., ge=1, description="Number of pages processed")
    processing_time_seconds: float = Field(..., ge=0.0, description="Total processing time")
    extraction_timestamp: datetime = Field(..., description="Extraction completion timestamp")
    
    # Quality metrics
    quality_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Quality assessment metrics"
    )
    validation_results: Dict[str, bool] = Field(
        default_factory=dict,
        description="Validation check results"
    )
    
    # Error and warning information
    errors: List[str] = Field(default_factory=list, description="Errors encountered during extraction")
    warnings: List[str] = Field(default_factory=list, description="Warnings during extraction")
    
    # Statistical summary
    extraction_statistics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extraction statistics and summary"
    )
    
    @field_validator('extraction_statistics', mode='before')
    @classmethod
    def calculate_statistics(cls, v, info):
        """Calculate extraction statistics from extracted fields."""
        if v is None and hasattr(info, 'data') and 'extracted_fields' in info.data:
            extracted_fields = info.data['extracted_fields']
            
            total_fields = len(extracted_fields)
            high_confidence = sum(1 for f in extracted_fields.values() 
                                if f.confidence >= 0.9)
            medium_confidence = sum(1 for f in extracted_fields.values() 
                                  if 0.7 <= f.confidence < 0.9)
            low_confidence = sum(1 for f in extracted_fields.values() 
                               if f.confidence < 0.7)
            
            return {
                "total_fields_extracted": total_fields,
                "high_confidence_fields": high_confidence,
                "medium_confidence_fields": medium_confidence,
                "low_confidence_fields": low_confidence,
                "average_confidence": sum(f.confidence for f in extracted_fields.values()) / total_fields if total_fields > 0 else 0.0,
                "extraction_completeness": (high_confidence + medium_confidence) / total_fields if total_fields > 0 else 0.0
            }
        return v or {}
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "pa_session_123",
                "extraction_id": "ext_456789",
                "document_type": "medical_referral",
                "document_path": "/uploads/session_123/referral_packet.pdf",
                "extraction_method": "mistral_ocr",
                "fallback_methods": [],
                "extracted_fields": {
                    "patient_name": {
                        "value": "John Smith",
                        "confidence": 0.95,
                        "confidence_level": "high",
                        "source_page": 1,
                        "extraction_method": "mistral_ocr"
                    }
                },
                "field_mappings": [
                    {
                        "source_field": "patient_info.name",
                        "target_field": "patient_name",
                        "mapped_value": "John Smith",
                        "validation_status": "approved",
                        "mapping_source": "ai"
                    }
                ],
                "confidence_summary": {
                    "overall_confidence": 0.88,
                    "confidence_level": "medium",
                    "extraction_method": "mistral_ocr"
                },
                "pages_processed": 15,
                "processing_time_seconds": 125.4,
                "extraction_timestamp": "2025-06-11T12:15:30Z",
                "quality_metrics": {
                    "text_clarity": 0.85,
                    "structure_detection": 0.92,
                    "field_recognition": 0.88
                },
                "validation_results": {
                    "format_validation": True,
                    "data_integrity": True,
                    "completeness_check": False
                },
                "errors": [],
                "warnings": ["Low confidence on provider NPI field"],
                "extraction_statistics": {
                    "total_fields_extracted": 25,
                    "high_confidence_fields": 18,
                    "medium_confidence_fields": 5,
                    "low_confidence_fields": 2,
                    "average_confidence": 0.84,
                    "extraction_completeness": 0.92
                }
            }
        }