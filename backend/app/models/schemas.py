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