"""
Form Filling Service for populating PA forms with mapped data.

This service handles the actual PDF form filling process using pdfforms,
validates field values, applies formatting transformations, and generates
filled PDF files with comprehensive error handling and validation.
"""

import logging
import re
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone
from enum import Enum

try:
    import pdfforms
except ImportError:
    pdfforms = None

from models.schemas import PAFormField, FieldType, FieldMapping, ConfidenceLevel
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FillingStatus(str, Enum):
    """Status of form filling operations."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    VALIDATION_ERROR = "validation_error"


class FieldValidationResult(str, Enum):
    """Result of field validation."""
    VALID = "valid"
    INVALID_FORMAT = "invalid_format"
    INVALID_LENGTH = "invalid_length"
    INVALID_VALUE = "invalid_value"
    REQUIRED_MISSING = "required_missing"


class FormFiller:
    """
    Advanced form filling service for PA form completion.
    
    Handles PDF form population using pdfforms with comprehensive validation,
    field formatting, error handling, and quality assurance for medical
    form completion workflows.
    """
    
    def __init__(self):
        """Initialize the form filler with validation patterns and rules."""
        # Field validation patterns
        self.validation_patterns = {
            "phone": r'^\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$',
            "date": r'^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}$',
            "npi": r'^\d{10}$',
            "zip_code": r'^\d{5}(-\d{4})?$',
            "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            "ssn": r'^\d{3}-?\d{2}-?\d{4}$',
            "member_id": r'^[A-Za-z0-9\-]{4,20}$'
        }
        
        # Field length limits
        self.field_length_limits = {
            "text": 500,
            "name": 100,
            "address": 200,
            "phone": 15,
            "email": 100,
            "npi": 10,
            "member_id": 20,
            "diagnosis": 300,
            "medication": 200
        }
        
        # Field formatting rules
        self.formatting_rules = {
            "phone": self._format_phone_number,
            "date": self._format_date,
            "name": self._format_name,
            "npi": self._format_npi,
            "zip_code": self._format_zip_code,
            "member_id": self._format_member_id
        }
        
        # Required field patterns
        self.required_field_patterns = [
            "patient_name", "patient.*name", "first.*name", "last.*name",
            "date.*birth", "dob", "member.*id", "insurance.*id",
            "prescriber.*name", "provider.*name", "diagnosis", "medication"
        ]
        
        logger.info("Form filler initialized with comprehensive validation rules")
    
    async def fill_widget_form(
        self,
        pa_form_path: Union[str, Path],
        field_mappings: Dict[str, FieldMapping],
        output_path: Union[str, Path],
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Fill PA form widgets with mapped field data.
        
        Args:
            pa_form_path: Path to the original PA form PDF
            field_mappings: Dictionary of field mappings to apply
            output_path: Path where filled PDF should be saved
            session_id: Optional session identifier for logging
            
        Returns:
            Dictionary containing filling results and statistics
        """
        if not pdfforms:
            raise RuntimeError("pdfforms package not available for form filling")
        
        pa_form_path = Path(pa_form_path)
        output_path = Path(output_path)
        
        if not pa_form_path.exists():
            raise FileNotFoundError(f"PA form file not found: {pa_form_path}")
        
        logger.info(f"Starting form filling for {pa_form_path.name} -> {output_path.name}")
        
        try:
            # Prepare field data for pdfforms
            field_data = await self._prepare_field_data(field_mappings)
            
            # Validate all field values before filling
            validation_results = await self._validate_all_fields(field_data, field_mappings)
            
            # Apply field formatting
            formatted_data = await self._apply_field_formatting(field_data, field_mappings)
            
            # Fill the PDF form
            filling_result = await self._execute_form_filling(
                pa_form_path, formatted_data, output_path
            )
            
            # Verify the filled form
            verification_result = await self._verify_filled_form(
                output_path, formatted_data, field_mappings
            )
            
            # Compile comprehensive results
            results = {
                "session_id": session_id,
                "source_form": str(pa_form_path),
                "output_file": str(output_path),
                "filling_status": filling_result["status"],
                "fields_processed": len(field_data),
                "fields_filled": filling_result["fields_filled"],
                "fields_failed": filling_result["fields_failed"],
                "validation_results": validation_results,
                "formatting_applied": len([f for f in formatted_data.values() if f.get("formatted")]),
                "verification_status": verification_result["status"],
                "processing_time": filling_result["processing_time"],
                "file_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
                "completion_timestamp": datetime.now(timezone.utc).isoformat(),
                "errors": filling_result.get("errors", []),
                "warnings": filling_result.get("warnings", [])
            }
            
            logger.info(f"Form filling completed: {results['fields_filled']}/{results['fields_processed']} fields filled")
            return results
            
        except Exception as e:
            logger.error(f"Form filling failed for {pa_form_path.name}: {e}")
            raise
    
    async def validate_field_value(
        self,
        field_name: str,
        field_value: str,
        field_type: FieldType,
        field_mapping: Optional[FieldMapping] = None
    ) -> Tuple[FieldValidationResult, str, Optional[str]]:
        """
        Validate a single field value against its requirements.
        
        Args:
            field_name: Name of the field being validated
            field_value: Value to validate
            field_type: Type of the field
            field_mapping: Optional field mapping with additional validation info
            
        Returns:
            Tuple of (validation_result, validated_value, error_message)
        """
        try:
            if not field_value or field_value.strip() == "":
                if self._is_required_field(field_name):
                    return FieldValidationResult.REQUIRED_MISSING, "", "Required field is empty"
                return FieldValidationResult.VALID, "", None
            
            # Clean the value
            cleaned_value = field_value.strip()
            
            # Length validation
            length_limit = self._get_field_length_limit(field_name, field_type)
            if len(cleaned_value) > length_limit:
                return (
                    FieldValidationResult.INVALID_LENGTH,
                    cleaned_value[:length_limit],
                    f"Value truncated to {length_limit} characters"
                )
            
            # Pattern validation
            validation_pattern = self._get_validation_pattern(field_name)
            if validation_pattern and not re.match(validation_pattern, cleaned_value):
                return (
                    FieldValidationResult.INVALID_FORMAT,
                    cleaned_value,
                    f"Value does not match expected format for {field_name}"
                )
            
            # Type-specific validation
            type_validation = self._validate_by_type(cleaned_value, field_type)
            if not type_validation["valid"]:
                return (
                    FieldValidationResult.INVALID_VALUE,
                    cleaned_value,
                    type_validation["error"]
                )
            
            # Additional validation from field mapping
            if field_mapping:
                mapping_validation = self._validate_from_mapping(cleaned_value, field_mapping)
                if not mapping_validation["valid"]:
                    return (
                        FieldValidationResult.INVALID_VALUE,
                        cleaned_value,
                        mapping_validation["error"]
                    )
            
            return FieldValidationResult.VALID, cleaned_value, None
            
        except Exception as e:
            logger.error(f"Field validation failed for {field_name}: {e}")
            return FieldValidationResult.INVALID_VALUE, field_value, str(e)
    
    async def save_filled_pdf(
        self,
        source_pdf_path: Union[str, Path],
        field_data: Dict[str, str],
        output_path: Union[str, Path],
        preserve_original: bool = True
    ) -> Dict[str, Any]:
        """
        Save a filled PDF with comprehensive error handling and verification.
        
        Args:
            source_pdf_path: Path to source PDF form
            field_data: Dictionary of field names to values
            output_path: Path for output filled PDF
            preserve_original: Whether to preserve the original file
            
        Returns:
            Dictionary containing save operation results
        """
        source_path = Path(source_pdf_path)
        output_path = Path(output_path)
        
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Backup original if requested
            backup_path = None
            if preserve_original and output_path.exists():
                backup_path = output_path.with_suffix(f".backup_{int(datetime.now().timestamp())}.pdf")
                shutil.copy2(output_path, backup_path)
            
            # Use pdfforms to fill the form
            start_time = datetime.now()
            
            # Create filled PDF
            success = pdfforms.fill_form(
                input_pdf_path=str(source_path),
                output_pdf_path=str(output_path),
                field_data=field_data
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if not success:
                raise RuntimeError("pdfforms fill_form operation failed")
            
            # Verify the output file was created and is valid
            if not output_path.exists():
                raise FileNotFoundError("Filled PDF was not created")
            
            file_size = output_path.stat().st_size
            if file_size == 0:
                raise ValueError("Filled PDF is empty")
            
            return {
                "success": True,
                "output_path": str(output_path),
                "file_size_bytes": file_size,
                "processing_time_seconds": processing_time,
                "backup_created": backup_path is not None,
                "backup_path": str(backup_path) if backup_path else None,
                "fields_filled": len(field_data),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to save filled PDF: {e}")
            
            # Restore backup if save failed and backup exists
            if backup_path and backup_path.exists() and not output_path.exists():
                try:
                    shutil.move(backup_path, output_path)
                    logger.info(f"Restored backup to {output_path}")
                except Exception as restore_error:
                    logger.error(f"Failed to restore backup: {restore_error}")
            
            return {
                "success": False,
                "error": str(e),
                "output_path": str(output_path),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _prepare_field_data(self, field_mappings: Dict[str, FieldMapping]) -> Dict[str, Dict[str, Any]]:
        """Prepare field data from mappings for form filling."""
        field_data = {}
        
        for field_name, mapping in field_mappings.items():
            field_data[field_name] = {
                "value": mapping.mapped_value,
                "original_value": mapping.original_value,
                "confidence": mapping.confidence_score.overall_confidence,
                "requires_review": mapping.requires_review,
                "field_type": getattr(mapping, 'field_type', FieldType.TEXT),
                "validation_status": mapping.validation_status
            }
        
        return field_data
    
    async def _validate_all_fields(
        self,
        field_data: Dict[str, Dict[str, Any]],
        field_mappings: Dict[str, FieldMapping]
    ) -> Dict[str, Dict[str, Any]]:
        """Validate all fields before filling."""
        validation_results = {}
        
        for field_name, data in field_data.items():
            field_mapping = field_mappings.get(field_name)
            field_type = data.get("field_type", FieldType.TEXT)
            
            validation_result, validated_value, error_message = await self.validate_field_value(
                field_name, data["value"], field_type, field_mapping
            )
            
            validation_results[field_name] = {
                "validation_result": validation_result,
                "validated_value": validated_value,
                "error_message": error_message,
                "original_value": data["value"]
            }
        
        return validation_results
    
    async def _apply_field_formatting(
        self,
        field_data: Dict[str, Dict[str, Any]],
        field_mappings: Dict[str, FieldMapping]
    ) -> Dict[str, Dict[str, Any]]:
        """Apply formatting rules to field values."""
        formatted_data = {}
        
        for field_name, data in field_data.items():
            formatted_value = data["value"]
            formatting_applied = False
            
            # Apply formatting based on field name patterns
            for pattern, formatter in self.formatting_rules.items():
                if re.search(pattern, field_name.lower()):
                    try:
                        formatted_value = formatter(formatted_value)
                        formatting_applied = True
                        break
                    except Exception as e:
                        logger.warning(f"Formatting failed for {field_name}: {e}")
            
            formatted_data[field_name] = {
                **data,
                "formatted_value": formatted_value,
                "formatted": formatting_applied
            }
        
        return formatted_data
    
    async def _execute_form_filling(
        self,
        source_path: Path,
        field_data: Dict[str, Dict[str, Any]],
        output_path: Path
    ) -> Dict[str, Any]:
        """Execute the actual form filling operation."""
        start_time = datetime.now()
        
        try:
            # Prepare simple field dictionary for pdfforms
            simple_fields = {
                name: data["formatted_value"] if data.get("formatted") else data["value"]
                for name, data in field_data.items()
                if data["value"]  # Only include non-empty values
            }
            
            # Execute form filling
            success = pdfforms.fill_form(
                input_pdf_path=str(source_path),
                output_pdf_path=str(output_path),
                field_data=simple_fields
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if not success:
                return {
                    "status": FillingStatus.FAILED,
                    "fields_filled": 0,
                    "fields_failed": len(field_data),
                    "processing_time": processing_time,
                    "errors": ["pdfforms fill_form operation failed"]
                }
            
            # Count successful fills (assume all non-empty fields were filled)
            fields_filled = len(simple_fields)
            fields_failed = len(field_data) - fields_filled
            
            return {
                "status": FillingStatus.SUCCESS if fields_failed == 0 else FillingStatus.PARTIAL,
                "fields_filled": fields_filled,
                "fields_failed": fields_failed,
                "processing_time": processing_time,
                "errors": [],
                "warnings": [] if fields_failed == 0 else [f"{fields_failed} fields were empty and not filled"]
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return {
                "status": FillingStatus.FAILED,
                "fields_filled": 0,
                "fields_failed": len(field_data),
                "processing_time": processing_time,
                "errors": [str(e)]
            }
    
    async def _verify_filled_form(
        self,
        output_path: Path,
        field_data: Dict[str, Dict[str, Any]],
        field_mappings: Dict[str, FieldMapping]
    ) -> Dict[str, Any]:
        """Verify the filled form was created successfully."""
        try:
            if not output_path.exists():
                return {"status": "failed", "error": "Output file does not exist"}
            
            file_size = output_path.stat().st_size
            if file_size == 0:
                return {"status": "failed", "error": "Output file is empty"}
            
            # Basic file verification - could be extended with actual PDF validation
            return {
                "status": "success",
                "file_size": file_size,
                "file_exists": True
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def _is_required_field(self, field_name: str) -> bool:
        """Check if a field is required based on name patterns."""
        field_lower = field_name.lower()
        return any(re.search(pattern, field_lower) for pattern in self.required_field_patterns)
    
    def _get_field_length_limit(self, field_name: str, field_type: FieldType) -> int:
        """Get length limit for a field."""
        field_lower = field_name.lower()
        
        # Check specific field patterns
        for pattern, limit in self.field_length_limits.items():
            if pattern in field_lower:
                return limit
        
        # Default by field type
        if field_type == FieldType.TEXT:
            return self.field_length_limits["text"]
        else:
            return 100  # Default limit
    
    def _get_validation_pattern(self, field_name: str) -> Optional[str]:
        """Get validation pattern for a field."""
        field_lower = field_name.lower()
        
        for pattern_name, pattern in self.validation_patterns.items():
            if pattern_name in field_lower:
                return pattern
        
        return None
    
    def _validate_by_type(self, value: str, field_type: FieldType) -> Dict[str, Any]:
        """Validate value based on field type."""
        try:
            if field_type == FieldType.DATE:
                # Try to parse as date
                from datetime import datetime
                datetime.strptime(value.replace("-", "/"), "%m/%d/%Y")
                return {"valid": True}
            
            elif field_type == FieldType.CHECKBOX:
                # Checkbox values should be Yes/No, True/False, or similar
                valid_values = ["yes", "no", "true", "false", "1", "0", "x", ""]
                return {"valid": value.lower() in valid_values}
            
            else:
                # Text fields - basic character validation
                if re.search(r'[^\w\s\-\.\,\(\)\@\/]', value):
                    return {"valid": False, "error": "Contains invalid characters"}
                return {"valid": True}
                
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def _validate_from_mapping(self, value: str, field_mapping: FieldMapping) -> Dict[str, Any]:
        """Additional validation based on field mapping metadata."""
        try:
            # Check confidence threshold
            if field_mapping.confidence_score.overall_confidence < 0.5:
                return {
                    "valid": False,
                    "error": f"Low confidence mapping ({field_mapping.confidence_score.overall_confidence:.2f})"
                }
            
            # Check if marked for review
            if field_mapping.requires_review:
                return {
                    "valid": False,
                    "error": "Field marked for manual review"
                }
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def _format_phone_number(self, phone: str) -> str:
        """Format phone number to (XXX) XXX-XXXX."""
        digits = re.sub(r'\D', '', phone)
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits.startswith('1'):
            return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        return phone
    
    def _format_date(self, date: str) -> str:
        """Format date to MM/DD/YYYY."""
        # Remove extra characters and normalize
        date_clean = re.sub(r'[^\d\/\-]', '', date)
        
        # Handle different formats
        if re.match(r'\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}', date_clean):
            # YYYY/MM/DD to MM/DD/YYYY
            parts = re.split(r'[\/\-]', date_clean)
            return f"{parts[1].zfill(2)}/{parts[2].zfill(2)}/{parts[0]}"
        elif re.match(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}', date_clean):
            # MM/DD/YY or MM/DD/YYYY
            parts = re.split(r'[\/\-]', date_clean)
            year = parts[2]
            if len(year) == 2:
                year = f"20{year}" if int(year) < 50 else f"19{year}"
            return f"{parts[0].zfill(2)}/{parts[1].zfill(2)}/{year}"
        
        return date
    
    def _format_name(self, name: str) -> str:
        """Format name to proper case."""
        return ' '.join(word.capitalize() for word in name.split())
    
    def _format_npi(self, npi: str) -> str:
        """Format NPI number (remove spaces/hyphens)."""
        return re.sub(r'[\s\-]', '', npi)
    
    def _format_zip_code(self, zip_code: str) -> str:
        """Format ZIP code."""
        digits = re.sub(r'\D', '', zip_code)
        if len(digits) == 9:
            return f"{digits[:5]}-{digits[5:]}"
        return digits[:5]
    
    def _format_member_id(self, member_id: str) -> str:
        """Format member ID (uppercase, remove extra spaces)."""
        return re.sub(r'\s+', '', member_id.upper())


# Global form filler instance
form_filler = FormFiller()


def get_form_filler() -> FormFiller:
    """
    Get the global form filler instance.
    
    Returns:
        FormFiller instance for dependency injection
    """
    return form_filler