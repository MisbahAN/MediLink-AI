# backend/app/services/widget_detector.py
"""
Widget Detection Service for PDF form field detection and analysis.

This service uses pdfforms to detect interactive form fields (widgets) in PA forms,
extract their properties, and create field templates for mapping and filling operations.
Optimized for medical form processing with comprehensive field analysis.
"""

import logging
import re
import io
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone
from enum import Enum

try:
    import pdfforms
except ImportError:
    pdfforms = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from app.models.schemas import PAFormField, FieldType
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FieldDetectionMethod(str, Enum):
    """Methods for detecting form fields."""
    PDFFORMS_WIDGETS = "pdfforms_widgets"
    PDFPLUMBER_ANNOTATIONS = "pdfplumber_annotations"
    COORDINATE_ANALYSIS = "coordinate_analysis"
    HYBRID_DETECTION = "hybrid_detection"


class FieldCategory(str, Enum):
    """Categories of detected form fields."""
    PATIENT_DEMOGRAPHICS = "patient_demographics"
    INSURANCE_INFO = "insurance_info"
    PROVIDER_INFO = "provider_info"
    CLINICAL_DATA = "clinical_data"
    MEDICATION_INFO = "medication_info"
    ADMINISTRATIVE = "administrative"
    UNKNOWN = "unknown"


class WidgetDetector:
    """
    Advanced widget detection service for medical form processing.
    
    Detects interactive form fields using pdfforms, analyzes field properties,
    and creates comprehensive field templates for PA form completion with
    specialized handling for medical form patterns.
    """
    
    def __init__(self):
        """Initialize the widget detector with medical form patterns."""
        self.detection_methods = [
            FieldDetectionMethod.PDFFORMS_WIDGETS,
            FieldDetectionMethod.PDFPLUMBER_ANNOTATIONS,
            FieldDetectionMethod.COORDINATE_ANALYSIS
        ]
        
        # Field name patterns for medical forms
        self.field_patterns = {
            "patient_name": [
                r"patient.*name", r"pt.*name", r"first.*name", r"last.*name",
                r"name.*patient", r"full.*name"
            ],
            "date_of_birth": [
                r"date.*birth", r"dob", r"birth.*date", r"birthday",
                r"patient.*dob", r"pt.*dob"
            ],
            "member_id": [
                r"member.*id", r"insurance.*id", r"policy.*number", r"subscriber.*id",
                r"plan.*id", r"member.*number"
            ],
            "phone": [
                r"phone", r"telephone", r"contact.*number", r"mobile",
                r"home.*phone", r"work.*phone"
            ],
            "address": [
                r"address", r"street", r"city", r"state", r"zip.*code",
                r"postal.*code", r"mailing.*address"
            ],
            "provider_name": [
                r"provider.*name", r"doctor.*name", r"physician.*name",
                r"prescriber.*name", r"md.*name", r"np.*name"
            ],
            "npi_number": [
                r"npi", r"national.*provider", r"provider.*id",
                r"license.*number", r"tax.*id"
            ],
            "diagnosis": [
                r"diagnosis", r"condition", r"icd.*code", r"primary.*diagnosis",
                r"medical.*condition", r"dx"
            ],
            "medication": [
                r"medication", r"drug.*name", r"prescription", r"treatment",
                r"med.*name", r"product.*name"
            ]
        }
        
        # Field type mappings
        self.widget_type_mapping = {
            "/Tx": FieldType.TEXT,           # Text field
            "/Ch": FieldType.DROPDOWN,       # Choice field (dropdown/listbox)
            "/Btn": FieldType.CHECKBOX,      # Button field (checkbox/radio)
            "/Sig": FieldType.SIGNATURE,     # Signature field
            "Text": FieldType.TEXT,
            "Choice": FieldType.DROPDOWN,
            "Button": FieldType.CHECKBOX,
            "Signature": FieldType.SIGNATURE
        }
        
        # Required field indicators
        self.required_indicators = [
            "required", "mandatory", "must", "necessary",
            "*", "(required)", "[required]"
        ]
        
        logger.info("Widget detector initialized with medical form patterns")
    
    def detect_form_fields(
        self,
        pdf_path: Union[str, Path],
        detection_method: FieldDetectionMethod = FieldDetectionMethod.HYBRID_DETECTION
    ) -> Dict[str, Any]:
        """
        Detect form fields in a PA form PDF using specified method.
        
        Args:
            pdf_path: Path to the PA form PDF file
            detection_method: Method to use for field detection
            
        Returns:
            Dictionary containing detected fields and metadata
        """
        if not pdfforms:
            logger.error("pdfforms package not available")
            raise RuntimeError("pdfforms package required for widget detection")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Detecting form fields in {pdf_path.name} using {detection_method.value}")
        
        try:
            if detection_method == FieldDetectionMethod.HYBRID_DETECTION:
                fields = self._hybrid_field_detection(pdf_path)
            elif detection_method == FieldDetectionMethod.PDFFORMS_WIDGETS:
                fields = self._detect_with_pdfforms(pdf_path)
            elif detection_method == FieldDetectionMethod.PDFPLUMBER_ANNOTATIONS:
                fields = self._detect_with_pdfplumber(pdf_path)
            else:
                fields = self._coordinate_based_detection(pdf_path)
            
            # Analyze and categorize fields
            analyzed_fields = self._analyze_detected_fields(fields, pdf_path)
            
            # Generate field statistics
            stats = self._generate_field_statistics(analyzed_fields)
            
            result = {
                "pdf_path": str(pdf_path),
                "detection_method": detection_method.value,
                "fields": analyzed_fields,
                "statistics": stats,
                "detection_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_fields": len(analyzed_fields),
                "success": True
            }
            
            logger.info(f"Detected {len(analyzed_fields)} fields in {pdf_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"Field detection failed for {pdf_path.name}: {e}")
            return {
                "pdf_path": str(pdf_path),
                "detection_method": detection_method.value,
                "fields": {},
                "error": str(e),
                "success": False
            }
    
    def extract_field_properties(
        self,
        field_data: Dict[str, Any],
        pdf_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Extract comprehensive properties from a detected form field.
        
        Args:
            field_data: Raw field data from detection
            pdf_path: Path to the source PDF
            
        Returns:
            Dictionary containing extracted field properties
        """
        try:
            # Extract basic properties
            properties = {
                "field_name": self._normalize_field_name(field_data.get("name", "")),
                "display_name": field_data.get("name", ""),
                "field_type": self._determine_field_type(field_data),
                "coordinates": self._extract_coordinates(field_data),
                "page_number": field_data.get("page", 1),
                "required": self._determine_if_required(field_data),
                "default_value": field_data.get("value", ""),
                "max_length": field_data.get("maxLength", None),
                "multiline": field_data.get("multiline", False),
                "readonly": field_data.get("readonly", False)
            }
            
            # Extract advanced properties
            properties.update({
                "field_category": self._categorize_field(properties["field_name"]),
                "validation_pattern": self._get_validation_pattern(properties["field_name"]),
                "placeholder_text": self._extract_placeholder_text(field_data),
                "tooltip": field_data.get("tooltip", ""),
                "tab_order": field_data.get("tabOrder", 0)
            })
            
            # Extract choice-specific properties
            if properties["field_type"] in [FieldType.DROPDOWN, FieldType.RADIO]:
                properties["options"] = field_data.get("options", [])
                properties["allow_custom"] = field_data.get("allowCustom", False)
            
            # Extract checkbox-specific properties
            if properties["field_type"] == FieldType.CHECKBOX:
                properties["checked_value"] = field_data.get("checkedValue", "Yes")
                properties["unchecked_value"] = field_data.get("uncheckedValue", "No")
            
            return properties
            
        except Exception as e:
            logger.error(f"Failed to extract field properties: {e}")
            return {
                "field_name": field_data.get("name", "unknown"),
                "field_type": FieldType.TEXT,
                "error": str(e)
            }
    
    def create_field_template(
        self,
        detected_fields: Dict[str, Any],
        template_name: str = "pa_form_template"
    ) -> Dict[str, Any]:
        """
        Create a comprehensive field template from detected fields.
        
        Args:
            detected_fields: Dictionary of detected form fields
            template_name: Name for the generated template
            
        Returns:
            Field template with mapping guidelines and validation rules
        """
        template = {
            "template_name": template_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "fields": {},
            "field_groups": {},
            "validation_rules": {},
            "mapping_guidelines": {},
            "statistics": {
                "total_fields": 0,
                "required_fields": 0,
                "optional_fields": 0,
                "field_types": {}
            }
        }
        
        try:
            # Process each detected field
            for field_id, field_data in detected_fields.items():
                # Create field template entry
                field_template = {
                    "id": field_id,
                    "name": field_data.get("field_name", field_id),
                    "display_name": field_data.get("display_name", field_id),
                    "type": field_data.get("field_type", FieldType.TEXT),
                    "category": field_data.get("field_category", FieldCategory.UNKNOWN),
                    "required": field_data.get("required", False),
                    "coordinates": field_data.get("coordinates", {}),
                    "page": field_data.get("page_number", 1),
                    "properties": {
                        "max_length": field_data.get("max_length"),
                        "multiline": field_data.get("multiline", False),
                        "readonly": field_data.get("readonly", False),
                        "validation_pattern": field_data.get("validation_pattern"),
                        "placeholder": field_data.get("placeholder_text", "")
                    }
                }
                
                # Add type-specific properties
                if field_template["type"] in [FieldType.DROPDOWN, FieldType.RADIO]:
                    field_template["options"] = field_data.get("options", [])
                elif field_template["type"] == FieldType.CHECKBOX:
                    field_template["checkbox_values"] = {
                        "checked": field_data.get("checked_value", "Yes"),
                        "unchecked": field_data.get("unchecked_value", "No")
                    }
                
                template["fields"][field_id] = field_template
                
                # Create mapping guidelines
                template["mapping_guidelines"][field_id] = self._create_mapping_guideline(field_data)
                
                # Create validation rules
                if field_data.get("validation_pattern"):
                    template["validation_rules"][field_id] = {
                        "pattern": field_data["validation_pattern"],
                        "required": field_data.get("required", False),
                        "max_length": field_data.get("max_length"),
                        "field_type": field_data.get("field_type")
                    }
            
            # Group fields by category
            template["field_groups"] = self._group_fields_by_category(template["fields"])
            
            # Calculate statistics
            template["statistics"] = self._calculate_template_statistics(template["fields"])
            
            logger.info(f"Created field template '{template_name}' with {len(template['fields'])} fields")
            return template
            
        except Exception as e:
            logger.error(f"Failed to create field template: {e}")
            template["error"] = str(e)
            return template
    
    def _hybrid_field_detection(self, pdf_path: Path) -> Dict[str, Any]:
        """Use multiple detection methods for comprehensive field detection."""
        all_fields = {}
        
        # Try pdfforms first (most reliable for widgets)
        try:
            pdfforms_fields = self._detect_with_pdfforms(pdf_path)
            all_fields.update(pdfforms_fields)
            logger.info(f"pdfforms detected {len(pdfforms_fields)} fields")
        except Exception as e:
            logger.warning(f"pdfforms detection failed: {e}")
        
        # Supplement with pdfplumber annotations
        try:
            pdfplumber_fields = self._detect_with_pdfplumber(pdf_path)
            # Merge fields, preferring pdfforms data
            for field_id, field_data in pdfplumber_fields.items():
                if field_id not in all_fields:
                    all_fields[field_id] = field_data
                else:
                    # Merge additional properties
                    all_fields[field_id].update({
                        k: v for k, v in field_data.items() 
                        if k not in all_fields[field_id] or not all_fields[field_id][k]
                    })
            logger.info(f"pdfplumber added/enhanced {len(pdfplumber_fields)} fields")
        except Exception as e:
            logger.warning(f"pdfplumber detection failed: {e}")
        
        return all_fields
    
    def _detect_with_pdfforms(self, pdf_path: Path) -> Dict[str, Any]:
        """Detect form fields using pdfforms library."""
        fields = {}
        
        try:
            # Get form fields using pdfforms
            form_fields = pdfforms.get_form_fields(str(pdf_path))
            
            for i, field in enumerate(form_fields):
                field_id = field.get("name", f"field_{i}")
                
                # Normalize field data
                normalized_field = {
                    "name": field_id,
                    "type": field.get("type", "Text"),
                    "value": field.get("value", ""),
                    "page": field.get("page", 1),
                    "rect": field.get("rect", [0, 0, 0, 0]),
                    "required": self._check_if_required(field.get("flags", 0)),
                    "readonly": self._check_if_readonly(field.get("flags", 0)),
                    "multiline": self._check_if_multiline(field.get("flags", 0)),
                    "maxLength": field.get("maxLength", None),
                    "options": field.get("options", []),
                    "detection_method": "pdfforms"
                }
                
                fields[field_id] = normalized_field
                
        except Exception as e:
            logger.error(f"pdfforms detection error: {e}")
            
        return fields
    
    def _detect_with_pdfplumber(self, pdf_path: Path) -> Dict[str, Any]:
        """Detect form fields using pdfplumber annotations."""
        fields = {}
        
        if not pdfplumber:
            logger.warning("pdfplumber not available")
            return fields
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    if hasattr(page, 'annots') and page.annots:
                        for i, annot in enumerate(page.annots):
                            field_id = f"page_{page_num}_annot_{i}"
                            
                            # Extract annotation properties
                            annot_data = {
                                "name": annot.get("title", field_id),
                                "type": annot.get("subtype", "Text"),
                                "page": page_num,
                                "rect": annot.get("rect", [0, 0, 0, 0]),
                                "contents": annot.get("contents", ""),
                                "detection_method": "pdfplumber_annotations"
                            }
                            
                            fields[field_id] = annot_data
                            
        except Exception as e:
            logger.error(f"pdfplumber detection error: {e}")
            
        return fields
    
    def _coordinate_based_detection(self, pdf_path: Path) -> Dict[str, Any]:
        """Detect potential form fields using coordinate analysis."""
        fields = {}
        
        # This is a placeholder for advanced coordinate-based detection
        # Would involve analyzing text patterns, whitespace, and layout
        logger.info("Coordinate-based detection not implemented yet")
        
        return fields
    
    def _analyze_detected_fields(self, fields: Dict[str, Any], pdf_path: Path) -> Dict[str, Any]:
        """Analyze and enhance detected fields with additional properties."""
        analyzed_fields = {}
        
        for field_id, field_data in fields.items():
            try:
                analyzed_field = self.extract_field_properties(field_data, pdf_path)
                analyzed_fields[field_id] = analyzed_field
            except Exception as e:
                logger.error(f"Failed to analyze field {field_id}: {e}")
                analyzed_fields[field_id] = field_data
        
        # Enhance fields with OCR-detected labels
        try:
            logger.info("Enhancing fields with OCR-detected labels...")
            analyzed_fields = self.enhance_fields_with_ocr_labels(pdf_path, analyzed_fields)
        except Exception as e:
            logger.warning(f"OCR enhancement failed, continuing with basic analysis: {e}")
        
        return analyzed_fields
    
    def _normalize_field_name(self, field_name: str) -> str:
        """Normalize field names for consistent processing."""
        if not field_name:
            return "unnamed_field"
        
        # Convert to lowercase and replace special characters
        normalized = re.sub(r'[^a-zA-Z0-9_]', '_', field_name.lower())
        normalized = re.sub(r'_{2,}', '_', normalized)
        normalized = normalized.strip('_')
        
        return normalized or "unnamed_field"
    
    def _determine_field_type(self, field_data: Dict[str, Any]) -> FieldType:
        """Determine the field type from field data."""
        field_type = field_data.get("type", "Text")
        
        # Map various type representations to standard FieldType
        if field_type in self.widget_type_mapping:
            return self.widget_type_mapping[field_type]
        
        # Analyze field name for type hints
        field_name = (field_data.get("name") or "").lower()
        
        if any(pattern in field_name for pattern in ["date", "dob", "birth"]):
            return FieldType.DATE
        elif any(pattern in field_name for pattern in ["check", "box", "yes", "no"]):
            return FieldType.CHECKBOX
        elif field_data.get("options"):
            return FieldType.DROPDOWN
        else:
            return FieldType.TEXT
    
    def _extract_coordinates(self, field_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract and normalize field coordinates."""
        rect = field_data.get("rect", [0, 0, 0, 0])
        
        if len(rect) >= 4:
            return {
                "x": float(rect[0]),
                "y": float(rect[1]),
                "width": float(rect[2] - rect[0]) if rect[2] > rect[0] else 0.0,
                "height": float(rect[3] - rect[1]) if rect[3] > rect[1] else 0.0
            }
        
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    
    def _determine_if_required(self, field_data: Dict[str, Any]) -> bool:
        """Determine if a field is required based on various indicators."""
        # Check field properties
        if field_data.get("required", False):
            return True
        
        # Check field name for required indicators
        field_name = (field_data.get("name") or "").lower()
        return any(indicator in field_name for indicator in self.required_indicators)
    
    def _categorize_field(self, field_name: str) -> FieldCategory:
        """Categorize field based on name patterns."""
        field_name_lower = field_name.lower()
        
        for category, patterns in self.field_patterns.items():
            if any(re.search(pattern, field_name_lower) for pattern in patterns):
                if "patient" in category or "date_of_birth" in category:
                    return FieldCategory.PATIENT_DEMOGRAPHICS
                elif "member" in category or "insurance" in category:
                    return FieldCategory.INSURANCE_INFO
                elif "provider" in category or "npi" in category:
                    return FieldCategory.PROVIDER_INFO
                elif "diagnosis" in category:
                    return FieldCategory.CLINICAL_DATA
                elif "medication" in category:
                    return FieldCategory.MEDICATION_INFO
                else:
                    return FieldCategory.ADMINISTRATIVE
        
        return FieldCategory.UNKNOWN
    
    def _get_validation_pattern(self, field_name: str) -> Optional[str]:
        """Get validation pattern for field based on name."""
        field_name_lower = field_name.lower()
        
        if any(pattern in field_name_lower for pattern in ["phone", "telephone"]):
            return r"^\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$"
        elif any(pattern in field_name_lower for pattern in ["date", "dob"]):
            return r"^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}$"
        elif "npi" in field_name_lower:
            return r"^\d{10}$"
        elif "zip" in field_name_lower:
            return r"^\d{5}(-\d{4})?$"
        elif "email" in field_name_lower:
            return r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        
        return None
    
    def _extract_placeholder_text(self, field_data: Dict[str, Any]) -> str:
        """Extract placeholder text from field data."""
        return field_data.get("placeholder", field_data.get("tooltip", ""))
    
    def _check_if_required(self, flags: int) -> bool:
        """Check if field is required based on PDF field flags."""
        # PDF field flag for required: 0x02
        return bool(flags & 2)
    
    def _check_if_readonly(self, flags: int) -> bool:
        """Check if field is readonly based on PDF field flags."""
        # PDF field flag for readonly: 0x01
        return bool(flags & 1)
    
    def _check_if_multiline(self, flags: int) -> bool:
        """Check if field is multiline based on PDF field flags."""
        # PDF field flag for multiline: 0x1000
        return bool(flags & 4096)
    
    def _create_mapping_guideline(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create mapping guidelines for a field."""
        return {
            "suggested_sources": self._get_suggested_sources(field_data),
            "confidence_threshold": 0.7,
            "validation_required": field_data.get("required", False),
            "transformation_notes": self._get_transformation_notes(field_data)
        }
    
    def _get_suggested_sources(self, field_data: Dict[str, Any]) -> List[str]:
        """Get suggested data sources for field mapping."""
        field_name = (field_data.get("field_name") or "").lower()
        
        if "patient" in field_name or "name" in field_name:
            return ["patient_demographics.name", "referral_header.patient_name"]
        elif "dob" in field_name or "birth" in field_name:
            return ["patient_demographics.date_of_birth", "patient_info.dob"]
        elif "member" in field_name or "insurance" in field_name:
            return ["insurance_info.member_id", "coverage_details.policy_number"]
        elif "provider" in field_name:
            return ["provider_info.name", "referring_physician.name"]
        elif "diagnosis" in field_name:
            return ["clinical_data.primary_diagnosis", "assessment.diagnosis"]
        
        return ["manual_entry_required"]
    
    def _get_transformation_notes(self, field_data: Dict[str, Any]) -> str:
        """Get transformation notes for field mapping."""
        field_type = field_data.get("field_type", FieldType.TEXT)
        
        if field_type == FieldType.DATE:
            return "Convert to MM/DD/YYYY format"
        elif field_type == FieldType.CHECKBOX:
            return "Map to Yes/No or checked/unchecked values"
        elif "phone" in (field_data.get("field_name") or "").lower():
            return "Format as (XXX) XXX-XXXX"
        
        return "Use value as-is with validation"
    
    def _group_fields_by_category(self, fields: Dict[str, Any]) -> Dict[str, List[str]]:
        """Group fields by their categories."""
        groups = {}
        
        for field_id, field_data in fields.items():
            category = field_data.get("category", FieldCategory.UNKNOWN)
            if category not in groups:
                groups[category] = []
            groups[category].append(field_id)
        
        return groups
    
    def _calculate_template_statistics(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for the field template."""
        stats = {
            "total_fields": len(fields),
            "required_fields": 0,
            "optional_fields": 0,
            "field_types": {},
            "field_categories": {}
        }
        
        for field_data in fields.values():
            if field_data.get("required", False):
                stats["required_fields"] += 1
            else:
                stats["optional_fields"] += 1
            
            field_type = field_data.get("type", FieldType.TEXT)
            stats["field_types"][field_type] = stats["field_types"].get(field_type, 0) + 1
            
            category = field_data.get("category", FieldCategory.UNKNOWN)
            stats["field_categories"][category] = stats["field_categories"].get(category, 0) + 1
        
        return stats
    
    def enhance_fields_with_ocr_labels(self, pdf_path: Union[str, Path], fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance field detection by reading text labels near form fields using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            fields: Dictionary of detected fields with coordinates
            
        Returns:
            Enhanced fields dictionary with semantic labels
        """
        try:
            import pytesseract
            from PIL import Image
            import fitz  # PyMuPDF
            import numpy as np
        except ImportError as e:
            logger.warning(f"OCR dependencies not available: {e}")
            return fields
        
        try:
            pdf_path = Path(pdf_path)
            enhanced_fields = {}
            
            # Open PDF with PyMuPDF for high-quality rendering
            doc = fitz.open(str(pdf_path))
            
            for field_id, field_data in fields.items():
                enhanced_field = field_data.copy()
                
                # Get field coordinates
                coords = field_data.get("coordinates", {})
                if not coords or not all(k in coords for k in ["x", "y", "width", "height"]):
                    enhanced_fields[field_id] = enhanced_field
                    continue
                
                # Find the page containing this field
                page_num = field_data.get("page", 0)
                if page_num >= len(doc):
                    enhanced_fields[field_id] = enhanced_field
                    continue
                
                page = doc[page_num]
                
                # Expand search area around the field to capture labels
                search_margin = 100  # pixels
                x = max(0, coords["x"] - search_margin)
                y = max(0, coords["y"] - search_margin)
                width = coords["width"] + (2 * search_margin)
                height = coords["height"] + search_margin
                
                # Create clip rectangle for the search area
                clip_rect = fitz.Rect(x, y, x + width, y + height)
                
                # Render this area as image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(img_data))
                
                # Use OCR to extract text from the area
                ocr_text = pytesseract.image_to_string(image, config='--psm 6')
                ocr_text = ocr_text.strip()
                
                if ocr_text:
                    # Clean and process the OCR text
                    cleaned_text = self._clean_ocr_text(ocr_text)
                    semantic_label = self._extract_semantic_label(cleaned_text)
                    
                    if semantic_label:
                        enhanced_field["semantic_label"] = semantic_label
                        enhanced_field["ocr_text"] = cleaned_text
                        enhanced_field["display_label"] = semantic_label
                        
                        # Update field name with semantic meaning
                        enhanced_field["field_name"] = semantic_label.lower().replace(" ", "_")
                        
                        logger.debug(f"Enhanced field {field_id}: {semantic_label}")
                
                enhanced_fields[field_id] = enhanced_field
            
            doc.close()
            
            logger.info(f"Enhanced {len([f for f in enhanced_fields.values() if 'semantic_label' in f])} fields with OCR labels")
            return enhanced_fields
            
        except Exception as e:
            logger.error(f"OCR enhancement failed: {e}")
            return fields
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR text by removing noise and normalizing."""
        import re
        
        # Remove extra whitespace and special characters
        cleaned = re.sub(r'\s+', ' ', text)
        cleaned = re.sub(r'[^\w\s\-\(\):]', '', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _extract_semantic_label(self, text: str) -> str:
        """Extract meaningful semantic label from OCR text."""
        text_lower = text.lower()
        
        # Common field label patterns in medical forms
        label_patterns = {
            "patient_name": ["patient name", "name", "patient", "full name"],
            "date_of_birth": ["date of birth", "dob", "birth date", "birthday"],
            "member_id": ["member id", "insurance id", "policy number", "subscriber id"],
            "phone_number": ["phone", "telephone", "phone number", "contact"],
            "address": ["address", "street address", "mailing address"],
            "provider_name": ["provider", "doctor", "physician", "prescriber"],
            "diagnosis": ["diagnosis", "condition", "medical condition"],
            "medication": ["medication", "drug", "prescription"],
            "insurance": ["insurance", "plan", "coverage"],
            "authorization": ["authorization", "prior auth", "pa number"]
        }
        
        # Find the best matching label
        best_match = None
        best_score = 0
        
        for semantic_name, patterns in label_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    score = len(pattern) / len(text_lower)  # Preference for exact matches
                    if score > best_score:
                        best_score = score
                        best_match = semantic_name
        
        if best_match:
            return best_match.replace("_", " ").title()
        
        # If no pattern matches, return cleaned text if it looks like a label
        if len(text) < 50 and not any(char.isdigit() for char in text):
            return text.title()
        
        return None
    
    def _generate_field_statistics(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistics for detected fields."""
        return self._calculate_template_statistics(fields)


# Global widget detector instance
widget_detector = WidgetDetector()


def get_widget_detector() -> WidgetDetector:
    """
    Get the global widget detector instance.
    
    Returns:
        WidgetDetector instance for dependency injection
    """
    return widget_detector