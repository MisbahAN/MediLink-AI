"""
Field Mapping Engine for Prior Authorization document processing.

This service provides specialized utilities for mapping, normalizing, and validating
medical data fields between referral packets and PA forms with high accuracy
and confidence scoring.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone
from difflib import SequenceMatcher
import unicodedata
from enum import Enum

from app.models.schemas import ExtractedField, ConfidenceLevel, PatientInfo, ClinicalData

logger = logging.getLogger(__name__)


class FieldCategory(str, Enum):
    """Categories of medical form fields for specialized processing."""
    PATIENT_DEMOGRAPHICS = "patient_demographics"
    INSURANCE = "insurance"
    PROVIDER = "provider"
    CLINICAL = "clinical"
    MEDICATION = "medication"
    DATES = "dates"
    IDENTIFIERS = "identifiers"
    CONTACT = "contact"


class MatchQuality(str, Enum):
    """Quality levels for field matching operations."""
    EXACT = "exact"          # Perfect match
    HIGH = "high"            # Very close match with minor differences
    MEDIUM = "medium"        # Good match with some normalization needed
    LOW = "low"              # Weak match requiring validation
    NONE = "none"            # No reasonable match found


class FieldMapper:
    """
    Advanced field mapping engine for medical document processing.
    
    Provides intelligent mapping, normalization, and validation of medical data
    fields with confidence scoring and specialized handling for different
    field categories.
    """
    
    def __init__(self):
        """Initialize the field mapping engine with patterns and rules."""
        self.confidence_weights = {
            "exact_match": 1.0,
            "fuzzy_match": 0.85,
            "pattern_match": 0.75,
            "inferred_match": 0.60,
            "partial_match": 0.45
        }
        
        # Common field name variations and aliases
        self.field_aliases = {
            "patient_name": ["name", "patient", "pt_name", "patient_full_name", "full_name"],
            "date_of_birth": ["dob", "birth_date", "birthdate", "date_birth", "birthday"],
            "member_id": ["insurance_id", "policy_number", "subscriber_id", "plan_id"],
            "phone_number": ["phone", "telephone", "contact_number", "mobile"],
            "provider_name": ["doctor", "physician", "practitioner", "md", "provider"],
            "npi_number": ["npi", "provider_npi", "national_provider_id"],
            "diagnosis": ["primary_diagnosis", "dx", "condition", "medical_condition"],
            "medication": ["drug", "prescription", "med", "treatment"],
            "address": ["street_address", "mailing_address", "home_address"]
        }
        
        # Field normalization patterns
        self.normalization_patterns = {
            "phone": r'[\(\)\-\s\.]',
            "ssn": r'[\-\s]',
            "date": r'[\-\/\.]',
            "insurance_id": r'[\-\s]',
            "name": r'\s{2,}',
            "npi": r'[\-\s]'
        }
        
        # Validation patterns
        self.validation_patterns = {
            "phone": r'^\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$',
            "ssn": r'^\d{3}-?\d{2}-?\d{4}$',
            "date": r'^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}$',
            "npi": r'^\d{10}$',
            "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            "zip_code": r'^\d{5}(-\d{4})?$'
        }
        
        # Common name prefixes and suffixes
        self.name_prefixes = ["dr", "dr.", "doctor", "prof", "prof.", "mr", "mr.", "ms", "ms.", "mrs", "mrs."]
        self.name_suffixes = ["jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "md", "do", "np", "pa"]
        
        logger.info("Field mapping engine initialized with comprehensive medical field patterns")
    
    def normalize_field_name(self, field_name: str) -> str:
        """
        Normalize field names for consistent mapping.
        
        Args:
            field_name: Raw field name from form or extracted data
            
        Returns:
            Normalized field name in standard format
        """
        if not field_name:
            return ""
        
        # Convert to lowercase and remove special characters
        normalized = field_name.lower().strip()
        normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
        normalized = re.sub(r'\s+', '_', normalized)
        
        # Remove common prefixes and suffixes
        prefixes_to_remove = ["patient_", "form_", "field_", "input_"]
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        
        # Handle common variations
        replacements = {
            "fname": "first_name",
            "lname": "last_name",
            "dob": "date_of_birth",
            "ssn": "social_security_number",
            "ph": "phone",
            "tel": "phone",
            "addr": "address",
            "zip": "zip_code",
            "dx": "diagnosis",
            "rx": "medication"
        }
        
        for old, new in replacements.items():
            if normalized == old:
                normalized = new
                break
        
        return normalized
    
    def match_patient_name(
        self, 
        extracted_names: List[str], 
        target_field: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[str], float, MatchQuality]:
        """
        Match patient names with intelligent handling of different formats.
        
        Args:
            extracted_names: List of potential patient names from referral
            target_field: Target field name requiring patient name
            context: Additional context for disambiguation
            
        Returns:
            Tuple of (best_match, confidence_score, match_quality)
        """
        if not extracted_names:
            return None, 0.0, MatchQuality.NONE
        
        best_match = None
        best_confidence = 0.0
        best_quality = MatchQuality.NONE
        
        for name in extracted_names:
            if not name or len(name.strip()) < 2:
                continue
            
            # Normalize the name
            normalized_name = self._normalize_name(name)
            
            # Calculate confidence based on name characteristics
            confidence = self._calculate_name_confidence(normalized_name, target_field, context)
            quality = self._determine_match_quality(confidence)
            
            if confidence > best_confidence:
                best_match = normalized_name
                best_confidence = confidence
                best_quality = quality
        
        return best_match, best_confidence, best_quality
    
    def normalize_date_format(
        self, 
        date_str: str, 
        target_format: str = "MM/DD/YYYY"
    ) -> Tuple[str, float]:
        """
        Normalize date strings to consistent format with validation.
        
        Args:
            date_str: Input date string in various formats
            target_format: Desired output format
            
        Returns:
            Tuple of (normalized_date, confidence_score)
        """
        if not date_str:
            return "", 0.0
        
        # Clean the input
        cleaned_date = date_str.strip()
        
        # Common date patterns with confidence scores
        date_patterns = [
            (r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})', 0.95),  # MM/DD/YYYY
            (r'(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})', 0.90),  # YYYY/MM/DD
            (r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2})', 0.85),  # MM/DD/YY
            (r'(\w{3,9})\s+(\d{1,2}),?\s+(\d{4})', 0.88),      # Month DD, YYYY
            (r'(\d{1,2})\s+(\w{3,9})\s+(\d{4})', 0.88),       # DD Month YYYY
        ]
        
        best_match = None
        best_confidence = 0.0
        
        for pattern, base_confidence in date_patterns:
            match = re.search(pattern, cleaned_date, re.IGNORECASE)
            if match:
                try:
                    normalized = self._parse_date_match(match, pattern, target_format)
                    if self._validate_date(normalized):
                        confidence = base_confidence * self._calculate_date_quality(normalized)
                        if confidence > best_confidence:
                            best_match = normalized
                            best_confidence = confidence
                except (ValueError, IndexError):
                    continue
        
        return best_match or cleaned_date, best_confidence
    
    def extract_insurance_id(
        self, 
        text_content: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, str]]:
        """
        Extract insurance/member ID numbers from text with confidence scoring.
        
        Args:
            text_content: Text content to search for insurance IDs
            context: Additional context about expected ID format
            
        Returns:
            List of tuples: (insurance_id, confidence, extraction_source)
        """
        if not text_content:
            return []
        
        insurance_ids = []
        
        # Insurance ID patterns with confidence scores
        id_patterns = [
            (r'Member\s+ID\s*:?\s*([A-Za-z0-9\-]{6,20})', 0.95, "member_id_label"),
            (r'Insurance\s+ID\s*:?\s*([A-Za-z0-9\-]{6,20})', 0.90, "insurance_id_label"),
            (r'Policy\s+#?\s*:?\s*([A-Za-z0-9\-]{6,20})', 0.88, "policy_number_label"),
            (r'Subscriber\s+ID\s*:?\s*([A-Za-z0-9\-]{6,20})', 0.85, "subscriber_id_label"),
            (r'Plan\s+ID\s*:?\s*([A-Za-z0-9\-]{6,20})', 0.80, "plan_id_label"),
            (r'\b([A-Z]{2,3}\d{6,12})\b', 0.70, "pattern_match"),  # Common format
            (r'\b(\d{9,12})\b', 0.60, "number_sequence"),  # Numeric sequence
        ]
        
        for pattern, base_confidence, source in id_patterns:
            matches = re.finditer(pattern, text_content, re.IGNORECASE)
            for match in matches:
                id_value = match.group(1).strip()
                
                # Additional validation and confidence adjustment
                confidence = base_confidence * self._calculate_id_quality(id_value, context)
                
                if confidence >= 0.3:  # Minimum threshold
                    insurance_ids.append((id_value, confidence, source))
        
        # Sort by confidence and remove duplicates
        insurance_ids.sort(key=lambda x: x[1], reverse=True)
        unique_ids = []
        seen_ids = set()
        
        for id_value, confidence, source in insurance_ids:
            normalized_id = re.sub(r'[\-\s]', '', id_value.upper())
            if normalized_id not in seen_ids:
                seen_ids.add(normalized_id)
                unique_ids.append((id_value, confidence, source))
        
        return unique_ids[:5]  # Return top 5 candidates
    
    def calculate_confidence_score(
        self, 
        extracted_value: str, 
        target_field: str, 
        field_category: FieldCategory,
        validation_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate comprehensive confidence score for field mapping.
        
        Args:
            extracted_value: The extracted value to score
            target_field: Target field name
            field_category: Category of the field
            validation_context: Additional validation context
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not extracted_value:
            return 0.0
        
        base_confidence = 0.5  # Starting point
        
        # Apply category-specific scoring
        category_score = self._calculate_category_confidence(
            extracted_value, field_category, validation_context
        )
        
        # Apply field-specific validation
        validation_score = self._calculate_validation_confidence(
            extracted_value, target_field
        )
        
        # Apply context-based scoring
        context_score = self._calculate_context_confidence(
            extracted_value, target_field, validation_context
        )
        
        # Quality indicators
        quality_score = self._calculate_quality_indicators(extracted_value)
        
        # Weighted combination
        final_confidence = (
            0.3 * category_score +
            0.3 * validation_score +
            0.2 * context_score +
            0.2 * quality_score
        )
        
        # Ensure bounds
        return max(0.0, min(1.0, final_confidence))
    
    def _normalize_name(self, name: str) -> str:
        """Normalize patient name handling various formats."""
        if not name:
            return ""
        
        # Remove unicode characters and normalize
        normalized = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
        
        # Clean and standardize
        normalized = re.sub(r'[^\w\s,\.]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Handle "Last, First" format
        if ',' in normalized:
            parts = [p.strip() for p in normalized.split(',')]
            if len(parts) == 2:
                last, first = parts
                normalized = f"{first} {last}"
        
        # Remove prefixes and suffixes
        words = normalized.split()
        filtered_words = []
        
        for word in words:
            word_lower = word.lower().rstrip('.')
            if word_lower not in self.name_prefixes and word_lower not in self.name_suffixes:
                filtered_words.append(word.title())
        
        return ' '.join(filtered_words)
    
    def _calculate_name_confidence(
        self, 
        name: str, 
        target_field: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate confidence score for patient name matching."""
        if not name:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Length and structure checks
        words = name.split()
        if len(words) >= 2:  # Has first and last name
            confidence += 0.2
        if len(words) == 2:  # Exactly first and last
            confidence += 0.1
        
        # Character composition
        if re.match(r'^[A-Za-z\s\.\-\']+$', name):  # Only valid name characters
            confidence += 0.2
        
        # No numbers or unusual characters
        if not re.search(r'\d', name):
            confidence += 0.1
        
        # Reasonable length (2-50 characters)
        if 2 <= len(name) <= 50:
            confidence += 0.1
        
        # Context-based adjustments
        if context and 'patient_context' in context:
            if any(word.lower() in name.lower() for word in context['patient_context']):
                confidence += 0.1
        
        return min(1.0, confidence)
    
    def _parse_date_match(self, match, pattern: str, target_format: str) -> str:
        """Parse date match and convert to target format."""
        groups = match.groups()
        
        if 'YYYY' in pattern and len(groups[0]) == 4:  # YYYY/MM/DD format
            year, month, day = groups
        elif 'MM/DD/YYYY' in pattern:  # MM/DD/YYYY format
            month, day, year = groups
        elif 'MM/DD/YY' in pattern:  # MM/DD/YY format
            month, day, year = groups
            year = f"20{year}" if int(year) < 50 else f"19{year}"
        elif 'Month' in pattern:  # Month DD, YYYY or DD Month YYYY
            if len(groups[2]) == 4:  # Month DD, YYYY
                month_name, day, year = groups
                month = str(self._month_name_to_number(month_name))
            else:  # DD Month YYYY
                day, month_name, year = groups
                month = str(self._month_name_to_number(month_name))
        else:
            raise ValueError("Unknown date pattern")
        
        # Ensure two-digit formatting
        month = month.zfill(2)
        day = day.zfill(2)
        
        return f"{month}/{day}/{year}"
    
    def _month_name_to_number(self, month_name: str) -> int:
        """Convert month name to number."""
        months = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
            'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
            'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
            'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
        }
        return months.get(month_name.lower(), 1)
    
    def _validate_date(self, date_str: str) -> bool:
        """Validate if date string represents a valid date."""
        try:
            from datetime import datetime
            datetime.strptime(date_str, "%m/%d/%Y")
            return True
        except ValueError:
            return False
    
    def _calculate_date_quality(self, date_str: str) -> float:
        """Calculate quality score for normalized date."""
        try:
            from datetime import datetime
            date_obj = datetime.strptime(date_str, "%m/%d/%Y")
            current_year = datetime.now().year
            
            # Check if date is reasonable (not in future, not too old)
            if 1900 <= date_obj.year <= current_year:
                return 1.0
            elif date_obj.year > current_year:
                return 0.7  # Future date, possible but suspicious
            else:
                return 0.5  # Very old date, possible but needs verification
        except ValueError:
            return 0.3
    
    def _calculate_id_quality(
        self, 
        id_value: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate quality score for insurance ID."""
        if not id_value:
            return 0.0
        
        score = 0.5
        
        # Length check (most insurance IDs are 6-15 characters)
        if 6 <= len(id_value) <= 15:
            score += 0.2
        
        # Format checks
        if re.match(r'^[A-Za-z0-9\-]+$', id_value):  # Alphanumeric with hyphens
            score += 0.2
        
        # Common patterns
        if re.match(r'^[A-Z]{2,3}\d+', id_value):  # Letters followed by numbers
            score += 0.1
        
        # Context-based validation
        if context and 'expected_format' in context:
            expected = context['expected_format']
            if re.match(expected, id_value):
                score += 0.2
        
        return min(1.0, score)
    
    def _calculate_category_confidence(
        self, 
        value: str, 
        category: FieldCategory, 
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate confidence based on field category."""
        if category == FieldCategory.PATIENT_DEMOGRAPHICS:
            return self._validate_demographic_field(value)
        elif category == FieldCategory.DATES:
            return self._validate_date_field(value)
        elif category == FieldCategory.CONTACT:
            return self._validate_contact_field(value)
        elif category == FieldCategory.IDENTIFIERS:
            return self._validate_identifier_field(value)
        else:
            return 0.7  # Default confidence for other categories
    
    def _calculate_validation_confidence(self, value: str, field_name: str) -> float:
        """Calculate confidence based on field-specific validation."""
        field_lower = field_name.lower()
        
        if 'phone' in field_lower:
            return 1.0 if re.match(self.validation_patterns['phone'], value) else 0.5
        elif 'date' in field_lower or 'dob' in field_lower:
            return 1.0 if re.match(self.validation_patterns['date'], value) else 0.5
        elif 'npi' in field_lower:
            return 1.0 if re.match(self.validation_patterns['npi'], value.replace('-', '')) else 0.3
        elif 'email' in field_lower:
            return 1.0 if re.match(self.validation_patterns['email'], value) else 0.3
        else:
            return 0.7  # Default for fields without specific validation
    
    def _calculate_context_confidence(
        self, 
        value: str, 
        field_name: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate confidence based on context information."""
        if not context:
            return 0.5
        
        score = 0.5
        
        # Check if value appears in expected context
        if 'surrounding_text' in context:
            surrounding = context['surrounding_text'].lower()
            if field_name.lower() in surrounding:
                score += 0.3
        
        # Check consistency with other extracted fields
        if 'related_fields' in context:
            related = context['related_fields']
            # Add logic for cross-field validation
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_quality_indicators(self, value: str) -> float:
        """Calculate quality indicators for extracted value."""
        if not value:
            return 0.0
        
        score = 0.0
        
        # Length appropriateness
        if 1 <= len(value) <= 100:
            score += 0.3
        
        # Character composition
        if not re.search(r'[^\w\s\-\.\,\(\)]', value):  # No unusual special characters
            score += 0.2
        
        # No excessive whitespace or formatting issues
        if not re.search(r'\s{3,}', value):  # No excessive spaces
            score += 0.2
        
        # Capitalization appropriateness
        if value.istitle() or value.isupper() or value.islower():
            score += 0.2
        
        # Not obviously corrupted
        if not re.search(r'(.)\1{4,}', value):  # No excessive repeated characters
            score += 0.1
        
        return min(1.0, score)
    
    def _validate_demographic_field(self, value: str) -> float:
        """Validate demographic field values."""
        if not value or len(value) < 2:
            return 0.2
        
        # Check for reasonable content
        if re.match(r'^[A-Za-z\s\-\.\']+$', value):
            return 0.9
        elif re.match(r'^[A-Za-z0-9\s\-\.\']+$', value):
            return 0.7
        else:
            return 0.4
    
    def _validate_date_field(self, value: str) -> float:
        """Validate date field values."""
        if re.match(self.validation_patterns['date'], value):
            return 0.9
        elif re.search(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}', value):
            return 0.7
        else:
            return 0.3
    
    def _validate_contact_field(self, value: str) -> float:
        """Validate contact field values."""
        if re.match(self.validation_patterns['phone'], value):
            return 0.9
        elif re.match(self.validation_patterns['email'], value):
            return 0.9
        elif re.search(r'\d{3}', value):  # Has some digits (phone-like)
            return 0.6
        else:
            return 0.4
    
    def _validate_identifier_field(self, value: str) -> float:
        """Validate identifier field values."""
        if len(value) >= 6 and re.match(r'^[A-Za-z0-9\-]+$', value):
            return 0.8
        elif len(value) >= 4:
            return 0.6
        else:
            return 0.3
    
    def _determine_match_quality(self, confidence: float) -> MatchQuality:
        """Determine match quality based on confidence score."""
        if confidence >= 0.95:
            return MatchQuality.EXACT
        elif confidence >= 0.80:
            return MatchQuality.HIGH
        elif confidence >= 0.60:
            return MatchQuality.MEDIUM
        elif confidence >= 0.30:
            return MatchQuality.LOW
        else:
            return MatchQuality.NONE


# Global field mapper instance
field_mapper = FieldMapper()


def get_field_mapper() -> FieldMapper:
    """
    Get the global field mapper instance.
    
    Returns:
        FieldMapper instance for dependency injection
    """
    return field_mapper