"""
OpenAI service for intelligent field mapping between extracted referral data and PA form fields.

This service uses GPT-4 to analyze extracted medical data and map it to specific
Prior Authorization form fields with confidence scoring and validation.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone
import asyncio

try:
    import openai
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
    openai = None

from models.schemas import ExtractedField, ConfidenceLevel, PatientInfo, ClinicalData
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class OpenAIService:
    """
    Service for AI-powered field mapping using OpenAI GPT-4.
    
    Analyzes extracted referral data and intelligently maps it to PA form fields
    with confidence scoring, validation, and missing field identification.
    """
    
    def __init__(self):
        """Initialize OpenAI service with API configuration."""
        self.client = None
        self.model_name = settings.OPENAI_MODEL
        self.api_key = settings.OPENAI_API_KEY
        self.max_tokens = settings.MAX_TOKENS
        self.timeout_seconds = settings.PROCESSING_TIMEOUT_SECONDS
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay_base = 2.0
        self.retry_backoff_factor = 2.0
        
        # Confidence thresholds for field mapping
        self.confidence_thresholds = {
            "high": 0.90,       # Direct match or high certainty
            "medium": 0.70,     # Good match with minor uncertainty
            "low": 0.50,        # Possible match requiring verification
            "very_low": 0.30    # Poor match, likely missing or incorrect
        }
        
        # Field mapping priorities
        self.field_priorities = {
            "critical": ["patient_name", "date_of_birth", "member_id", "prescriber_name"],
            "important": ["diagnosis", "medication", "npi_number", "insurance_group"],
            "optional": ["phone", "address", "office_contact", "fax_number"]
        }
        
        logger.info("OpenAI field mapping service initialized")
    
    def initialize_client(self) -> bool:
        """
        Initialize the OpenAI API client.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not AsyncOpenAI:
            logger.error("openai package not installed")
            return False
        
        if not self.api_key:
            logger.error("OPENAI_API_KEY not configured")
            return False
        
        try:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout_seconds
            )
            
            logger.info(f"OpenAI client initialized successfully with model {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return False
    
    def create_field_mapping_prompt(
        self, 
        referral_data: Dict[str, Any], 
        pa_form_fields: Dict[str, Any]
    ) -> str:
        """
        Create a structured prompt for field mapping analysis.
        
        Args:
            referral_data: Extracted data from referral packet
            pa_form_fields: Available fields from PA form
            
        Returns:
            Formatted prompt for GPT-4 analysis
        """
        referral_data_str = json.dumps(referral_data, indent=2)
        pa_form_fields_str = json.dumps(pa_form_fields, indent=2)
        
        prompt = f"""
You are a medical document processing expert tasked with mapping extracted referral data to Prior Authorization (PA) form fields.

EXTRACTED REFERRAL DATA:
{referral_data_str}

PA FORM FIELDS TO FILL:
{pa_form_fields_str}

TASK:
Map the referral data to the appropriate PA form fields. For each PA form field:

1. Identify the best matching data from the referral
2. Assign a confidence score (0.0-1.0) based on:
   - Exact match: 0.95-1.0
   - Close match with formatting differences: 0.80-0.94
   - Inferred/derived match: 0.60-0.79
   - Uncertain/partial match: 0.30-0.59
   - No reasonable match: 0.0-0.29

3. Provide the mapped value with proper formatting
4. Include source reference from referral data
5. Note any transformations applied

FIELD MAPPING RULES:
- Patient names: Handle "Last, First" vs "First Last" formats
- Dates: Convert to MM/DD/YYYY format for PA forms
- Phone numbers: Use (XXX) XXX-XXXX format
- Insurance IDs: Use exact format from referral
- Medications: Include dosage and frequency if available
- Diagnoses: Include ICD codes when present

RESPONSE FORMAT (JSON):
{{
  "field_mappings": {{
    "pa_field_name": {{
      "mapped_value": "extracted and formatted value",
      "confidence": 0.85,
      "source_field": "referral.patient_info.name",
      "transformation": "converted Last, First to First Last",
      "notes": "any relevant observations"
    }}
  }},
  "missing_fields": [
    {{
      "field_name": "provider_npi",
      "priority": "critical|important|optional",
      "reason": "not found in referral data"
    }}
  ],
  "overall_confidence": 0.78,
  "processing_notes": "summary of mapping quality and concerns"
}}

Focus on accuracy over completeness. Mark fields as missing rather than making poor guesses.
"""
        return prompt
    
    async def extract_and_map_fields(
        self, 
        referral_data: Dict[str, Any], 
        pa_form_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract and map fields from referral data to PA form using GPT-4.
        
        Args:
            referral_data: Extracted data from referral packet
            pa_form_fields: PA form field definitions
            
        Returns:
            Dictionary containing mapped fields with confidence scores
        """
        if not self.client and not self.initialize_client():
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            # Create the mapping prompt
            prompt = self.create_field_mapping_prompt(referral_data, pa_form_fields)
            
            # Process with GPT-4 using retry logic
            result = await self._process_mapping_with_retry(prompt)
            
            # Parse and validate the response
            mapping_result = self._parse_mapping_response(result)
            
            # Post-process and enhance the mappings
            enhanced_result = self._enhance_field_mappings(
                mapping_result, referral_data, pa_form_fields
            )
            
            # Add metadata
            enhanced_result.update({
                "mapping_method": "openai_gpt4",
                "model_used": self.model_name,
                "mapping_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_pa_fields": len(pa_form_fields),
                "mapped_fields": len(enhanced_result.get("field_mappings", {})),
                "missing_fields": len(enhanced_result.get("missing_fields", []))
            })
            
            logger.info(
                f"OpenAI field mapping completed: "
                f"mapped={enhanced_result.get('mapped_fields', 0)}, "
                f"missing={enhanced_result.get('missing_fields', 0)}, "
                f"confidence={enhanced_result.get('overall_confidence', 0):.2f}"
            )
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"OpenAI field mapping failed: {e}")
            raise
    
    async def _process_mapping_with_retry(self, prompt: str) -> str:
        """
        Process field mapping with GPT-4 using retry logic.
        
        Args:
            prompt: The mapping prompt to send to GPT-4
            
        Returns:
            GPT-4 response text
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a medical document processing expert specializing in Prior Authorization form completion. Provide accurate, structured responses in JSON format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.1,  # Low temperature for consistent output
                    response_format={"type": "json_object"}
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"OpenAI field mapping failed after {self.max_retries} attempts: {e}"
                    )
                    raise
                
                delay = self.retry_delay_base * (self.retry_backoff_factor ** attempt)
                logger.warning(
                    f"OpenAI mapping attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.1f} seconds..."
                )
                await asyncio.sleep(delay)
    
    def _parse_mapping_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate GPT-4 response for field mappings.
        
        Args:
            response_text: Raw response from GPT-4
            
        Returns:
            Parsed and validated mapping result
        """
        try:
            result = json.loads(response_text)
            
            # Validate required structure
            if "field_mappings" not in result:
                result["field_mappings"] = {}
            
            if "missing_fields" not in result:
                result["missing_fields"] = []
            
            if "overall_confidence" not in result:
                result["overall_confidence"] = 0.0
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            
            # Attempt to extract JSON from response text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Return minimal structure if parsing fails
            return {
                "field_mappings": {},
                "missing_fields": [],
                "overall_confidence": 0.0,
                "processing_notes": f"Failed to parse OpenAI response: {e}"
            }
    
    def _enhance_field_mappings(
        self, 
        mapping_result: Dict[str, Any], 
        referral_data: Dict[str, Any], 
        pa_form_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance and validate field mappings with additional processing.
        
        Args:
            mapping_result: Initial mapping result from GPT-4
            referral_data: Original referral data
            pa_form_fields: PA form field definitions
            
        Returns:
            Enhanced mapping result with improved accuracy
        """
        enhanced_mappings = {}
        field_mappings = mapping_result.get("field_mappings", {})
        
        for field_name, mapping in field_mappings.items():
            enhanced_mapping = self._validate_and_enhance_mapping(
                field_name, mapping, referral_data, pa_form_fields
            )
            enhanced_mappings[field_name] = enhanced_mapping
        
        # Update missing fields with priority classification
        missing_fields = self._classify_missing_fields(
            mapping_result.get("missing_fields", []), pa_form_fields
        )
        
        # Recalculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            enhanced_mappings, missing_fields, pa_form_fields
        )
        
        return {
            "field_mappings": enhanced_mappings,
            "missing_fields": missing_fields,
            "overall_confidence": overall_confidence,
            "processing_notes": mapping_result.get("processing_notes", "")
        }
    
    def _validate_and_enhance_mapping(
        self, 
        field_name: str, 
        mapping: Dict[str, Any], 
        referral_data: Dict[str, Any], 
        pa_form_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and enhance a single field mapping.
        
        Args:
            field_name: Name of the PA form field
            mapping: Original mapping from GPT-4
            referral_data: Original referral data
            pa_form_fields: PA form field definitions
            
        Returns:
            Enhanced and validated mapping
        """
        enhanced = mapping.copy()
        
        # Validate confidence score
        confidence = float(mapping.get("confidence", 0.0))
        if confidence > 1.0:
            confidence = 1.0
        elif confidence < 0.0:
            confidence = 0.0
        enhanced["confidence"] = confidence
        
        # Apply field-specific validations
        mapped_value = mapping.get("mapped_value", "")
        
        if field_name.lower() in ["date_of_birth", "dob", "birth_date"]:
            enhanced["mapped_value"] = self._normalize_date(mapped_value)
            if not self._is_valid_date(enhanced["mapped_value"]):
                enhanced["confidence"] *= 0.7  # Reduce confidence for invalid dates
        
        elif field_name.lower() in ["phone", "phone_number", "contact_number"]:
            enhanced["mapped_value"] = self._normalize_phone(mapped_value)
            if not self._is_valid_phone(enhanced["mapped_value"]):
                enhanced["confidence"] *= 0.8
        
        elif field_name.lower() in ["npi", "npi_number", "provider_npi"]:
            if not self._is_valid_npi(mapped_value):
                enhanced["confidence"] *= 0.6
        
        # Add field priority
        enhanced["priority"] = self._get_field_priority(field_name)
        
        # Add validation status
        enhanced["validation_status"] = "valid" if enhanced["confidence"] >= 0.7 else "needs_review"
        
        return enhanced
    
    def _classify_missing_fields(
        self, 
        missing_fields: List[Dict[str, Any]], 
        pa_form_fields: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Classify missing fields by priority and add recommendations.
        
        Args:
            missing_fields: List of missing field information
            pa_form_fields: PA form field definitions
            
        Returns:
            Classified and enhanced missing fields list
        """
        classified = []
        
        for field_info in missing_fields:
            field_name = field_info.get("field_name", "")
            priority = self._get_field_priority(field_name)
            
            enhanced_field = {
                "field_name": field_name,
                "priority": priority,
                "reason": field_info.get("reason", "not found in referral data"),
                "required": priority == "critical",
                "recommendations": self._get_field_recommendations(field_name)
            }
            
            classified.append(enhanced_field)
        
        # Sort by priority
        priority_order = {"critical": 0, "important": 1, "optional": 2}
        classified.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return classified
    
    def _get_field_priority(self, field_name: str) -> str:
        """
        Determine priority level of a PA form field.
        
        Args:
            field_name: Name of the field
            
        Returns:
            Priority level: "critical", "important", or "optional"
        """
        field_lower = field_name.lower()
        
        for priority, fields in self.field_priorities.items():
            if any(f in field_lower for f in fields):
                return priority
        
        return "optional"
    
    def _get_field_recommendations(self, field_name: str) -> List[str]:
        """
        Get recommendations for missing field completion.
        
        Args:
            field_name: Name of the missing field
            
        Returns:
            List of recommendations for completing the field
        """
        field_lower = field_name.lower()
        
        if "npi" in field_lower:
            return ["Check provider credentials in referral", "Search NPI registry"]
        elif "insurance" in field_lower or "member" in field_lower:
            return ["Verify insurance card information", "Check patient demographics"]
        elif "phone" in field_lower:
            return ["Check provider contact information", "Verify with office directory"]
        elif "address" in field_lower:
            return ["Check provider letterhead", "Verify with practice information"]
        else:
            return ["Review referral packet thoroughly", "Contact referring provider if needed"]
    
    def _calculate_overall_confidence(
        self, 
        field_mappings: Dict[str, Any], 
        missing_fields: List[Dict[str, Any]], 
        pa_form_fields: Dict[str, Any]
    ) -> float:
        """
        Calculate overall confidence score for the mapping process.
        
        Args:
            field_mappings: Successfully mapped fields
            missing_fields: List of missing fields
            pa_form_fields: All available PA form fields
            
        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        if not pa_form_fields:
            return 0.0
        
        total_fields = len(pa_form_fields)
        mapped_count = len(field_mappings)
        missing_count = len(missing_fields)
        
        # Calculate weighted confidence based on field priorities and individual confidence
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for field_name, mapping in field_mappings.items():
            priority = mapping.get("priority", "optional")
            field_confidence = mapping.get("confidence", 0.0)
            
            if priority == "critical":
                weight = 3.0
            elif priority == "important":
                weight = 2.0
            else:
                weight = 1.0
            
            weighted_confidence += field_confidence * weight
            total_weight += weight
        
        # Penalty for missing critical fields
        critical_missing = sum(1 for f in missing_fields if f.get("priority") == "critical")
        critical_penalty = critical_missing * 0.2
        
        if total_weight > 0:
            base_confidence = weighted_confidence / total_weight
        else:
            base_confidence = 0.0
        
        # Completion bonus (up to 20% boost for high completion rate)
        completion_rate = mapped_count / total_fields if total_fields > 0 else 0.0
        completion_bonus = completion_rate * 0.2
        
        final_confidence = base_confidence + completion_bonus - critical_penalty
        
        return max(0.0, min(1.0, final_confidence))
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to MM/DD/YYYY format."""
        if not date_str:
            return date_str
        
        # Remove extra whitespace
        date_str = date_str.strip()
        
        # Common date patterns
        patterns = [
            r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})',  # MM/DD/YYYY or MM-DD-YYYY
            r'(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})',  # YYYY/MM/DD or YYYY-MM-DD
            r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2})'   # MM/DD/YY or MM-DD-YY
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                groups = match.groups()
                if len(groups[2]) == 2:  # 2-digit year
                    year = f"20{groups[2]}" if int(groups[2]) < 50 else f"19{groups[2]}"
                    return f"{groups[0].zfill(2)}/{groups[1].zfill(2)}/{year}"
                elif len(groups[0]) == 4:  # YYYY format first
                    return f"{groups[1].zfill(2)}/{groups[2].zfill(2)}/{groups[0]}"
                else:  # Standard MM/DD/YYYY
                    return f"{groups[0].zfill(2)}/{groups[1].zfill(2)}/{groups[2]}"
        
        return date_str
    
    def _normalize_phone(self, phone_str: str) -> str:
        """Normalize phone number to (XXX) XXX-XXXX format."""
        if not phone_str:
            return phone_str
        
        # Extract digits only
        digits = re.sub(r'\D', '', phone_str)
        
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits.startswith('1'):
            return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        
        return phone_str  # Return original if can't normalize
    
    def _is_valid_date(self, date_str: str) -> bool:
        """Validate date format and reasonableness."""
        try:
            from datetime import datetime
            datetime.strptime(date_str, "%m/%d/%Y")
            return True
        except ValueError:
            return False
    
    def _is_valid_phone(self, phone_str: str) -> bool:
        """Validate phone number format."""
        return bool(re.match(r'^\(\d{3}\) \d{3}-\d{4}$', phone_str))
    
    def _is_valid_npi(self, npi_str: str) -> bool:
        """Validate NPI number format (10 digits)."""
        return bool(re.match(r'^\d{10}$', npi_str.replace(' ', '').replace('-', '')))
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get current service status and health information.
        
        Returns:
            Service status dictionary
        """
        return {
            "service_name": "openai_field_mapping_service",
            "service_role": "field_mapping",
            "client_initialized": self.client is not None,
            "model_name": self.model_name,
            "api_key_configured": bool(self.api_key),
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "confidence_thresholds": self.confidence_thresholds,
            "field_priorities": self.field_priorities,
            "max_tokens": self.max_tokens
        }


# Global service instance
openai_service = OpenAIService()


def get_openai_service() -> OpenAIService:
    """
    Get the global OpenAI service instance.
    
    Returns:
        OpenAIService instance for dependency injection
    """
    return openai_service