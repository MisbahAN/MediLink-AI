# backend/tests/test_field_mapping.py
"""
Tests for field mapping functionality.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.services.field_mapper import FieldMapper
from app.services.openai_service import OpenAIService
from app.core.config import Settings


class TestFieldMapper:
    """Test FieldMapper functionality."""

    @pytest.fixture
    def field_mapper(self, test_settings):
        """Create FieldMapper instance."""
        return FieldMapper()

    def test_should_normalize_field_names(self, field_mapper):
        """Test field name normalization."""
        test_cases = [
            ("Patient Name", "patient_name"),
            ("Date of Birth", "patient_dob"),
            ("Insurance ID #", "insurance_id"),
            ("Provider NPI Number", "provider_npi"),
            ("Primary Diagnosis Code", "primary_diagnosis_code")
        ]
        
        for input_name, expected in test_cases:
            result = field_mapper.normalize_field_name(input_name)
            assert result == expected

    def test_should_match_patient_names(self, field_mapper):
        """Test patient name matching logic."""
        extracted_data = {
            "page_1": {"text": "Patient: John Michael Doe Jr."},
            "page_2": {"text": "Name: DOE, JOHN MICHAEL JR"}
        }
        
        pa_field = {"field_name": "patient_name", "type": "text"}
        
        result = field_mapper.match_patient_name(extracted_data, pa_field)
        
        assert result["value"] in ["John Michael Doe Jr.", "John Doe"]
        assert result["confidence"] >= 0.8
        assert "source_page" in result

    def test_should_normalize_date_formats(self, field_mapper):
        """Test date format normalization."""
        test_dates = [
            ("01/15/1980", "01/15/1980"),
            ("January 15, 1980", "01/15/1980"),
            ("1980-01-15", "01/15/1980"),
            ("15-Jan-1980", "01/15/1980"),
            ("01-15-80", "01/15/1980")
        ]
        
        for input_date, expected in test_dates:
            result = field_mapper.normalize_date_format(input_date)
            assert result == expected

    def test_should_handle_invalid_dates(self, field_mapper):
        """Test handling of invalid date formats."""
        invalid_dates = ["not a date", "32/15/1980", "", "???"]
        
        for invalid_date in invalid_dates:
            result = field_mapper.normalize_date_format(invalid_date)
            assert result is None

    def test_should_extract_insurance_ids(self, field_mapper):
        """Test insurance ID extraction."""
        text_samples = [
            "Member ID: 123456789",
            "Insurance Number: ABC-123-456",
            "Policy #: XYZ789123",
            "ID Number 987654321"
        ]
        
        for text in text_samples:
            result = field_mapper.extract_insurance_id(text)
            assert result is not None
            assert len(result) >= 6  # Minimum reasonable ID length

    def test_should_calculate_confidence_scores(self, field_mapper):
        """Test confidence score calculation."""
        # High confidence: exact match
        high_conf = field_mapper.calculate_confidence_score(
            extracted_value="John Doe",
            field_type="text",
            source_quality=0.95,
            match_method="exact"
        )
        assert high_conf >= 0.9
        
        # Medium confidence: fuzzy match
        medium_conf = field_mapper.calculate_confidence_score(
            extracted_value="J. Doe",
            field_type="text",
            source_quality=0.8,
            match_method="fuzzy"
        )
        assert 0.6 <= medium_conf < 0.9
        
        # Low confidence: inferred
        low_conf = field_mapper.calculate_confidence_score(
            extracted_value="Doe",
            field_type="text",
            source_quality=0.6,
            match_method="inferred"
        )
        assert low_conf < 0.7

    def test_should_extract_medical_codes(self, field_mapper):
        """Test extraction of medical codes (ICD, CPT, etc.)."""
        text_with_codes = """
        Primary Diagnosis: M05.9 Rheumatoid arthritis, unspecified
        Secondary: M79.3 Panniculitis
        Procedure: 96365 Therapeutic injection
        """
        
        icd_codes = field_mapper.extract_icd_codes(text_with_codes)
        assert "M05.9" in icd_codes
        assert "M79.3" in icd_codes
        
        cpt_codes = field_mapper.extract_cpt_codes(text_with_codes)
        assert "96365" in cpt_codes

    def test_should_match_medication_names(self, field_mapper):
        """Test medication name matching."""
        extracted_text = "Patient will receive Rituximab infusion (Rituxan 500mg)"
        medication_field = {"field_name": "medication_name", "options": ["Rituximab", "Rituxan", "Riabni"]}
        
        result = field_mapper.match_medication(extracted_text, medication_field)
        
        assert result["value"] in ["Rituximab", "Rituxan"]
        assert result["confidence"] >= 0.8

    def test_should_extract_provider_information(self, field_mapper):
        """Test provider information extraction."""
        provider_text = """
        Dr. Sarah Johnson, MD
        License: MD12345
        NPI: 1234567890
        Phone: (555) 123-4567
        """
        
        provider_info = field_mapper.extract_provider_info(provider_text)
        
        assert provider_info["name"]["value"] == "Sarah Johnson"
        assert provider_info["license"]["value"] == "MD12345"
        assert provider_info["npi"]["value"] == "1234567890"
        assert provider_info["phone"]["value"] == "(555) 123-4567"

    def test_should_handle_missing_data(self, field_mapper):
        """Test handling of missing data scenarios."""
        empty_data = {"page_1": {"text": ""}}
        field = {"field_name": "patient_name", "type": "text", "required": True}
        
        result = field_mapper.map_field(empty_data, field)
        
        assert result["value"] is None
        assert result["confidence"] == 0.0
        assert result["missing_reason"] == "not_found"

    def test_should_validate_field_values(self, field_mapper):
        """Test field value validation."""
        # Valid phone number
        phone_result = field_mapper.validate_field_value("(555) 123-4567", "phone")
        assert phone_result["is_valid"] is True
        
        # Invalid phone number
        invalid_phone = field_mapper.validate_field_value("123", "phone")
        assert invalid_phone["is_valid"] is False
        
        # Valid date
        date_result = field_mapper.validate_field_value("01/15/1980", "date")
        assert date_result["is_valid"] is True
        
        # Invalid date
        invalid_date = field_mapper.validate_field_value("32/15/1980", "date")
        assert invalid_date["is_valid"] is False

    def test_should_prioritize_high_confidence_sources(self, field_mapper):
        """Test prioritization of high confidence data sources."""
        extracted_data = {
            "page_1": {"text": "Patient Name: John Doe", "confidence": 0.95},
            "page_2": {"text": "Name: J. D.", "confidence": 0.6},
            "page_3": {"text": "Patient: John Michael Doe", "confidence": 0.88}
        }
        
        field = {"field_name": "patient_name", "type": "text"}
        
        result = field_mapper.map_field(extracted_data, field)
        
        # Should prefer page_1 due to highest confidence
        assert result["source_page"] == 1
        assert result["confidence"] >= 0.9

    def test_should_handle_checkbox_fields(self, field_mapper):
        """Test handling of checkbox field mappings."""
        checkbox_text = "☑ Yes, patient has tried alternative medications"
        checkbox_field = {
            "field_name": "tried_alternatives",
            "type": "checkbox",
            "options": ["yes", "no"]
        }
        
        result = field_mapper.map_checkbox_field(checkbox_text, checkbox_field)
        
        assert result["value"] == "yes"
        assert result["confidence"] >= 0.8

    def test_should_extract_dosage_information(self, field_mapper):
        """Test extraction of medication dosage information."""
        dosage_text = "Rituximab 500mg IV every 6 months"
        
        dosage_info = field_mapper.extract_dosage_info(dosage_text)
        
        assert dosage_info["amount"] == "500mg"
        assert dosage_info["route"] == "IV"
        assert dosage_info["frequency"] == "every 6 months"

    def test_should_normalize_addresses(self, field_mapper):
        """Test address normalization."""
        address_variations = [
            "123 Main St, Anytown, CA 12345",
            "123 Main Street\nAnytown, California 12345",
            "123 Main St.\nAnytown CA 12345-1234"
        ]
        
        for address in address_variations:
            normalized = field_mapper.normalize_address(address)
            assert "123 Main St" in normalized
            assert "Anytown" in normalized
            assert "CA" in normalized
            assert "12345" in normalized


class TestOpenAIService:
    """Test OpenAI field mapping service."""

    @pytest.fixture
    def openai_service(self, test_settings):
        """Create OpenAIService instance."""
        return OpenAIService()

    @pytest.mark.asyncio
    async def test_should_initialize_client(self, openai_service):
        """Test OpenAI client initialization."""
        await openai_service.initialize_client()
        
        assert openai_service.client is not None
        assert openai_service.api_key == "test_openai_key"

    def test_should_create_field_mapping_prompt(self, openai_service, sample_extracted_data, sample_pa_form_fields):
        """Test creation of field mapping prompt."""
        prompt = openai_service.create_field_mapping_prompt(
            extracted_data=sample_extracted_data,
            pa_form_fields=sample_pa_form_fields
        )
        
        assert isinstance(prompt, str)
        assert "patient" in prompt.lower()
        assert "map" in prompt.lower()
        assert "confidence" in prompt.lower()

    @pytest.mark.asyncio
    @patch('app.services.openai_service.OpenAI')
    async def test_should_map_fields_with_ai(self, mock_openai, openai_service, mock_openai_response):
        """Test AI-powered field mapping."""
        # Mock OpenAI response
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create = AsyncMock(
            return_value=Mock(choices=[Mock(message=Mock(content=str(mock_openai_response)))])
        )
        
        extracted_data = {"patient_info": {"name": "John Doe"}}
        pa_fields = {"patient_name": {"type": "text", "required": True}}
        
        result = await openai_service.extract_and_map_fields(extracted_data, pa_fields)
        
        assert isinstance(result, dict)
        assert "mapped_fields" in result
        assert "missing_fields" in result

    @pytest.mark.asyncio
    async def test_should_handle_openai_api_errors(self, openai_service):
        """Test handling of OpenAI API errors."""
        with patch.object(openai_service, 'client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("OpenAI API Error"))
            
            with pytest.raises(Exception) as exc_info:
                await openai_service.extract_and_map_fields({}, {})
            
            assert "OpenAI API Error" in str(exc_info.value)

    def test_should_parse_ai_response(self, openai_service, mock_openai_response):
        """Test parsing of AI response."""
        response_text = str(mock_openai_response)
        
        parsed = openai_service.parse_mapping_response(response_text)
        
        assert isinstance(parsed, dict)
        assert "mapped_fields" in parsed
        assert "missing_fields" in parsed

    @pytest.mark.asyncio
    async def test_should_validate_ai_mappings(self, openai_service):
        """Test validation of AI-generated mappings."""
        ai_mappings = {
            "patient_name": {"value": "John Doe", "confidence": 0.95},
            "patient_dob": {"value": "01/15/1980", "confidence": 0.92},
            "invalid_phone": {"value": "123", "confidence": 0.8}  # Invalid phone
        }
        
        validated = openai_service.validate_mappings(ai_mappings)
        
        # Valid fields should pass
        assert validated["patient_name"]["is_valid"] is True
        assert validated["patient_dob"]["is_valid"] is True
        
        # Invalid field should be flagged
        assert validated["invalid_phone"]["is_valid"] is False

    def test_should_format_medical_context(self, openai_service):
        """Test formatting of medical context for AI."""
        context = openai_service.format_medical_context(
            form_type="rituximab_pa",
            condition="rheumatoid_arthritis"
        )
        
        assert "rituximab" in context.lower()
        assert "rheumatoid" in context.lower()
        assert "prior authorization" in context.lower()


class TestFieldMappingIntegration:
    """Integration tests for field mapping pipeline."""

    @pytest.mark.asyncio
    async def test_should_map_complete_patient_record(self, test_settings, sample_extracted_data, sample_pa_form_fields):
        """Test complete patient record mapping."""
        mapper = FieldMapper(test_settings)
        openai_service = OpenAIService(test_settings)
        
        # Mock OpenAI service
        with patch.object(openai_service, 'extract_and_map_fields') as mock_ai:
            mock_ai.return_value = {
                "mapped_fields": {
                    "patient_name": {"value": "John Doe", "confidence": 0.95, "source": "demographics"},
                    "patient_dob": {"value": "01/15/1980", "confidence": 0.92, "source": "demographics"}
                },
                "missing_fields": []
            }
            
            # Traditional mapping first
            traditional_mappings = {}
            for field_name, field_info in sample_pa_form_fields.items():
                traditional_mappings[field_name] = mapper.map_field(sample_extracted_data, field_info)
            
            # AI enhancement
            ai_mappings = await openai_service.extract_and_map_fields(
                sample_extracted_data, sample_pa_form_fields
            )
            
            # Combine results
            final_mappings = mapper.combine_mapping_results(traditional_mappings, ai_mappings)
            
            assert "patient_name" in final_mappings
            assert "patient_dob" in final_mappings
            assert all(mapping["confidence"] >= 0.8 for mapping in final_mappings.values())

    def test_should_identify_missing_required_fields(self, test_settings):
        """Test identification of missing required fields."""
        mapper = FieldMapper(test_settings)
        
        incomplete_data = {
            "page_1": {"text": "Patient: John Doe"}  # Missing DOB, insurance, etc.
        }
        
        required_fields = {
            "patient_name": {"type": "text", "required": True},
            "patient_dob": {"type": "date", "required": True},
            "insurance_id": {"type": "text", "required": True}
        }
        
        mappings = {}
        missing_fields = []
        
        for field_name, field_info in required_fields.items():
            result = mapper.map_field(incomplete_data, field_info)
            if result["value"] is None and field_info.get("required"):
                missing_fields.append({
                    "field": field_name,
                    "reason": result.get("missing_reason", "not_found"),
                    "priority": "high" if field_info.get("required") else "medium"
                })
            else:
                mappings[field_name] = result
        
        assert len(missing_fields) >= 2  # Should identify missing DOB and insurance ID
        assert any(field["field"] == "patient_dob" for field in missing_fields)
        assert any(field["field"] == "insurance_id" for field in missing_fields)

    def test_should_handle_conflicting_data_sources(self, test_settings):
        """Test handling of conflicting data from different sources."""
        mapper = FieldMapper(test_settings)
        
        conflicting_data = {
            "page_1": {"text": "Patient: John Smith", "confidence": 0.85},
            "page_2": {"text": "Patient Name: John Doe", "confidence": 0.95},
            "page_3": {"text": "Name: J. Smith", "confidence": 0.6}
        }
        
        field = {"field_name": "patient_name", "type": "text", "required": True}
        
        result = mapper.map_field(conflicting_data, field)
        
        # Should prefer highest confidence source (page_2)
        assert result["value"] == "John Doe"
        assert result["source_page"] == 2
        assert result["confidence"] >= 0.9

    def test_should_apply_medical_field_rules(self, test_settings):
        """Test application of medical-specific field mapping rules."""
        mapper = FieldMapper(test_settings)
        
        # Test medication dose validation
        dose_text = "Rituximab 1000mg IV"
        dose_result = mapper.validate_medication_dose(dose_text, "rituximab")
        assert dose_result["is_valid"] is True
        
        # Test ICD code validation
        icd_text = "M05.9"
        icd_result = mapper.validate_icd_code(icd_text)
        assert icd_result["is_valid"] is True
        assert icd_result["description"] is not None

    @pytest.mark.asyncio
    async def test_should_optimize_for_processing_time(self, test_settings):
        """Test field mapping performance optimization."""
        mapper = FieldMapper(test_settings)
        
        # Large dataset simulation
        large_extracted_data = {}
        for i in range(50):  # 50 pages
            large_extracted_data[f"page_{i}"] = {
                "text": f"Page {i} content with patient data",
                "confidence": 0.8 + (i % 3) * 0.05
            }
        
        many_fields = {}
        for i in range(30):  # 30 form fields
            many_fields[f"field_{i}"] = {
                "field_name": f"field_{i}",
                "type": "text",
                "required": i < 10
            }
        
        import time
        start_time = time.time()
        
        # Process all mappings
        results = {}
        for field_name, field_info in many_fields.items():
            results[field_name] = mapper.map_field(large_extracted_data, field_info)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time
        assert processing_time < 10  # 10 seconds max
        assert len(results) == 30

    def test_should_maintain_data_traceability(self, test_settings):
        """Test that field mappings maintain source traceability."""
        mapper = FieldMapper(test_settings)
        
        traced_data = {
            "page_1": {
                "text": "Patient: John Doe DOB: 01/15/1980",
                "source_file": "referral_packet.pdf",
                "extraction_method": "mistral_ocr",
                "confidence": 0.92
            }
        }
        
        field = {"field_name": "patient_name", "type": "text"}
        
        result = mapper.map_field(traced_data, field)
        
        assert "source_page" in result
        assert "source_file" in result
        assert "extraction_method" in result
        assert result["source_file"] == "referral_packet.pdf"
        assert result["extraction_method"] == "mistral_ocr"

    @pytest.mark.asyncio
    async def test_should_handle_multilingual_content(self, test_settings):
        """Test handling of multilingual content in documents."""
        mapper = FieldMapper(test_settings)
        
        multilingual_data = {
            "page_1": {"text": "Paciente: Juan Pérez\nPatient: John Perez", "confidence": 0.85}
        }
        
        field = {"field_name": "patient_name", "type": "text"}
        
        result = mapper.map_field(multilingual_data, field)
        
        # Should extract English version when available
        assert "John Perez" in result["value"] or "Juan Pérez" in result["value"]
        assert result["confidence"] >= 0.8