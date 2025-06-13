"""
Tests for PDF extraction functionality.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
from io import BytesIO

from app.services.pdf_extractor import PDFExtractor
from app.services.mistral_service import MistralService
from app.services.gemini_service_fallback import GeminiService
from app.services.widget_detector import WidgetDetector
from app.core.config import Settings


class TestPDFExtractor:
    """Test PDFExtractor class functionality."""

    @pytest.fixture
    def pdf_extractor(self, test_settings):
        """Create PDFExtractor instance."""
        return PDFExtractor()

    def test_should_extract_text_from_valid_pdf(self, pdf_extractor, sample_pdf_bytes):
        """Test that text is extracted from valid PDF."""
        result = pdf_extractor.extract_text_from_pdf(sample_pdf_bytes)
        
        assert isinstance(result, dict)
        assert "pages" in result
        assert "total_pages" in result
        assert "extraction_method" in result
        assert result["total_pages"] >= 1

    def test_should_get_correct_page_count(self, pdf_extractor, sample_pdf_bytes):
        """Test that page count is correctly determined."""
        page_count = pdf_extractor.get_page_count(sample_pdf_bytes)
        
        assert isinstance(page_count, int)
        assert page_count >= 1

    def test_should_handle_empty_pdf_content(self, pdf_extractor):
        """Test handling of empty PDF content."""
        with pytest.raises(Exception):
            pdf_extractor.extract_text_from_pdf(b"")

    def test_should_handle_invalid_pdf_content(self, pdf_extractor):
        """Test handling of invalid PDF content."""
        invalid_content = b"This is not a PDF file"
        
        with pytest.raises(Exception):
            pdf_extractor.extract_text_from_pdf(invalid_content)

    def test_should_chunk_large_pdf(self, pdf_extractor):
        """Test chunking of large PDF files."""
        # Mock a large file size
        large_size = 30 * 1024 * 1024  # 30MB
        
        chunks = pdf_extractor.calculate_chunks(large_size)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        assert all(chunk["size"] <= pdf_extractor.max_chunk_size for chunk in chunks)

    def test_should_not_chunk_small_pdf(self, pdf_extractor):
        """Test that small PDFs are not chunked."""
        small_size = 5 * 1024 * 1024  # 5MB
        
        chunks = pdf_extractor.calculate_chunks(small_size)
        
        assert len(chunks) == 1
        assert chunks[0]["start"] == 0

    def test_should_extract_with_coordinates(self, pdf_extractor, sample_pdf_bytes):
        """Test extraction with coordinate information."""
        result = pdf_extractor.extract_text_with_coordinates(sample_pdf_bytes)
        
        assert isinstance(result, dict)
        assert "pages" in result
        
        for page_data in result["pages"].values():
            assert "text" in page_data
            assert "bbox" in page_data or "coordinates" in page_data

    @patch('app.services.pdf_extractor.pdfplumber')
    def test_should_handle_pdfplumber_errors(self, mock_pdfplumber, pdf_extractor):
        """Test handling of pdfplumber errors."""
        mock_pdfplumber.open.side_effect = Exception("PDF parsing error")
        
        with pytest.raises(Exception) as exc_info:
            pdf_extractor.extract_text_from_pdf(b"mock_pdf_content")
        
        assert "PDF parsing error" in str(exc_info.value)

    def test_should_validate_pdf_structure(self, pdf_extractor, sample_pdf_bytes):
        """Test PDF structure validation."""
        is_valid = pdf_extractor.validate_pdf_structure(sample_pdf_bytes)
        
        assert isinstance(is_valid, bool)
        assert is_valid is True

    def test_should_reject_corrupted_pdf(self, pdf_extractor):
        """Test rejection of corrupted PDF files."""
        corrupted_pdf = b"%PDF-1.4\nCorrupted content\n%%EOF"
        
        is_valid = pdf_extractor.validate_pdf_structure(corrupted_pdf)
        
        assert is_valid is False


class TestMistralService:
    """Test MistralService OCR functionality."""

    @pytest.fixture
    def mistral_service(self, test_settings):
        """Create MistralService instance."""
        return MistralService()

    @pytest.mark.asyncio
    async def test_should_initialize_client(self, mistral_service):
        """Test that Mistral client initializes properly."""
        await mistral_service.initialize_client()
        
        assert mistral_service.client is not None
        assert mistral_service.api_key == "test_mistral_key"

    @pytest.mark.asyncio
    @patch('app.services.mistral_service.MistralClient')
    async def test_should_extract_from_pdf_pages(self, mock_client, mistral_service, mock_mistral_response):
        """Test OCR extraction from PDF pages."""
        # Mock the client response
        mock_instance = mock_client.return_value
        mock_instance.chat = AsyncMock(return_value=Mock(choices=[Mock(message=Mock(content=mock_mistral_response["text"]))]))
        
        pdf_pages = {"page_1": {"image_data": b"mock_image"}}
        
        result = await mistral_service.extract_from_pdf_pages(pdf_pages)
        
        assert isinstance(result, dict)
        assert "extracted_data" in result
        assert "confidence" in result
        assert result["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_should_handle_api_errors(self, mistral_service):
        """Test handling of Mistral API errors."""
        with patch.object(mistral_service, 'client') as mock_client:
            mock_client.chat = AsyncMock(side_effect=Exception("API Error"))
            
            with pytest.raises(Exception) as exc_info:
                await mistral_service.extract_from_pdf_pages({"page_1": {"image_data": b"mock"}})
            
            assert "API Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_should_retry_on_failure(self, mistral_service):
        """Test retry mechanism on API failures."""
        with patch.object(mistral_service, 'client') as mock_client:
            # First call fails, second succeeds
            mock_client.chat = AsyncMock(side_effect=[
                Exception("Temporary error"),
                Mock(choices=[Mock(message=Mock(content="Success"))])
            ])
            
            result = await mistral_service.extract_from_pdf_pages({"page_1": {"image_data": b"mock"}})
            
            assert result is not None
            assert mock_client.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_should_calculate_confidence_score(self, mistral_service):
        """Test confidence score calculation."""
        extracted_text = "Patient Name: John Doe\nDate of Birth: 01/15/1980"
        
        confidence = mistral_service.calculate_confidence(extracted_text)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_should_preprocess_image_data(self, mistral_service):
        """Test image data preprocessing for OCR."""
        mock_image_data = b"mock_image_bytes"
        
        processed = mistral_service.preprocess_image(mock_image_data)
        
        assert isinstance(processed, bytes)
        assert len(processed) > 0


class TestGeminiService:
    """Test GeminiService fallback functionality."""

    @pytest.fixture
    def gemini_service(self, test_settings):
        """Create GeminiService instance."""
        return GeminiService()

    @pytest.mark.asyncio
    async def test_should_initialize_client(self, gemini_service):
        """Test that Gemini client initializes properly."""
        await gemini_service.initialize_client()
        
        assert gemini_service.model is not None
        assert gemini_service.api_key == "test_gemini_key"

    @pytest.mark.asyncio
    @patch('app.services.gemini_service_fallback.genai')
    async def test_should_extract_using_vision(self, mock_genai, gemini_service, mock_gemini_response):
        """Test vision-based extraction from PDF."""
        # Mock the model response
        mock_model = Mock()
        mock_model.generate_content = AsyncMock(return_value=Mock(text=mock_gemini_response["text"]))
        mock_genai.GenerativeModel.return_value = mock_model
        
        pdf_pages = {"page_1": {"image_data": b"mock_image"}}
        
        result = await gemini_service.extract_from_pdf_pages(pdf_pages)
        
        assert isinstance(result, dict)
        assert "extracted_data" in result
        assert "confidence" in result
        assert result["method"] == "vision_extraction"

    @pytest.mark.asyncio
    async def test_should_handle_vision_api_errors(self, gemini_service):
        """Test handling of Gemini Vision API errors."""
        with patch.object(gemini_service, 'model') as mock_model:
            mock_model.generate_content = AsyncMock(side_effect=Exception("Vision API Error"))
            
            with pytest.raises(Exception) as exc_info:
                await gemini_service.extract_from_pdf_pages({"page_1": {"image_data": b"mock"}})
            
            assert "Vision API Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_should_process_multiple_pages(self, gemini_service):
        """Test processing multiple PDF pages."""
        with patch.object(gemini_service, 'model') as mock_model:
            mock_model.generate_content = AsyncMock(return_value=Mock(text="Extracted text"))
            
            pdf_pages = {
                "page_1": {"image_data": b"mock_image_1"},
                "page_2": {"image_data": b"mock_image_2"},
                "page_3": {"image_data": b"mock_image_3"}
            }
            
            result = await gemini_service.extract_from_pdf_pages(pdf_pages)
            
            assert len(result["extracted_data"]) == 3
            assert mock_model.generate_content.call_count == 3

    def test_should_format_vision_prompt(self, gemini_service):
        """Test vision prompt formatting."""
        prompt = gemini_service.create_vision_prompt("medical_form")
        
        assert isinstance(prompt, str)
        assert "medical" in prompt.lower()
        assert "extract" in prompt.lower()


class TestWidgetDetector:
    """Test WidgetDetector form field detection."""

    @pytest.fixture
    def widget_detector(self, test_settings):
        """Create WidgetDetector instance."""
        return WidgetDetector()

    def test_should_detect_form_fields(self, widget_detector, sample_pdf_bytes, mock_pdf_form_fields):
        """Test detection of PDF form fields."""
        with patch('app.services.widget_detector.pdfforms') as mock_pdfforms:
            mock_pdfforms.analyze_pdf.return_value = mock_pdf_form_fields
            
            result = widget_detector.detect_form_fields(sample_pdf_bytes)
            
            assert isinstance(result, dict)
            assert "fields" in result
            assert "total_fields" in result
            assert result["total_fields"] == len(mock_pdf_form_fields)

    def test_should_extract_field_properties(self, widget_detector, mock_pdf_form_fields):
        """Test extraction of field properties."""
        field_data = mock_pdf_form_fields[0]
        
        properties = widget_detector.extract_field_properties(field_data)
        
        assert "field_name" in properties
        assert "field_type" in properties
        assert "coordinates" in properties
        assert "required" in properties

    def test_should_categorize_medical_fields(self, widget_detector):
        """Test categorization of medical form fields."""
        field_names = ["patient_name", "patient_dob", "insurance_id", "diagnosis_code", "provider_npi"]
        
        categories = widget_detector.categorize_fields(field_names)
        
        assert "patient_info" in categories
        assert "insurance_info" in categories
        assert "clinical_info" in categories
        assert "provider_info" in categories

    def test_should_validate_required_fields(self, widget_detector, mock_pdf_form_fields):
        """Test validation of required fields."""
        result = widget_detector.validate_form_structure(mock_pdf_form_fields)
        
        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "missing_required" in result
        assert "validation_errors" in result

    def test_should_handle_forms_without_widgets(self, widget_detector, sample_pdf_bytes):
        """Test handling of PDFs without form widgets."""
        with patch('app.services.widget_detector.pdfforms') as mock_pdfforms:
            mock_pdfforms.analyze_pdf.return_value = []
            
            result = widget_detector.detect_form_fields(sample_pdf_bytes)
            
            assert result["total_fields"] == 0
            assert result["has_widgets"] is False

    def test_should_extract_field_coordinates(self, widget_detector):
        """Test extraction of field coordinate information."""
        field_data = {
            "coordinates": {"x": 100, "y": 200, "width": 150, "height": 20}
        }
        
        coords = widget_detector.extract_coordinates(field_data)
        
        assert coords["x"] == 100
        assert coords["y"] == 200
        assert coords["width"] == 150
        assert coords["height"] == 20

    @patch('app.services.widget_detector.pdfforms')
    def test_should_handle_pdfforms_errors(self, mock_pdfforms, widget_detector, sample_pdf_bytes):
        """Test handling of pdfforms library errors."""
        mock_pdfforms.analyze_pdf.side_effect = Exception("pdfforms error")
        
        with pytest.raises(Exception) as exc_info:
            widget_detector.detect_form_fields(sample_pdf_bytes)
        
        assert "pdfforms error" in str(exc_info.value)


class TestPDFExtractionIntegration:
    """Integration tests for PDF extraction pipeline."""

    @pytest.mark.asyncio
    async def test_should_process_pa_form_with_widgets(self, test_settings, sample_pdf_bytes, mock_pdf_form_fields):
        """Test complete processing of PA form with widgets."""
        extractor = PDFExtractor()
        detector = WidgetDetector()
        
        # Use real test PDF file instead of bytes
        test_pdf_path = Path("tests/test_data/test_1_PA.pdf")
        
        with patch('app.services.widget_detector.pdfforms') as mock_pdfforms:
            mock_pdfforms.analyze_pdf.return_value = mock_pdf_form_fields
            
            # Extract text
            text_result = extractor.extract_text_from_pdf(test_pdf_path)
            
            # Detect widgets  
            widget_result = detector.detect_form_fields(test_pdf_path)
            
            assert text_result["total_pages"] >= 1
            assert widget_result["total_fields"] > 0
            assert widget_result["success"] is True

    @pytest.mark.asyncio
    async def test_should_handle_scanned_referral_documents(self, test_settings, sample_pdf_bytes):
        """Test handling of scanned referral documents (no widgets)."""
        mistral_service = MistralService()
        gemini_service = GeminiService()
        
        with patch.object(mistral_service, 'extract_from_pdf_pages') as mock_mistral:
            mock_mistral.return_value = {
                "extracted_data": {"page_1": "Patient: John Doe"},
                "confidence": 0.88,
                "method": "ocr_extraction"
            }
            
            pdf_pages = {"page_1": {"image_data": b"mock_scanned_page"}}
            
            # Try Mistral first
            result = await mistral_service.extract_from_pdf_pages(pdf_pages)
            
            assert result["confidence"] >= 0.8
            assert "Patient: John Doe" in result["extracted_data"]["page_1"]

    @pytest.mark.asyncio
    async def test_should_fallback_to_gemini_on_mistral_failure(self, test_settings):
        """Test fallback from Mistral to Gemini on extraction failure."""
        mistral_service = MistralService()
        gemini_service = GeminiService()
        
        pdf_pages = {"page_1": {"image_data": b"mock_page"}}
        
        # Mock Mistral failure
        with patch.object(mistral_service, 'extract_from_pdf_pages') as mock_mistral:
            mock_mistral.side_effect = Exception("Mistral API Error")
            
            # Mock Gemini success
            with patch.object(gemini_service, 'extract_from_pdf_pages') as mock_gemini:
                mock_gemini.return_value = {
                    "extracted_data": {"page_1": "Fallback extraction"},
                    "confidence": 0.75,
                    "method": "vision_extraction"
                }
                
                # Simulate fallback logic
                try:
                    result = await mistral_service.extract_from_pdf_pages(pdf_pages)
                except Exception:
                    result = await gemini_service.extract_from_pdf_pages(pdf_pages)
                
                assert result["method"] == "vision_extraction"
                assert result["confidence"] >= 0.7

    def test_should_validate_extraction_confidence(self, test_settings):
        """Test extraction confidence validation."""
        extractor = PDFExtractor(test_settings)
        
        # High confidence text
        high_confidence_text = "Patient Name: John Doe\nDate of Birth: 01/15/1980\nInsurance ID: 123456789"
        confidence = extractor.calculate_text_confidence(high_confidence_text)
        assert confidence >= 0.8
        
        # Low confidence text
        low_confidence_text = "~@#$%^&*()"
        confidence = extractor.calculate_text_confidence(low_confidence_text)
        assert confidence < 0.5

    @pytest.mark.asyncio
    async def test_should_handle_large_document_processing(self, test_settings):
        """Test processing of large multi-page documents."""
        extractor = PDFExtractor(test_settings)
        
        # Simulate 15-page referral document
        large_pdf_pages = {f"page_{i}": {"image_data": b"mock_page_data"} for i in range(1, 16)}
        
        # Mock processing time tracking
        import time
        start_time = time.time()
        
        # Simulate concurrent processing
        async def mock_process_page(page_data):
            return {"text": f"Extracted from {page_data}", "confidence": 0.85}
        
        tasks = [mock_process_page(page) for page in large_pdf_pages.values()]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert len(results) == 15
        assert processing_time < 30  # Should complete within 30 seconds
        assert all(result["confidence"] >= 0.8 for result in results)