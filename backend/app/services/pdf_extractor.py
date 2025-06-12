"""
PDF extraction service for processing PA forms and referral documents.

This service provides comprehensive PDF text extraction with support for both
native text PDFs and scanned image PDFs, following the extraction strategies
defined in the PA referral guide documentation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timezone
import pdfplumber
import pdfforms
from io import BytesIO

from ..models.schemas import ExtractedField, ConfidenceLevel
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class PDFExtractor:
    """
    Base PDF extraction class supporting multiple extraction methods.
    
    Handles both native text PDFs (high confidence) and scanned image PDFs
    (requiring OCR/AI vision) with appropriate confidence scoring based on
    the extraction method used.
    """
    
    def __init__(self):
        """Initialize PDF extractor with confidence thresholds."""
        self.confidence_levels = {
            "native_pdf_text": 0.95,
            "form_annotations": 0.90,
            "regex_patterns": 0.85,
            "coordinate_based": 0.80,
            "fallback_required": 0.40  # Indicates OCR/AI vision needed
        }
        
        # Processing limits for large files
        self.max_pages_per_chunk = 10
        self.max_file_size_mb = 50
        
        logger.info("PDF extractor initialized with confidence thresholds")
    
    def get_page_count(self, pdf_path: Union[str, Path, bytes]) -> int:
        """
        Get the total number of pages in a PDF document.
        
        Args:
            pdf_path: Path to PDF file or bytes content
            
        Returns:
            Number of pages in the PDF
            
        Raises:
            ValueError: If PDF cannot be opened or is invalid
        """
        try:
            if isinstance(pdf_path, bytes):
                pdf_file = BytesIO(pdf_path)
            else:
                pdf_file = pdf_path
                
            with pdfplumber.open(pdf_file) as pdf:
                page_count = len(pdf.pages)
                logger.info(f"PDF contains {page_count} pages")
                return page_count
                
        except Exception as e:
            logger.error(f"Failed to get page count from PDF: {e}")
            raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")
    
    def extract_text_from_pdf(
        self, 
        pdf_path: Union[str, Path, bytes],
        extract_coordinates: bool = True,
        page_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Extract text content from PDF with coordinate information.
        
        Args:
            pdf_path: Path to PDF file or bytes content
            extract_coordinates: Whether to include text coordinate information
            page_range: Optional tuple (start_page, end_page) for partial extraction
            
        Returns:
            Dictionary containing:
            - pages: List of page data with text and coordinates
            - total_pages: Total number of pages processed
            - extraction_method: Method used for extraction
            - confidence: Overall confidence score
            - has_form_fields: Whether PDF contains fillable form fields
            - processing_timestamp: Extraction completion time
            
        Raises:
            ValueError: If PDF processing fails
        """
        try:
            if isinstance(pdf_path, bytes):
                pdf_file = BytesIO(pdf_path)
            else:
                pdf_file = pdf_path
                
            with pdfplumber.open(pdf_file) as pdf:
                total_pages = len(pdf.pages)
                
                # Determine page range
                if page_range:
                    start_page, end_page = page_range
                    start_page = max(0, start_page)
                    end_page = min(total_pages, end_page)
                    pages_to_process = pdf.pages[start_page:end_page]
                    page_offset = start_page
                else:
                    pages_to_process = pdf.pages
                    page_offset = 0
                
                # Extract text from each page
                pages_data = []
                total_text_length = 0
                has_form_fields = False
                
                for idx, page in enumerate(pages_to_process):
                    page_num = idx + page_offset + 1
                    page_data = self._extract_page_content(page, page_num, extract_coordinates)
                    pages_data.append(page_data)
                    total_text_length += len(page_data.get('text', ''))
                    
                    # Check for form fields on this page
                    if page_data.get('form_fields'):
                        has_form_fields = True
                
                # Determine extraction method and confidence
                extraction_method, confidence = self._determine_extraction_confidence(
                    total_text_length, has_form_fields, len(pages_data)
                )
                
                result = {
                    "pages": pages_data,
                    "total_pages": len(pages_data),
                    "extraction_method": extraction_method,
                    "confidence": confidence,
                    "has_form_fields": has_form_fields,
                    "total_text_length": total_text_length,
                    "processing_timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                logger.info(
                    f"Extracted {total_text_length} characters from {len(pages_data)} pages "
                    f"using {extraction_method} (confidence: {confidence:.2f})"
                )
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise ValueError(f"PDF text extraction failed: {str(e)}")
    
    def _extract_page_content(
        self, 
        page, 
        page_num: int, 
        extract_coordinates: bool
    ) -> Dict[str, Any]:
        """
        Extract content from a single PDF page.
        
        Args:
            page: pdfplumber Page object
            page_num: Page number (1-indexed)
            extract_coordinates: Whether to extract coordinate information
            
        Returns:
            Dictionary with page content and metadata
        """
        page_data = {
            "page_number": page_num,
            "text": "",
            "text_objects": [],
            "form_fields": [],
            "tables": [],
            "images": [],
            "bbox": None
        }
        
        try:
            # Extract raw text
            text = page.extract_text() or ""
            page_data["text"] = text
            
            # Extract text with coordinates if requested
            if extract_coordinates and text:
                chars = page.chars
                if chars:
                    # Group characters into text objects with bounding boxes
                    text_objects = self._group_text_objects(chars)
                    page_data["text_objects"] = text_objects
            
            # Extract table information
            tables = page.extract_tables()
            if tables:
                page_data["tables"] = [
                    {
                        "table_index": i,
                        "rows": len(table),
                        "cols": len(table[0]) if table else 0,
                        "data": table[:5]  # First 5 rows for preview
                    }
                    for i, table in enumerate(tables)
                ]
            
            # Check for images/figures
            images = getattr(page, 'images', [])
            if images:
                page_data["images"] = [
                    {
                        "bbox": img.get('bbox', []),
                        "width": img.get('width', 0),
                        "height": img.get('height', 0)
                    }
                    for img in images
                ]
            
            # Get page dimensions
            if hasattr(page, 'bbox'):
                page_data["bbox"] = {
                    "x0": page.bbox[0],
                    "y0": page.bbox[1], 
                    "x1": page.bbox[2],
                    "y1": page.bbox[3],
                    "width": page.bbox[2] - page.bbox[0],
                    "height": page.bbox[3] - page.bbox[1]
                }
            
        except Exception as e:
            logger.warning(f"Error extracting content from page {page_num}: {e}")
        
        return page_data
    
    def _group_text_objects(self, chars: List[Dict]) -> List[Dict[str, Any]]:
        """
        Group individual characters into text objects with bounding boxes.
        
        Args:
            chars: List of character dictionaries from pdfplumber
            
        Returns:
            List of text objects with combined bounding boxes
        """
        if not chars:
            return []
        
        text_objects = []
        current_object = None
        
        for char in chars:
            char_text = char.get('text', '')
            if not char_text or char_text.isspace():
                # Finalize current object if exists
                if current_object:
                    text_objects.append(current_object)
                    current_object = None
                continue
            
            # Start new text object or continue current one
            if current_object is None:
                current_object = {
                    "text": char_text,
                    "bbox": {
                        "x0": char.get('x0', 0),
                        "y0": char.get('y0', 0),
                        "x1": char.get('x1', 0),
                        "y1": char.get('y1', 0)
                    },
                    "font": char.get('fontname', ''),
                    "size": char.get('size', 0)
                }
            else:
                # Check if character continues current object (same line, similar font)
                char_y0 = char.get('y0', 0)
                obj_y0 = current_object["bbox"]["y0"]
                same_line = abs(char_y0 - obj_y0) < 2  # Allow small vertical variance
                
                if same_line:
                    # Extend current object
                    current_object["text"] += char_text
                    current_object["bbox"]["x1"] = char.get('x1', current_object["bbox"]["x1"])
                else:
                    # Finalize current object and start new one
                    text_objects.append(current_object)
                    current_object = {
                        "text": char_text,
                        "bbox": {
                            "x0": char.get('x0', 0),
                            "y0": char.get('y0', 0),
                            "x1": char.get('x1', 0),
                            "y1": char.get('y1', 0)
                        },
                        "font": char.get('fontname', ''),
                        "size": char.get('size', 0)
                    }
        
        # Add final object
        if current_object:
            text_objects.append(current_object)
        
        return text_objects
    
    def _determine_extraction_confidence(
        self, 
        text_length: int, 
        has_form_fields: bool, 
        page_count: int
    ) -> Tuple[str, float]:
        """
        Determine extraction method and confidence based on extracted content.
        
        Args:
            text_length: Total length of extracted text
            has_form_fields: Whether PDF contains form fields
            page_count: Number of pages processed
            
        Returns:
            Tuple of (extraction_method, confidence_score)
        """
        # High confidence: Native text with substantial content
        if text_length > 1000:  # Substantial text content
            if has_form_fields:
                return "form_annotations", self.confidence_levels["form_annotations"]
            else:
                return "native_pdf_text", self.confidence_levels["native_pdf_text"]
        
        # Medium confidence: Some text but limited
        elif text_length > 100:
            return "regex_patterns", self.confidence_levels["regex_patterns"]
        
        # Low confidence: Minimal or no text (likely scanned)
        else:
            logger.warning(
                f"Low text extraction ({text_length} chars) - document likely requires OCR"
            )
            return "fallback_required", self.confidence_levels["fallback_required"]
    
    def detect_form_fields(self, pdf_path: Union[str, Path, bytes]) -> Dict[str, Any]:
        """
        Detect fillable form fields in PDF using pdfforms.
        
        Args:
            pdf_path: Path to PDF file or bytes content
            
        Returns:
            Dictionary containing form field information and metadata
        """
        try:
            if isinstance(pdf_path, bytes):
                # pdfforms requires file path, so write bytes to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_file.write(pdf_path)
                    tmp_path = tmp_file.name
                
                try:
                    form_data = pdfforms.get_form_data(tmp_path)
                finally:
                    # Clean up temp file
                    Path(tmp_path).unlink(missing_ok=True)
            else:
                form_data = pdfforms.get_form_data(str(pdf_path))
            
            if not form_data:
                return {
                    "has_form_fields": False,
                    "field_count": 0,
                    "fields": [],
                    "detection_method": "pdfforms"
                }
            
            # Process form fields
            processed_fields = []
            for field_name, field_info in form_data.items():
                field_data = {
                    "name": field_name,
                    "type": field_info.get('type', 'unknown'),
                    "value": field_info.get('value', ''),
                    "required": field_info.get('required', False),
                    "readonly": field_info.get('readonly', False),
                    "coordinates": field_info.get('rect', [])
                }
                processed_fields.append(field_data)
            
            result = {
                "has_form_fields": True,
                "field_count": len(processed_fields),
                "fields": processed_fields,
                "detection_method": "pdfforms",
                "detection_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Detected {len(processed_fields)} form fields in PDF")
            return result
            
        except Exception as e:
            logger.warning(f"Form field detection failed: {e}")
            return {
                "has_form_fields": False,
                "field_count": 0,
                "fields": [],
                "detection_method": "pdfforms",
                "error": str(e)
            }
    
    def chunk_pdf(
        self, 
        pdf_path: Union[str, Path, bytes], 
        max_pages_per_chunk: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Split large PDF into smaller chunks for processing.
        
        Args:
            pdf_path: Path to PDF file or bytes content
            max_pages_per_chunk: Maximum pages per chunk (defaults to class setting)
            
        Returns:
            List of chunk information dictionaries
            
        Raises:
            ValueError: If PDF cannot be processed
        """
        chunk_size = max_pages_per_chunk or self.max_pages_per_chunk
        total_pages = self.get_page_count(pdf_path)
        
        if total_pages <= chunk_size:
            # Single chunk
            return [{
                "chunk_index": 0,
                "start_page": 1,
                "end_page": total_pages,
                "page_count": total_pages,
                "is_single_chunk": True
            }]
        
        # Multiple chunks needed
        chunks = []
        for i in range(0, total_pages, chunk_size):
            start_page = i + 1
            end_page = min(i + chunk_size, total_pages)
            
            chunks.append({
                "chunk_index": len(chunks),
                "start_page": start_page,
                "end_page": end_page,
                "page_count": end_page - start_page + 1,
                "is_single_chunk": False
            })
        
        logger.info(f"Split {total_pages} pages into {len(chunks)} chunks")
        return chunks
    
    def extract_chunk(
        self, 
        pdf_path: Union[str, Path, bytes], 
        chunk_info: Dict[str, Any],
        extract_coordinates: bool = True
    ) -> Dict[str, Any]:
        """
        Extract content from a specific PDF chunk.
        
        Args:
            pdf_path: Path to PDF file or bytes content
            chunk_info: Chunk information from chunk_pdf method
            extract_coordinates: Whether to extract coordinate information
            
        Returns:
            Extraction result for the specified chunk
        """
        start_page = chunk_info["start_page"] - 1  # Convert to 0-based indexing
        end_page = chunk_info["end_page"]
        
        result = self.extract_text_from_pdf(
            pdf_path=pdf_path,
            extract_coordinates=extract_coordinates,
            page_range=(start_page, end_page)
        )
        
        # Add chunk metadata
        result["chunk_info"] = chunk_info
        
        return result
    
    def validate_pdf(self, pdf_path: Union[str, Path, bytes]) -> Dict[str, Any]:
        """
        Validate PDF file and determine processing requirements.
        
        Args:
            pdf_path: Path to PDF file or bytes content
            
        Returns:
            Validation result with processing recommendations
        """
        try:
            page_count = self.get_page_count(pdf_path)
            
            # Quick content sample
            sample_result = self.extract_text_from_pdf(
                pdf_path=pdf_path,
                extract_coordinates=False,
                page_range=(0, min(3, page_count))  # Sample first 3 pages
            )
            
            form_fields = self.detect_form_fields(pdf_path)
            
            # Determine processing strategy
            confidence = sample_result["confidence"]
            needs_chunking = page_count > self.max_pages_per_chunk
            
            if confidence >= 0.8:
                strategy = "direct_extraction"
                recommended_method = "pdfplumber"
            elif confidence >= 0.5:
                strategy = "hybrid_extraction" 
                recommended_method = "pdfplumber_with_ocr_fallback"
            else:
                strategy = "ai_vision_required"
                recommended_method = "gemini_vision_api"
            
            return {
                "is_valid": True,
                "page_count": page_count,
                "needs_chunking": needs_chunking,
                "has_form_fields": form_fields["has_form_fields"],
                "form_field_count": form_fields["field_count"],
                "sample_confidence": confidence,
                "extraction_strategy": strategy,
                "recommended_method": recommended_method,
                "sample_text_length": sample_result["total_text_length"],
                "validation_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"PDF validation failed: {e}")
            return {
                "is_valid": False,
                "error": str(e),
                "validation_timestamp": datetime.now(timezone.utc).isoformat()
            }


# Global extractor instance
pdf_extractor = PDFExtractor()


def get_pdf_extractor() -> PDFExtractor:
    """
    Get the global PDF extractor instance.
    
    Returns:
        PDFExtractor instance for dependency injection
    """
    return pdf_extractor