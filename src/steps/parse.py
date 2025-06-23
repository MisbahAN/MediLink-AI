#!/usr/bin/env python3
"""
Parse Step
Extracts widgets from PDF files following the Step interface
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import fitz

from ..lib.context import Context, Data
from ..lib.step import Step
from ..lib.util import write_json

logger = logging.getLogger(__name__)


class Parse(Step):
    def __init__(self):
        super().__init__("Parse", "Extract widgets from PDF files and save to JSON")

    def run(self, context: Context):
        """Run the parse step with patient data"""
        logger.info(f"Starting parse step for {len(context.data)} data entries")
        
        for i, data in enumerate(context.data, 1):
            logger.info(f"Processing data entry {i}/{len(context.data)}: {data.name}")
            self._parse(context, data)
        
        logger.info("Parse step completed for all data entries")

    def _parse(self, context: Context, data: Data):
        """Parse widgets for a single patient"""
        try:
            prior_authorization_path = Path(data.input_dir).joinpath("prior_authorization.pdf")
            
            if not prior_authorization_path.exists():
                logger.error(f"  PDF file not found: {prior_authorization_path}")
                raise FileNotFoundError(f"PDF file not found: {prior_authorization_path}")
            
            widgets = self._extract_widgets(prior_authorization_path)
            
            output_path = Path(data.output_dir).joinpath("widgets.json")
            write_json(output_path, widgets)
            
        except Exception as e:
            logger.error(f"  Error parsing data for {data.name}: {str(e)}")
            raise

    def _extract_widgets(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract widgets from a PDF file and sort by reading order"""
        widgets = []
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_widgets = page.widgets()
                
                for widget in page_widgets:
                    try:
                        # Get widget coordinates
                        rect = getattr(widget, 'rect', None)
                        x0, y0, x1, y1 = rect if rect else (0, 0, 0, 0)
                        
                        widget_data = {
                            # Core field properties
                            'field_name': getattr(widget, 'field_name', None),
                            'field_value': getattr(widget, 'field_value', None),
                            'field_label': getattr(widget, 'field_label', None),
                            # Signature and structure
                            'is_signed': getattr(widget, 'is_signed', None),
                            # Position data for sorting
                            'page_num': page_num,
                            'x0': x0,
                            'y0': y0,
                            'x1': x1,
                            'y1': y1,
                        }
                        
                        widgets.append(widget_data)
                        
                    except Exception as e:
                        logger.warning(f"Error processing widget on page {page_num + 1}: {str(e)}")
                        continue
            
            doc.close()
            
            # Sort widgets by reading order: page, then top to bottom, then left to right
            # Use a small tolerance for Y coordinates to handle widgets on the same logical row
            def sort_key(widget):
                # Round Y coordinate to nearest 5 pixels to group widgets on same row
                y_rounded = round(widget['y0'] / 5) * 5
                
                # Priority boost for start date fields
                priority = 0
                if widget['field_label'] and 'start date' in widget['field_label'].lower():
                    priority = -1  # Higher priority (smaller number)
                
                return (widget['page_num'], priority, y_rounded, widget['x0'])
            
            widgets.sort(key=sort_key)
            
            logger.info(f"Extracted and sorted {len(widgets)} widgets from {file_path}")
            
        except Exception as e:
            logger.error(f"Error extracting widgets from {file_path}: {str(e)}")
            raise

        return widgets
