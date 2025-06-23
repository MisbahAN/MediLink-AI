import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF

from ..ai.openai import OpenAI
from ..lib.context import Context
from ..lib.step import Step
from ..lib.util import write_json, read_json, parse_json, read_file

logger = logging.getLogger(__name__)


class Fill(Step):
    def __init__(self):
        super().__init__("Fill", "Fill PDF forms with AI")
        self.openai = OpenAI()

    def run(self, context: Context):
        logger.info(f"Starting fill step for {len(context.data)} data entries")

        for i, data in enumerate(context.data, 1):
            logger.info(
                f"Processing fill for data entry {i}/{len(context.data)}: {data.name}")

            try:
                name = data.name
                widgets_path = Path(data.output_dir).joinpath("widgets.json")
                referral_package_path = Path(
                    data.output_dir).joinpath("referral_package.md")

                input_path = Path(data.input_dir).joinpath(
                    "prior_authorization.pdf")
                output_path = Path(data.output_dir).joinpath(
                    "prior_authorization_filled.pdf")
                # Check if required files exist
                if not widgets_path.exists():
                    logger.error(f"  Widgets file not found: {widgets_path}")
                    continue

                # Read the original widgets structure and filled information
                widgets = read_json(widgets_path)
                referral_package_content = read_file(referral_package_path)

                # Use OpenAI to populate widgets with information from pa_filled.md
                populated_response = self._populate_widgets_with_ai(
                    name, referral_package_content, widgets)
                write_json(widgets_path, populated_response)

                # Extract field values for PDF filling
                field_values = self._extract_field_values(populated_response)

                # Fill the actual PDF form
                self._fill_pdf(str(input_path), field_values, str(output_path))

            except Exception as e:
                logger.error(
                    f"  Error during fill processing for {data.name}: {str(e)}")
                raise

        logger.info("Fill step completed for all data entries")

    def _populate_widgets_with_ai(self, name: str, referral_package_content: str, widgets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use OpenAI to populate widget values based on pa_filled.md content"""
        try:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "widget_population_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "populated_widgets": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "field_name": {"type": "string"},
                                        "field_type": {"type": "string"},
                                        "suggested_value": {"type": ["string", "null"]},
                                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                        "reasoning": {"type": "string"},
                                    },
                                    "required": ["field_name", "field_type", "suggested_value", "confidence", "reasoning"],
                                    "additionalProperties": True
                                }
                            }
                        },
                        "additionalProperties": False
                    }
                }
            }

            prompt = self._get_population_prompt(
                name, referral_package_content, widgets)

            response = self.openai.client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical form data extraction assistant. Extract relevant information from filled medical forms and populate form widgets accurately."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format=response_format,
                # temperature=0.1
            )

            response_content = response.choices[0].message.content

            # Handle truncated responses
            if response.choices[0].finish_reason == 'length':
                if not response_content.strip().endswith('}'):
                    logger.error(f"Response was truncated: {response_content}")
                    raise Exception("Response was truncated")

            return parse_json(response_content)

        except Exception as e:
            logger.error(f"    Error in AI processing for {name}: {str(e)}")
            return {"user_name": name, "populated_widgets": widgets}

    def _get_population_prompt(self, name: str, referral_package_content: str, widgets: List[Dict[str, Any]]) -> str:
        """Generate prompt for populating widgets from referral_package.md content"""
        return f"""
        You are populating medical form widgets for user: {name}

        REFERRAL PACKAGE CONTENT:
        {referral_package_content}

        FORM WIDGETS TO POPULATE:
        {json.dumps(widgets, indent=2)}

        INSTRUCTIONS:
        1. Analyze the filled medical form content to extract relevant information
        2. For each widget, determine the most appropriate value based on the content
        3. Match information from the filled form to the corresponding form fields
        4. Provide confidence scores based on how well the information matches
        5. Include reasoning for each populated value
        6. The widgets are always in the same order as the fields in the prior authorization form, so you can use the order to match the information to the correct field if the field_name is not enough to identify the field.
        7. Try to infer the value of the field based on the context especially for yes/no fields.

        FIELD POPULATION RULES:
        - Text fields: Extract exact text values from the content
        - Checkboxes: Use "On" for yes/true/checked, "Off" for no/false/unchecked
        - Dropdowns/Comboboxes: Match to available options or provide closest match
        - Dates: Format as MM/DD/YYYY or MM/DD
        - Numbers: Extract numeric values
        - If information is not found, set suggested_value to null with low confidence
        - Always preserve the original widget structure in "original_widget"
        - Keep reasoning short and concise.
        - Make sure to respond with {len(widgets)} items.

        Format your response as JSON:
        {{
            "populated_widgets": [
                {{
                    "field_name": "exact_field_name",
                    "field_type": "text|checkbox|combobox|etc",
                    "suggested_value": "extracted_value_or_null",
                    "confidence": 0.95,
                    "reasoning": "Found in section X: specific reason"
                }}
            ]
        }}
        """

    def _extract_field_values(self, populated_response: Dict[str, Any]) -> Dict[str, str]:
        """Extract field values from populated response for PDF filling"""
        field_values = {}

        if "populated_widgets" in populated_response:
            for widget in populated_response["populated_widgets"]:
                field_name = widget.get("field_name")
                suggested_value = widget.get("suggested_value")
                confidence = widget.get("confidence", 0.0)

                if field_name and suggested_value is not None and confidence > 0.5:
                    field_values[field_name] = suggested_value

        return field_values

    def _fill_pdf(self, input_pdf_path: str, field_values: dict, output_pdf_path: str):
        """Fill PDF form fields with provided values"""
        try:
            doc = fitz.open(input_pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                widgets = page.widgets()

                for widget in widgets:
                    field_name = widget.field_name

                    if field_name in field_values:
                        value = field_values[field_name]

                        if widget.field_type == fitz.PDF_WIDGET_TYPE_TEXT:
                            widget.field_value = str(value)
                            widget.update()

                        elif widget.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                            widget.field_value = str(value).lower() in [
                                'on', 'true', '1', 'yes']
                            widget.update()

                        elif widget.field_type == fitz.PDF_WIDGET_TYPE_COMBOBOX:
                            widget.field_value = str(value)
                            widget.update()

                        elif widget.field_type == fitz.PDF_WIDGET_TYPE_LISTBOX:
                            widget.field_value = str(value)
                            widget.update()

            doc.save(output_pdf_path)
            doc.close()

        except Exception as e:
            logger.error(f"    Error filling PDF form: {str(e)}")
            try:
                shutil.copy2(input_pdf_path, output_pdf_path)
            except Exception as copy_error:
                logger.error(
                    f"    Failed to create fallback copy: {str(copy_error)}")
