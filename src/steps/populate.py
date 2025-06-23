from ..lib.step import Step
from ..lib.context import Context
from pathlib import Path
from ..lib.util import parse_json, read_json, read_file, write_json
from ..ai.openai import OpenAI
from typing import List, Dict, Any

class Populate(Step):
    def __init__(self):
        super().__init__("Populate", "Add more information to the widgets")

    def run(self, context: Context):
        openai = OpenAI()
        for data in context.data:
            widgets_path = Path(data.output_dir).joinpath("widgets.json")
            prior_authorization_path = Path(
                data.output_dir).joinpath("prior_authorization.md")

            prior_authorization_content = read_file(prior_authorization_path)
            widgets = read_json(widgets_path)
            widgets = [{"field_name": widget["field_name"],
                        "field_label": widget["field_label"]} for widget in widgets]
                        
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
                                        "field_name": {"type": "string", "description": "The name of the field"},
                                        "context": {"type": "string", "description": "The context of the field"},
                                    },
                                    "required": ["field_name", "context"],
                                    "additionalProperties": True
                                }
                            }
                        },
                        "additionalProperties": False
                    }
                }
            }


            response = openai.client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": "You are a medical data extraction assistant. Try to add more information to the pdf widgets based on the markdown file. Return the response JSON."},
                    {"role": "user", "content": self._get_prompt(
                        prior_authorization_content, widgets)}
                ],
                response_format=response_format
            )
            response_content = parse_json(response.choices[0].message.content)
            response_widgets = response_content["populated_widgets"]

            # O(n^2) but does not matter since number of widgets is few hundreds at most
            for item in response_widgets:
                for widget in widgets:
                    if widget["field_name"] == item["field_name"]:
                        widget["context"] = item["context"]
            write_json(widgets_path, widgets)

    def _get_prompt(self, prior_authorization_content: str, widgets: List[Dict[str, Any]]) -> str:
        return f"""
        PRIOR AUTHORIZATION FORM:
        {prior_authorization_content}

        WIDGETS:
        {widgets}

        INSTRUCTIONS:
        1. Add more information to each widget based on the prior authorization form.
        2. The widgets are always in the same order as the fields in the prior authorization form, so you can use the order to match the information to the correct field if the field_name is not enough to identify the field.
        3. RESPOND WITH THE JSON
        4. {len(widgets)} items in the JSON.
        5. Context should be a short description of the field.
        6. Make sure to add context for all the widgets especially for yes/no fields i.e. checkbox fields.
        """
