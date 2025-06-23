# smart_form_filler_app.py

import os
import json
import re
from io import BytesIO
import fitz  # PyMuPDF
import streamlit as st
from google import genai
from google.genai import types

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
# Load API key
API_KEY = "AIzaSyDJ497eBkfxthH8T0cbXvYX7k3QHMsfdFo"  # <- YOUR API KEY HERE
MODEL_CTX = "gemini-2.5-flash"
MODEL_MAP = "gemini-2.5-flash"
PDF_PATH = "PA.pdf"
REFERRAL_PROMPT = (
    """
    "pdf1 is a referral package for a patient and pdf2 is a Prior Authorization form. pdf1 consists of all the details of the patient that needs to be extracted to fill Prior Authorization form. Some pages of Pdf2 consists of all the questions that needs to be answered inferring from the details in pdf1. Please extract the details and present answers to questions in Pdf2. I want answers to all the fields to pdf2 in a structured format."
    "Go through entire referral package in detail and try to aextract answers to as many questions in PA form as possible. "
    "Go through every page of referral package carefully and extract all the information by visually examining."
    "Only return the following JSON format in output"
    
    "Wrap all patient fields under a single top‐level object. For example:"
        Return output in this format:

        {
        "patient_info": {
            "First_Name": "...",
            "Last_Name": "...",
            …
        }
        }
    """
)

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)

# ── UTILITIES ───────────────────────────────────────────────────────────────────
def extract_json(text):
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found")
    raw = text[start:end+1]
    raw = re.sub(r'(?m)^\s*([A-Za-z0-9_]+)\s*:', r'"\1":', raw)
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    return raw

# Wrap referral context extraction code exactly

def extract_patient_info(referral_bytes, pa_bytes):
    """
    Runs the user-provided Gemini prompt to extract patient info.
    """
    pdf1 = types.Part.from_bytes(data=referral_bytes, mime_type='application/pdf')
    pdf2 = types.Part.from_bytes(data=pa_bytes, mime_type='application/pdf')
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            pdf1,
            pdf2,
            REFERRAL_PROMPT
        ]
    ).text
    print(response)
    raw = extract_json(response)
    return json.loads(raw)

# Wrap existing page-by-page logic into functions

def make_page_part(pdf_bytes, page_no):
    """
    Create a one-page PDF part from the given PDF bytes.
    Tries to copy the form page; on XRef errors, falls back to image-based PDF.
    """
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    # Attempt to copy with widgets
    try:
        dst = fitz.open()  # new empty PDF
        dst.insert_pdf(src, from_page=page_no-1, to_page=page_no-1)
        # remove any leftover widget annotations to avoid XRef errors
        for w in dst[0].widgets() or []:
            dst[0].delete_widget(w)
        buf = BytesIO()
        dst.save(buf)
        return types.Part.from_bytes(data=buf.getvalue(), mime_type="application/pdf")
    except Exception:
        # Fallback: render page as image PDF
        page = src[page_no-1]
        pix = page.get_pixmap()
        new_pdf = fitz.open()
        rect = page.rect
        new_page = new_pdf.new_page(width=rect.width, height=rect.height)
        new_page.insert_image(rect, pixmap=pix)
        buf = BytesIO()
        new_pdf.save(buf)
        return types.Part.from_bytes(data=buf.getvalue(), mime_type="application/pdf")


def extract_fields_with_positions(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    fields = []
    for page_num, page in enumerate(doc, start=1):
        for w in page.widgets() or []:
            fields.append({
                "name":  w.field_name,
                "type":  "checkbox" if w.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX else "text",
                "value": w.field_value,
                "page":  page_num,
                "rect":  list(map(float, w.rect))
            })
    return fields

# Main Streamlit app
st.title("MedFill - Automate Insurance Forms")

pa_file = st.file_uploader("Upload PA Form PDF", type=["pdf"])
ref_file = st.file_uploader("Upload Referral Package PDF", type=["pdf"])

if st.button("Process and Fill"):  # Button triggers entire workflow
    if not pa_file or not ref_file:
        st.error("Please upload both the PA form and the referral package.")
        st.stop()

    # Read file bytes
    pa_bytes = pa_file.read()
    ref_bytes = ref_file.read()

    # 1) Extract patient info
    try:
        patient_info = extract_patient_info(ref_bytes, pa_bytes)
    except Exception as e:
        st.error(f"Failed to extract patient info: {e}")
        st.stop()

    # 2) Extract PA fields
    fields = extract_fields_with_positions(pa_bytes)
    fields_by_page = {}
    for f in fields:
        fields_by_page.setdefault(f["page"], []).append({
            "id":   f["name"],
            "type": f["type"],
            "rect": f["rect"]
        })

    # 3) Page-by-page context and mapping
    field_context_by_page = {}
    field_mapping = {}

    for page_no, page_fields in sorted(fields_by_page.items()):
        # Context extraction (user's prompt)
        page_part = make_page_part(pa_bytes, page_no)
        prompt_ctx = f"""
            You’re annotating page {page_no} of a medical form.
            Given:
            - This page’s form fields (id,type,rect).
            - Actual Prior Authorization form.

            For each field:
            Add question and context fields to already existing field objects
            Move sequentially through the page for each field along with fields info attached.

            Generate the best possible context around that field object in 25 words. Provide a proper context to actually give more insight into what the question actually is. If the question needs any background info to make sense, add that into the context.
            Getting the correct context for correct field is extremely important for correct mapping.

            Return a JSON object mapping field IDs to:
            - For each form field object add its question corresponding to it i.e what is the question asked for CB1 or T1 for all the fields
            - Also indicate the context in which the question is asked. Sometimes, the question itself does not give a lot of context. For example if a question is First Name it should be known if it is the patients name or the insurer's. Also if a question is a sub question of another question, it should be known the context of it.
            - Only add context and question as additional fields to each object
            - Return a JSON object mapping field names (e.g., "T1", "CB1") to their filled values, as given with an additional element in the object as the question corresponding to it.

            Only output valid JSON.

            Each output JSON object should only contain the fields - name, page, question, context in the following format
            {{"name": "T67", "page": 2, "question": "","context": "" }}

            Return all the objects as a JSON inside {{}}

            Here are the fields:
            {json.dumps(page_fields, indent=2)}
            """
        resp_ctx = client.models.generate_content(
            model=MODEL_CTX,
            contents=[page_part, prompt_ctx]
        ).text
        print(resp_ctx)
        ctx = json.loads(extract_json(resp_ctx))
        field_context_by_page[page_no] = ctx

        # Mapping extraction (user's prompt)
        prompt_map = f"""
        You’re filling page {page_no} of the attached Prior Authorization form.
        Given:
        1. This page’s form fields (id,type).
        2. Detailed patient info.
        3. This page’s field context mapping.
        You are a smart form-filling assistant.

        Carefully look into the context and question for each field. Then see if the patient info has information for it. If it has good create mapping. But if it does not have do not create a mapping for it.

        Be intelligent in creating mappings. See the context and decide on the mapping. Make full use of understanding context.
        Do not map inappropriate content with inappropriate field.

            Cleverly match the right patient info to applicable field. Use the info from patient info to map form fields based on questions and context from structured info. Use:
            - Text for text fields
            - `true` / `false` for checkboxes
            - Leave irrelevant fields blank or `false`
            - Do Not fill in the fields for which information is not present in patient info

            A name field should be filled with name, an address field with address etc.
            Do not add the field in mapping if info about it is not present.

        Return valid JSON {{field_id:value}}.
        
        --- PATIENT INFO ---
        {json.dumps(patient_info, indent=2)}
        --- FIELD CONTEXT ---
        {json.dumps(field_context_by_page[page_no], indent=2)}
        """
        resp_map = client.models.generate_content(
            model=MODEL_MAP,
            contents=[page_part, prompt_map]
        ).text
        print(resp_map)
        mapping = json.loads(extract_json(resp_map))
        field_mapping.update(mapping)

    # 4) Fill PDF
    doc = fitz.open(stream=pa_bytes, filetype="pdf")
    for page in doc:
        for w in page.widgets() or []:
            fid = w.field_name
            if fid in field_mapping:
                val = field_mapping[fid]
                if w.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                    w.field_value = "Yes" if bool(val) else "Off"
                else:
                    w.field_value = str(val)
                w.update()
    out_buf = BytesIO()
    doc.save(out_buf)

    # Download filled PDF
    st.download_button(
        label="Download Filled PA Form",
        data=out_buf.getvalue(),
        file_name="filled_PA.pdf",
        mime="application/pdf"
    )