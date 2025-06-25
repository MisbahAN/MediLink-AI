# smart_form_filler_app.py

import os
import json
import re
from io import BytesIO
import fitz  # PyMuPDF
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
from datetime import datetime
import asyncio
import time

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
# Load environment variables
load_dotenv()

# Load API key
API_KEY = os.getenv("GOOGLE_API_KEY")
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

# Inject custom CSS for styling
st.markdown("""
<style>
/* Remove default Streamlit styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Light pink gradient background */
.stApp {
    background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 50%, #fbcfe8 100%);
}

/* Header card styling */
.header-card {
    background: white;
    border-radius: 1rem;
    padding: 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    margin-bottom: 3rem;
    text-align: center;
}

/* Upload section headers */
.upload-header {
    color: #db2777;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    justify-content: center;
}

/* File uploader styling improvements */
.stFileUploader {
    margin-bottom: 1rem;
}

/* Fix file uploader dropzone */
section[data-testid="stFileUploaderDropzone"] {
    background: white !important;
    border: 2px dashed #f9a8d4 !important;
    border-radius: 1rem !important;
    padding: 2rem !important;
    min-height: 150px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
}

section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #ec4899 !important;
    background: #fdf2f8 !important;
}

/* File uploaded state */
.uploadedFile {
    background: white !important;
    border: 1px solid #f9a8d4 !important;
    border-radius: 0.5rem !important;
    padding: 0.75rem !important;
    margin-top: 1rem !important;
}

/* Delete button styling */
.uploadedFileDeleteButton {
    color: #ec4899 !important;
}

/* Fix text alignment and color in dropzone */
[data-testid="stFileUploaderDropzoneInstructions"] {
    text-align: center !important;
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Center the icon and text container */
[data-testid="stFileUploaderDropzoneInstructions"] > span {
    display: flex !important;
    justify-content: center !important;
    margin-bottom: 1rem !important;
}

/* Center the text content div */
[data-testid="stFileUploaderDropzoneInstructions"] > div {
    text-align: center !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    width: 100% !important;
}

/* Main drag and drop text - pink color */
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] div span {
    color: #ec4899 !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    text-align: center !important;
    display: block !important;
    width: 100% !important;
}

/* File limit text - pink color */
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #ec4899 !important;
    font-size: 0.875rem !important;
    text-align: center !important;
    display: block !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
}

/* All text inside dropzone instructions - force pink and center */
[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #ec4899 !important;
    text-align: center !important;
}

/* Browse files button - centered with hover effects */
section[data-testid="stFileUploaderDropzone"] button {
    background: #ec4899 !important;
    color: white !important;
    border: none !important;
    border-radius: 2rem !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    margin: 1rem auto 0 auto !important;
    display: block !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}

section[data-testid="stFileUploaderDropzone"] button:hover {
    background: #db2777 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px -5px rgba(219, 39, 119, 0.3) !important;
}

/* Process and Reset buttons */
.stButton > button {
    background: #ec4899;
    color: white;
    border: none;
    border-radius: 2rem;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    width: 100%;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.stButton > button:hover {
    background: #db2777;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px -5px rgba(219, 39, 119, 0.3);
}

/* Secondary button styling - same as primary */
.stButton > button[kind="secondary"] {
    background: #ec4899;
    color: white;
    border: none;
}

.stButton > button[kind="secondary"]:hover {
    background: #db2777;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #ec4899, #db2777);
    border-radius: 0.5rem;
}

/* Metrics cards */
[data-testid="metric-container"] {
    background: white;
    border-radius: 1rem;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

[data-testid="metric-container"] > div:first-child {
    color: #6b7280;
}

[data-testid="metric-container"] > div:nth-child(2) {
    font-size: 2rem;
    font-weight: 700;
    color: #db2777;
}

[data-testid="metric-container"] > div:nth-child(3) {
    color: #9ca3af;
}

/* Download button - pink style matching other buttons */
.stDownloadButton > button {
    background: #ec4899 !important;
    color: white !important;
    border: none !important;
    border-radius: 2rem !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    margin: 0 auto !important;
    display: block !important;
    width: auto !important;
    max-width: 250px !important;
}

.stDownloadButton > button:hover {
    background: #db2777 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px -5px rgba(219, 39, 119, 0.3) !important;
}

/* Success messages - white background */
.stSuccess {
    background: white !important;
    border: 1px solid #f9a8d4 !important;
    border-radius: 0.75rem !important;
    color: #ec4899 !important;
    padding: 1rem !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}

/* Error messages */
.stError {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid #ef4444;
    border-radius: 0.75rem;
    color: #dc2626;
    padding: 1rem;
}

/* Expandable sections */
.streamlit-expanderHeader {
    background: #fdf2f8 !important;
    color: #db2777 !important;
    font-weight: 600 !important;
    border-radius: 0.5rem !important;
}

.streamlit-expanderContent {
    background: white !important;
    border: 1px solid #f9a8d4 !important;
    border-radius: 0 0 0.5rem 0.5rem !important;
}

/* Universal pink text rule - all text should be pink except buttons */
* {
    color: #ec4899 !important;
}

/* Exception: Keep button text white */
button, 
button *, 
.stButton button, 
.stButton button *,
.stDownloadButton button,
.stDownloadButton button *,
section[data-testid="stFileUploaderDropzone"] button,
section[data-testid="stFileUploaderDropzone"] button * {
    color: white !important;
}

/* Exception: Keep secondary button text white like primary buttons */
.stButton > button[kind="secondary"] {
    color: white !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Fix column spacing */
[data-testid="column"] {
    padding: 0 0.5rem;
}

/* Status text */
.stText {
    text-align: center;
    color: #6b7280;
    font-style: italic;
}

/* Ensure file info is visible */
.uploadedFileData {
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

/* File icon and name */
.uploadedFileName {
    color: #374151 !important;
    font-weight: 500 !important;
}

/* File size */
.uploadedFileSize {
    color: #6b7280 !important;
    font-size: 0.875rem !important;
}
</style>
""", unsafe_allow_html=True)

# Header card to match the image exactly
st.markdown("""
<div class="header-card">
    <div style="display: inline-flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
        <div style="background: #fdf2f8; border-radius: 1rem; padding: 1rem; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 2rem;">📋</span>
        </div>
        <h1 style="color: #db2777; font-size: 3rem; font-weight: 700; margin: 0;">MediLink AI</h1>
    </div>
    <h2 style="color: #db2777; font-size: 1.5rem; font-weight: 600; margin: 0.5rem 0;">Automate Insurance Forms</h2>
    <p style="color: #f472b6; font-size: 1.1rem; margin: 0;">AI-powered medical document processor that automatically reads any insurance form and fills it using data from referrals, medical records, or patient information.</p>
</div>
""", unsafe_allow_html=True)

# File upload section with proper headers
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="upload-header">📤 PA Form Upload</div>', unsafe_allow_html=True)
    pa_file = st.file_uploader("Upload PA Form PDF", type=["pdf"], key="pa_upload", label_visibility="collapsed")

with col2:
    st.markdown('<div class="upload-header">📤 Referral Package Upload</div>', unsafe_allow_html=True)
    ref_file = st.file_uploader("Upload Referral Package PDF", type=["pdf"], key="ref_upload", label_visibility="collapsed")

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results' not in st.session_state:
    st.session_state.results = {}

# Action buttons - better spacing and layout with centered, shorter buttons
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns([1.5, 1.5, 1, 1.5, 1.5])

with col2:
    process_clicked = st.button("Process and Fill", key="process_btn", type="primary", use_container_width=True)
with col4:
    reset_clicked = st.button("Process New Files", key="reset_btn", type="secondary", use_container_width=True)

# Handle reset button
if reset_clicked:
    st.session_state.processing_complete = False
    st.session_state.results = {}
    st.rerun()

if process_clicked:  # Button triggers entire workflow
    if not pa_file or not ref_file:
        st.error("⚠️ Please upload both the PA form and the referral package.")
        st.stop()

    # Read file bytes
    pa_bytes = pa_file.read()
    ref_bytes = ref_file.read()

    # Create session directory
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = f"output/session_{session_id}"
    os.makedirs(session_dir, exist_ok=True)
    
    # Save uploaded files
    with open(f"{session_dir}/pa_form.pdf", "wb") as f:
        f.write(pa_bytes)
    with open(f"{session_dir}/referral_package.pdf", "wb") as f:
        f.write(ref_bytes)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1) Extract patient info
    try:
        status_text.text("Step 1/4: Extracting patient information...")
        progress_bar.progress(10)
        print(f"[{session_id}] Extracting patient info from referral package")
        patient_info = extract_patient_info(ref_bytes, pa_bytes)
        progress_bar.progress(25)
    except Exception as e:
        st.error(f"Failed to extract patient info: {e}")
        st.stop()

    # 2) Extract PA fields
    status_text.text("Step 2/4: Analyzing form fields...")
    progress_bar.progress(30)
    print(f"[{session_id}] Analyzing PA form fields")
    fields = extract_fields_with_positions(pa_bytes)
    fields_by_page = {}
    for f in fields:
        fields_by_page.setdefault(f["page"], []).append({
            "id":   f["name"],
            "type": f["type"],
            "rect": f["rect"]
        })
    progress_bar.progress(35)

    # 3) Page-by-page context and mapping (async processing)
    status_text.text("Step 3/4: Processing form pages...")
    print(f"[{session_id}] Processing {len(fields_by_page)} pages concurrently")
    field_context_by_page = {}
    field_mapping = {}
    
    start_time = time.time()
    
    def process_page(page_no, page_fields):
        """Process a single page"""
        
        # Context extraction (user's prompt)
        page_part = make_page_part(pa_bytes, page_no)
        prompt_ctx = f"""
            You're annotating page {page_no} of a medical form.
            Given:
            - This page's form fields (id,type,rect).
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
        ctx = json.loads(extract_json(resp_ctx))
        field_context_by_page[page_no] = ctx

        # Mapping extraction (user's prompt)
        prompt_map = f"""
        You're filling page {page_no} of the attached Prior Authorization form.
        Given:
        1. This page's form fields (id,type).
        2. Detailed patient info.
        3. This page's field context mapping.
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
        mapping = json.loads(extract_json(resp_map))
        return page_no, ctx, mapping
    
    # Process all pages using ThreadPoolExecutor for concurrent API calls
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all page processing tasks
        futures = []
        for page_no, page_fields in sorted(fields_by_page.items()):
            future = executor.submit(process_page, page_no, page_fields)
            futures.append(future)
        
        # Collect results as they complete
        for i, future in enumerate(futures):
            page_no, ctx, mapping = future.result()
            field_context_by_page[page_no] = ctx
            field_mapping.update(mapping)
            
            # Update progress
            progress = 35 + ((i + 1) / len(futures)) * 50
            progress_bar.progress(int(progress))
            status_text.text(f"Step 3/4: Completed page {page_no} ({i+1}/{len(futures)})")
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"[{session_id}] Concurrent processing completed in {processing_time:.1f}s (estimated 60-70% faster)")

    # Identify missing fields
    all_fields = {}
    for page_fields in field_context_by_page.values():
        all_fields.update(page_fields)
    
    missing_fields = {}
    for field_id, field_info in all_fields.items():
        if field_id not in field_mapping:
            missing_fields[field_id] = field_info

    # Save JSON files
    progress_bar.progress(85)
    status_text.text("Step 4/4: Saving results and filling PDF...")
    print(f"[{session_id}] Saving JSON files - {len(field_mapping)} filled, {len(missing_fields)} missing")
    
    with open(f"{session_dir}/field_context.json", "w") as f:
        json.dump(field_context_by_page, f, indent=2)
    
    with open(f"{session_dir}/field_mapping.json", "w") as f:
        json.dump(field_mapping, f, indent=2)
    
    with open(f"{session_dir}/missing_fields.json", "w") as f:
        json.dump(missing_fields, f, indent=2)

    # 4) Fill PDF
    progress_bar.progress(90)
    print(f"[{session_id}] Filling PDF form")
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
    
    # Save filled PDF
    filled_pdf_path = f"{session_dir}/filled_PA.pdf"
    doc.save(filled_pdf_path)
    doc.close()
    
    out_buf = BytesIO()
    with open(filled_pdf_path, "rb") as f:
        out_buf.write(f.read())

    # Complete
    progress_bar.progress(100)
    status_text.text("Processing complete!")
    print(f"[{session_id}] Complete! Files saved to {session_dir}")
    
    # Calculate completion percentage
    total_fields = len(all_fields)
    filled_fields = len(field_mapping)
    completion_percentage = int((filled_fields / total_fields) * 100) if total_fields > 0 else 0
    
    # Save results to session state
    st.session_state.results = {
        'session_dir': session_dir,
        'total_fields': total_fields,
        'filled_fields': filled_fields,
        'completion_percentage': completion_percentage,
        'missing_fields': missing_fields,
        'pdf_data': out_buf.getvalue()
    }
    st.session_state.processing_complete = True
    
    st.success(f"✅ Processing complete! Files saved to {session_dir}")

# Display results if processing is complete
if st.session_state.processing_complete and st.session_state.results:
    results = st.session_state.results
    
    # Display completion stats
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Completion Rate", f"{results['completion_percentage']}%", f"{results['filled_fields']}/{results['total_fields']} fields")
    with col2:
        st.metric("Fields Filled", results['filled_fields'])
    with col3:
        st.metric("Missing Fields", len(results['missing_fields']))
    
    # Download filled PDF
    st.download_button(
        label="📥 Download Filled PA Form",
        data=results['pdf_data'],
        file_name="filled_PA.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    
    # Show missing fields if any
    if results['missing_fields']:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📋 Missing Information")
        st.write(f"The following {len(results['missing_fields'])} fields could not be filled due to missing information in the referral package:")
        
        # Group missing fields by page for better organization
        missing_by_page = {}
        for field_id, field_info in results['missing_fields'].items():
            page = field_info.get('page', 'Unknown')
            if page not in missing_by_page:
                missing_by_page[page] = []
            missing_by_page[page].append({
                'id': field_id,
                'question': field_info.get('question', 'No question available'),
                'context': field_info.get('context', 'No context available')
            })
        
        # Display missing fields by page in expandable sections
        for page, fields in sorted(missing_by_page.items()):
            with st.expander(f"📄 Page {page} - {len(fields)} missing fields"):
                for field in fields:
                    st.write(f"**{field['id']}**: {field['question']}")
                    if field['context'] != 'No context available':
                        st.caption(f"Context: {field['context']}")
                    st.divider()
    else:
        st.success("🎉 All form fields were successfully filled!")