# Prior Authorization & Referral Document Analysis Guide

## Overview

This document provides a comprehensive guide for understanding and processing Prior Authorization (PA) forms and medical referral packets. It's based on analysis of real-world examples and designed to help with automated form filling systems.

## Document Types

### 1. Prior Authorization Forms

PA forms are insurance-required documents that must be completed before certain treatments or medications can be approved.

### 2. Referral Packets

Medical referral packets contain patient information, medical history, treatment plans, and supporting documentation from healthcare providers.

---

## Prior Authorization Form Structures

### Common PA Form Types Encountered

#### 1. Aetna Medicare Rituximab PA Form (GR-68535-3)

- **Pages**: 5 pages
- **Form ID**: GR-68535-3 (1-25)
- **Target**: Medicare Advantage Part B
- **Medication**: Riabni®, Rituxan®, Ruxience, Truxima (rituximab variants)

**Key Sections:**

- **Page 1**: Contact information and routing details for different plan types
- **Page 2**: Patient, Insurance, Prescriber, Dispensing Provider, Product, Diagnosis info
- **Page 3-4**: Clinical information with extensive checkbox logic for different conditions
- **Page 5**: Continuation requests and acknowledgment

#### 2. Aetna Skyrizi PA Form (2060 9-23)

- **Pages**: 2 pages
- **Medication**: Skyrizi (risankizumab-rzaa)
- **Target**: Medicare Advantage Part B

**Key Sections:**

- Patient demographics with checkbox grids
- Clinical questions about Crohn's disease
- Step therapy requirements
- Tuberculosis screening requirements

#### 3. Anthem Vyepti PA Form (VABCBS-CD-061331-24)

- **Pages**: 3 pages
- **Medication**: Vyepti® (Eptinezumab-jmmr)
- **Target**: HealthKeepers/Anthem Medicaid

**Key Sections:**

- Antimigraine agents classification
- Preferred vs non-preferred drug logic
- Step edit requirements
- Renewal criteria with specific metrics

---

## Standard PA Form Field Categories

### A. Patient Information (Always Present)

```
- First Name: [Text field]
- Last Name: [Text field]
- Date of Birth: [MM/DD/YYYY format]
- Address: [Street, City, State, ZIP]
- Phone Numbers: [Home, Work, Cell]
- Current Weight: [lbs or kgs]
- Height: [inches or cms]
- Allergies: [Text field]
```

### B. Insurance Information (Always Present)

```
- Member ID #: [Alphanumeric]
- Group #: [Alphanumeric]
- Insured: [Relationship to patient]
- Other Coverage: [Yes/No with details]
- Carrier Name: [If other coverage exists]
```

### C. Prescriber Information (Always Present)

```
- First/Last Name: [Text fields]
- Credentials: [M.D., D.O., N.P., P.A. - Checkbox]
- Address: [Street, City, State, ZIP]
- Phone/Fax: [Numbers]
- License Numbers: [St Lic #, NPI #, DEA #, UPIN]
- Provider Email: [Email address]
- Office Contact: [Name and phone]
```

### D. Dispensing Provider Information

```
- Place of Administration: [Checkboxes for various locations]
- Administration Codes: [CPT codes]
- Pharmacy Information: [Name, address, TIN, PIN, NPI]
```

### E. Product Information (Always Present)

```
- Drug Name: [Specific medication]
- Dose: [Amount and frequency]
- Directions for Use: [Text field]
- HCPCS Code: [Medical billing code]
```

### F. Diagnosis Information (Always Present)

```
- Primary ICD Code: [ICD-10 format]
- Other ICD Codes: [Additional diagnoses]
```

### G. Clinical Information (Most Complex Section)

This varies significantly by medication but common patterns include:

#### Trial and Failure Logic

```
- Has patient tried preferred alternatives? [Yes/No]
- List medications tried: [Text fields with dates]
- Reason for failure: [Checkboxes: ineffective, not tolerated, contraindicated]
- Duration of trial: [Time periods]
```

#### Adverse Reactions

```
- Has patient had adverse reactions? [Yes/No]
- Nature of adverse reaction: [Text field]
- Date of adverse reaction: [MM/DD/YYYY]
```

#### Specific Condition Requirements

Varies by medication class (biologics, antimigraines, etc.)

---

## Referral Packet Structures

### Common Components in Referral Packets

#### 1. Fax Cover Sheet

- **Purpose**: Identifies sender, recipient, reason for referral
- **Key Fields**: Date, patient name, provider info, brief clinical note

#### 2. Patient Demographics Sheet

```
- Full Name: [Last, First]
- Date of Birth: [MM/DD/YYYY]
- Medical Record Number (MRN): [Alphanumeric]
- Address: [Complete address]
- Phone Numbers: [Multiple types]
- Emergency Contact: [Name, relationship, phone]
- Insurance Information: [Primary/Secondary coverage]
```

#### 3. Medical History Documentation

```
- Chief Complaint: [Brief description]
- History of Present Illness: [Detailed narrative]
- Past Medical History: [List of conditions]
- Medications: [Current medication list with dosages]
- Allergies: [Known allergies and reactions]
- Social History: [Relevant social factors]
- Family History: [Relevant family medical history]
```

#### 4. Clinical Notes

```
- Progress Notes: [Dated entries from healthcare providers]
- Assessment and Plan: [Clinical reasoning and treatment plan]
- Vital Signs: [Blood pressure, weight, temperature, etc.]
- Physical Exam Findings: [Systematic examination results]
```

#### 5. Treatment Plans and Orders

```
- Medication Orders: [Detailed prescription information]
- Infusion Orders: [For IV medications like rituximab]
- Follow-up Instructions: [Next steps and appointments]
- Laboratory Orders: [Required tests and monitoring]
```

#### 6. Supporting Documentation

```
- Lab Results: [Blood work, cultures, specialized tests]
- Imaging Reports: [MRI, CT, X-ray results]
- Consultation Notes: [Specialist recommendations]
- Insurance Coverage Information: [Plan details and authorizations]
```

---

## Data Mapping Patterns

### Patient Identification

**PA Form Fields** → **Referral Sources**

- `First Name` ← `Patient Demographics` (various formats)
- `Last Name` ← `Patient Demographics`
- `DOB` ← `Patient Demographics` (format: MM/DD/YYYY)
- `Member ID` ← `Insurance Coverage Information`

### Clinical Information

**PA Form Fields** → **Referral Sources**

- `Primary ICD Code` ← `Assessment/Diagnosis` sections
- `Diagnosis Description` ← `Clinical Notes` and `Progress Notes`
- `Current Medications` ← `Medication Lists` and `Treatment Plans`
- `Previous Treatments` ← `Medical History` and `Progress Notes`

### Provider Information

**PA Form Fields** → **Referral Sources**

- `Prescriber Name` ← `Provider Contact Information` or `Signature blocks`
- `NPI Number` ← `Provider credentials` in clinical notes
- `Office Contact` ← `Fax cover sheets` or `Contact information`

---

## Common Challenges & Edge Cases

### 1. Multiple Date Formats

- MM/DD/YYYY (most common in US forms)
- DD/MM/YYYY (some international formats)
- YYYY-MM-DD (database/system formats)
- Written formats: "May 22, 2024"

### 2. Name Variations

- Last, First format vs First Last format
- Middle names or initials
- Suffixes (Jr., Sr., III)
- Hyphenated names
- Professional titles mixed with names

### 3. Medication Name Variations

- Brand names vs generic names
- Different formulations (tablets, injections, etc.)
- Dosage information embedded in names
- Biosimilar medications with similar names

### 4. Provider Identification

- Multiple providers in same referral
- Nurse practitioners vs physicians
- Referring provider vs treating provider
- Contact information scattered across documents

### 5. Insurance Information Complexity

- Multiple insurance plans (primary/secondary)
- Different member ID formats
- Group numbers vs individual numbers
- Plan name variations

---

## Field Mapping Confidence Scoring

### High Confidence (>90%)

- Exact text matches for standard fields
- Structured data in consistent formats
- Clear field labels and values

### Medium Confidence (70-90%)

- Fuzzy text matches
- Data requiring format conversion
- Fields found in expected locations but with variations

### Low Confidence (<70%)

- Data requiring inference
- Multiple possible matches
- Missing or unclear source information

---

## Validation Rules

### Required Fields (Cannot be empty)

- Patient Name
- Date of Birth
- Member ID
- Prescriber Name
- Medication Name
- Primary Diagnosis

### Format Validations

- DOB: Must be valid date, patient must be reasonable age
- Phone numbers: Various formats accepted
- Zip codes: 5 or 9 digit formats
- NPI numbers: 10-digit format

### Cross-Reference Validations

- Patient name consistency across documents
- Date logic (treatment dates after diagnosis dates)
- Age-appropriate medications and diagnoses

---

## Output Requirements

### Filled PA Form

- All mapped fields populated
- Checkboxes marked appropriately
- Required signatures noted (but not filled)

### Missing Fields Report

```markdown
# Missing Fields Report

## High Priority Missing Fields

- [Field name]: [Reason not found]

## Medium Priority Missing Fields

- [Field name]: [Reason not found]

## Fields with Low Confidence

- [Field name]: [Confidence score] - [Source information]

## Source Page References

- [Field name]: Found on page [X] of [document type]
```

---

## Medication-Specific Notes

### Rituximab (Biologics)

- Requires extensive prior authorization
- Multiple biosimilar options with preference hierarchies
- Complex clinical criteria based on specific conditions
- Infusion center administration requirements

### Skyrizi (IBD medications)

- Crohn's disease specific requirements
- Step therapy with multiple alternatives
- Tuberculosis screening mandatory
- Weight-based dosing considerations

### Vyepti (Antimigraine)

- Migraine prevention vs acute treatment distinction
- CGRP inhibitor class considerations
- Frequency and severity documentation requirements
- Previous preventive medication trial requirements

---

## Processing Workflow

### 1. Document Classification

- Identify PA form type and version
- Identify referral packet components
- Validate document completeness

### 2. Data Extraction

- Extract structured data from forms
- Parse clinical notes for relevant information
- Identify and resolve conflicts between sources

### 3. Field Mapping

- Map extracted data to PA form fields
- Apply validation rules
- Calculate confidence scores

### 4. Quality Assurance

- Flag missing required fields
- Identify low-confidence mappings
- Generate missing fields report

### 5. Output Generation

- Create filled PA form
- Generate missing fields report
- Provide source page references

---

## Test Document Analysis (CRITICAL REFERENCE)

### Actual Test PDFs Analysis Results

Based on comprehensive analysis of the 6 test documents in `backend/tests/test_data/`:

#### PA Forms (Native Text PDFs - ✅ Direct Extraction)

| Document | Pages | Annotations | Text Quality | Extraction Method |
|----------|-------|-------------|--------------|-------------------|
| `test_1_PA.pdf` | 5 | 354 | ✅ Native text (17,894 chars) | pdfplumber + form fields |
| `test_2_PA.pdf` | 2 | 131 | ✅ Native text (4,799 chars) | pdfplumber + form fields |
| `test_3_PA.pdf` | 3 | 0 | ✅ Native text (5,333 chars) | pdfplumber + regex patterns |

**PA Form Extraction Strategy:**
```python
# High confidence extraction for PA forms
confidence_level = 0.90+
extraction_methods = ["pdfplumber_text", "form_annotations", "regex_patterns"]
fallback_required = False  # Native text works well
```

#### Referral Packages (Scanned Image PDFs - ⚠️ Requires OCR/AI)

| Document | Pages | Text Extracted | Extraction Challenge |
|----------|-------|----------------|---------------------|
| `test_1_referral_package.pdf` | 15 | 0 characters | ⚠️ Fully scanned - needs OCR + AI vision |
| `test_2_referral_package.pdf` | 10 | 0 characters | ⚠️ Fully scanned - needs OCR + AI vision |
| `test_3_referral_package.pdf` | 9 | 0 characters | ⚠️ Fully scanned - needs OCR + AI vision |

**Referral Extraction Strategy:**
```python
# Multi-method extraction pipeline for scanned referrals
extraction_pipeline = [
    "gemini_vision",      # Primary: AI vision model (confidence: 0.75-0.90)
    "ocr_tesseract",      # Secondary: OCR (confidence: 0.50-0.75)
    "manual_review"       # Fallback: Human review (confidence: 1.0)
]
minimum_confidence = 0.60
requires_manual_review = True  # For critical fields
```

### Critical Code Requirements

**Any extraction code MUST handle:**

1. **Mixed Document Types**
   - Native text PDFs (PA forms) - high confidence extraction
   - Scanned image PDFs (referrals) - multi-method extraction with fallbacks

2. **Page Range Handling** 
   - Minimum: 2 pages (test_2_PA.pdf)
   - Maximum: 15 pages (test_1_referral_package.pdf)
   - Average: 6.3 pages per document

3. **Form Field Detection**
   - Variable annotation counts: 0-354 per document
   - Multiple field types: text, checkboxes, dates, signatures
   - Coordinate-based positioning required

4. **Confidence Scoring Framework**
   ```python
   # Confidence levels based on extraction method
   CONFIDENCE_LEVELS = {
       "native_pdf_text": 0.95,
       "form_annotations": 0.90,
       "gemini_vision": 0.80,
       "ocr_high_quality": 0.70,
       "ocr_low_quality": 0.50,
       "manual_review": 1.0
   }
   ```

5. **Missing Field Management**
   - Expected for scanned documents
   - Priority-based flagging (high/medium/low)
   - Source page tracking for manual review

### Validation Against Test Documents

**Before any extraction code is considered complete:**

- [ ] Successfully extracts patient name from all 3 PA forms
- [ ] Handles 15-page referral document without timeout
- [ ] Produces confidence scores for each extracted field
- [ ] Identifies missing fields with appropriate priority levels
- [ ] Tracks source page numbers for all extractions
- [ ] Handles both native text and scanned image scenarios
- [ ] Generates proper error messages for extraction failures

### Performance Benchmarks

Based on test document characteristics:

- **Processing Time**: <30 seconds per document (including OCR)
- **Memory Usage**: <500MB for largest document (15 pages)
- **Confidence Thresholds**:
  - High confidence: ≥0.90 (PA forms with native text)
  - Medium confidence: 0.70-0.89 (AI vision extraction)
  - Low confidence: 0.50-0.69 (OCR extraction)
  - Manual review required: <0.50

### Error Scenarios to Handle

1. **OCR Failures**: Pages 8-12 in scanned documents commonly fail
2. **Partial Extraction**: Some pages readable, others corrupted
3. **Mixed Quality**: High-quality scan pages mixed with poor quality
4. **Missing Required Fields**: Provider NPI, insurance group numbers
5. **Format Variations**: Date formats, name formats, phone numbers

This analysis ensures all extraction code works reliably with the actual test documents and scales to similar document types in production.

---

This guide should be updated as new PA forms and referral patterns are encountered to maintain accuracy and completeness.
