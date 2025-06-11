# Prior Authorization Automation - Project Specification

## Project Overview

An AI-powered system that automates the extraction of information from medical referral packets and populates insurance-specific Prior Authorization (PA) forms, eliminating manual data entry and reducing processing time from weeks to minutes.

## Technical Stack

### Frontend
- **Framework**: Next.js 14+ (App Router)
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui
- **File Upload**: react-dropzone
- **PDF Preview**: react-pdf or pdf.js
- **Deployment**: Vercel

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **PDF Processing**: 
  - pdfforms (widget-based forms)
  - pdfplumber (text extraction with coordinates)
- **AI/OCR Services**:
  - Google Gemini API (primary)
  - Mistral OCR (fallback)
  - OpenAI GPT-4 (field mapping)
- **Deployment**: Render
- **Storage**: Temporary file system + Redis for caching

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js UI    │────▶│  FastAPI Backend │────▶│  AI Services    │
│    (Vercel)     │     │    (Render)      │     │ (Gemini/OpenAI) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  File Storage   │
                        │  (Temp + Cache) │
                        └─────────────────┘
```

## Core Features

### 1. Document Processing Pipeline

#### Phase 1: Extraction
```python
# Pseudo-flow
1. Upload referral PDF + PA form PDF
2. For referral packet:
   - Check file size (chunk if > 20MB)
   - Process pages concurrently with Gemini API
   - If confidence < threshold, use Mistral OCR
   - Extract all text with coordinates
3. For PA form:
   - Use pdfforms to detect widget fields
   - Extract field names, types, and positions
   - Create field mapping template
```

#### Phase 2: Data Mapping
```python
# AI-powered field matching
1. Analyze PA form fields
2. Use GPT-4/Gemini to match:
   - Patient demographics
   - Clinical information
   - Treatment details
   - Insurance information
3. Apply confidence scoring:
   - High (>90%): Auto-fill
   - Medium (70-90%): Fill with flag
   - Low (<70%): Mark as missing
```

#### Phase 3: Form Population
```python
# Fill PA form
1. Use pdfforms to populate widget fields
2. Generate filled PDF
3. Create missing fields report (markdown)
4. Return both files to user
```

### 2. Data Structure

```json
{
  "session_id": "uuid",
  "timestamp": "2024-01-01T00:00:00Z",
  "referral_data": {
    "patient_info": {
      "name": {"value": "John Doe", "confidence": 0.95, "source_page": 1},
      "dob": {"value": "01/01/1980", "confidence": 0.92, "source_page": 1},
      "insurance_id": {"value": "123456789", "confidence": 0.88, "source_page": 3}
    },
    "clinical_data": {
      "diagnosis": {"value": "...", "confidence": 0.91, "source_page": 5},
      "treatment_plan": {"value": "...", "confidence": 0.85, "source_page": 7}
    },
    "raw_extracted": {
      "page_1": {"text": "...", "bbox": [...]}
    }
  },
  "pa_form_fields": {
    "patient_name": {
      "type": "text",
      "required": true,
      "coordinates": {"x": 100, "y": 200},
      "mapped_value": "John Doe",
      "confidence": 0.95
    }
  },
  "missing_fields": [
    {"field": "provider_npi", "reason": "Not found in referral"},
    {"field": "prior_auth_number", "reason": "Low confidence (0.45)"}
  ]
}
```

### 3. API Endpoints

```python
# FastAPI endpoints
POST   /api/upload          # Upload referral + PA form
GET    /api/process/{id}    # Get processing status
GET    /api/download/{id}   # Download filled form + report
POST   /api/extract         # Extract data from single document
POST   /api/map-fields      # Map extracted data to form fields
GET    /api/health          # Health check
```

### 4. Frontend Components

```
src/
├── app/
│   ├── page.tsx           # Main upload interface
│   ├── process/[id]/      # Processing status page
│   └── api/               # API route handlers
├── components/
│   ├── FileUpload.tsx     # Drag-drop upload
│   ├── ProcessingStatus.tsx
│   ├── PDFPreview.tsx
│   └── ResultsDisplay.tsx
└── lib/
    ├── api-client.ts
    └── utils.ts
```

## Implementation Plan (10-Day Sprint)

### Days 1-2: Backend Core Setup
1. Set up FastAPI project structure
2. Implement PDF extraction with Gemini API
3. Create pdfforms integration for widget detection
4. Build basic file upload/storage system
5. Implement chunking for large files

### Days 3-4: AI Integration & Processing
1. Add Mistral OCR fallback mechanism
2. Implement GPT-4/Gemini field mapping
3. Build confidence scoring system
4. Add concurrent processing for pages
5. Create mapping rules engine

### Day 5: Backend Finalization
1. Build missing fields detection & reporting
2. Implement caching with Redis
3. Complete API endpoints
4. Backend testing and debugging
5. Prepare backend for deployment

### Days 6-7: Frontend Development
1. Create Next.js app with upload interface
2. Build drag-and-drop file upload component
3. Implement processing status page
4. Add PDF preview functionality
5. Create results download interface

### Days 8-9: Integration & Testing
1. Connect frontend to backend APIs
2. End-to-end testing with test cases
3. Create golden outputs for validation
4. Fix integration issues
5. Performance optimization

### Day 10: Deployment & Documentation
1. Deploy backend to Render
2. Deploy frontend to Vercel
3. Configure all environment variables
4. Final testing in production
5. Complete documentation and submit

## Testing Strategy

### Test Cases
1. **Simple PA Form**: Single page, clear text
2. **Complex Referral**: 30+ pages, mixed quality
3. **Handwritten Notes**: Poor OCR quality test

### Golden Outputs
- Create manual reference outputs for each test case
- Use for regression testing
- Validate with PDF diff tools

### Testing Approach
```python
# Unit tests
- PDF extraction functions
- Field mapping logic
- Confidence scoring

# Integration tests
- Full pipeline processing
- API endpoint testing
- Error handling

# Manual validation
- Visual comparison of outputs
- Field accuracy checks
- Missing data reports
```

## Error Handling

### OCR Failures
1. Primary: Gemini API extraction
2. Fallback: Mistral OCR
3. Final: Mark for manual review

### Processing Errors
- Graceful degradation
- Detailed error logging
- User-friendly error messages

### Missing Data
- Comprehensive reporting
- Confidence thresholds
- Clear documentation of gaps

## Performance Optimizations

### Concurrent Processing
- Async page processing
- Parallel OCR requests
- Batch field mapping

### Caching Strategy
- Redis for processed documents
- 24-hour cache TTL
- Cache invalidation on errors

### File Size Management
- Automatic chunking for files > 20MB
- Stream processing for large documents
- Memory-efficient processing

## Security Considerations

- HIPAA compliance considerations
- Secure file handling
- API key management (.env)
- No permanent storage of PHI
- SSL/TLS for all communications

## Future Enhancements

### Phase 2 Features
1. **Non-widget PDF Support**
   - Use pdfplumber for coordinate-based filling
   - Visual field detection with AI
   - Template matching for common forms

2. **Form Versioning System**
   - Template library for different insurers
   - Version detection algorithms
   - Automatic template updates

3. **Performance Improvements**
   - Minimize processing time
   - Batch processing for multiple patients
   - GPU acceleration for OCR

4. **Advanced Features**
   - Multi-language support
   - Handwriting recognition improvement
   - Conditional logic handling
   - Real-time collaboration

## Configuration

### Environment Variables
```env
# .env.local (Frontend)
NEXT_PUBLIC_API_URL=https://your-api.onrender.com

# .env (Backend)
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
REDIS_URL=your_redis_url
```

### Dependencies

#### Backend (requirements.txt)
```
fastapi==0.104.1
uvicorn==0.24.0
pdfforms==1.2.0
pdfplumber==0.10.3
google-generativeai==0.3.0
openai==1.3.0
mistralai==0.0.1
redis==5.0.1
python-multipart==0.0.6
aiofiles==23.2.1
```

#### Frontend (package.json)
```json
{
  "dependencies": {
    "next": "14.0.0",
    "react": "18.2.0",
    "react-dropzone": "14.2.3",
    "react-pdf": "7.5.0",
    "axios": "1.6.0",
    "@radix-ui/react-*": "latest",
    "tailwindcss": "3.3.0"
  }
}
```

## Deliverables

1. **Source Code**
   - Branch: `automation-pa-filling-[your-name]`
   - Modular, well-commented code
   - Clear separation of concerns

2. **Documentation**
   - Updated README.md with installation instructions
   - API documentation
   - Architecture diagrams

3. **Output Examples**
   - Sample filled PA forms
   - Missing fields reports
   - Test case results

## Success Metrics

- Extraction accuracy > 95% for typed text
- Processing time < 2 minutes per document
- Support for 3+ different PA form templates
- Zero data loss for critical fields
- Clear identification of all missing information