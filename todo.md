# Prior Authorization Automation - TODO Checklist

## Day 1-2: Backend Core Setup

### Project Structure Setup
- [ ] Create project structure:
  ```
  MediLink-AI/
  ├── backend/
  │   ├── app/
  │   ├── tests/
  │   │   └── test_data/     # Move testcases here
  │   └── requirements.txt
  ├── frontend/
  ├── docs/
  │   ├── spec.md           # Move from root
  │   ├── todo.md           # Move from root
  │   ├── progress_notes.md # Move from root
  │   └── prompt_plan.md    # Move from root
  ├── testcases/            # Already exists
  ├── CLAUDE.md             # Already exists
  └── README.md             # Keep in root
  ```
- [ ] Move `testcases/` folder to `backend/tests/test_data/`
- [ ] Move `spec.md` to `docs/spec.md`
- [ ] Move `todo.md` to `docs/todo.md`
- [ ] Move `progress_notes.md` to `docs/progress_notes.md`
- [ ] Move `prompt_plan.md` to `docs/prompt_plan.md`
- [ ] Create `.gitignore` file in root with Python, Node, .env exclusions

### Backend Initial Setup
- [ ] Create `backend/requirements.txt`:
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
  python-jose[cryptography]
  passlib[bcrypt]
  pytest==7.4.3
  httpx==0.25.2
  ```
- [ ] Create virtual environment in backend: `cd backend && python -m venv venv`
- [ ] Activate venv and install dependencies: `pip install -r requirements.txt`
- [ ] Create `backend/.env` file for API keys
- [ ] Create `backend/.env.example`:
  ```
  GEMINI_API_KEY=your_key_here
  OPENAI_API_KEY=your_key_here
  MISTRAL_API_KEY=your_key_here
  REDIS_URL=redis://localhost:6379
  UPLOAD_DIR=./uploads
  MAX_FILE_SIZE=52428800
  ```

### FastAPI Base Files
- [ ] Create `backend/app/__init__.py` (empty file)
- [ ] Create `backend/app/main.py`:
  - FastAPI app initialization
  - CORS middleware setup
  - Include routers
  - Exception handlers
- [ ] Create `backend/app/api/__init__.py`
- [ ] Create `backend/app/api/routes/__init__.py`

### Core Configuration Files
- [ ] Create `backend/app/core/__init__.py`
- [ ] Create `backend/app/core/config.py`:
  - Settings class with all env variables
  - File upload settings
  - API configuration
- [ ] Create `backend/app/core/deps.py`:
  - Common dependencies
  - Settings dependency

### Model/Schema Files
- [ ] Create `backend/app/models/__init__.py`
- [ ] Create `backend/app/models/schemas.py`:
  - `UploadResponse` model
  - `ProcessingStatus` enum and model
  - `ExtractedData` model with confidence
  - `PAFormField` model
  - `MissingField` model
  - `ProcessingResult` model

### Service Layer Structure
- [ ] Create `backend/app/services/__init__.py`
- [ ] Create `backend/app/utils/__init__.py`

### Upload Route Implementation
- [ ] Create `backend/app/api/routes/upload.py`:
  - POST `/api/upload` endpoint
  - File validation logic
  - Session ID generation
  - File storage handling

### Storage Service Implementation
- [ ] Create `backend/app/services/storage.py`:
  - `FileStorage` class
  - `save_file()` method
  - `get_file()` method
  - `delete_file()` method
  - `create_session_directory()` method

### Basic PDF Extractor
- [ ] Create `backend/app/services/pdf_extractor.py`:
  - `PDFExtractor` base class
  - `extract_text_from_pdf()` method
  - `get_page_count()` method
  - `chunk_pdf()` method for large files

### Gemini Service Setup
- [ ] Create `backend/app/services/gemini_service.py`:
  - `GeminiService` class
  - `initialize_client()` method
  - `extract_from_pdf()` method
  - Basic confidence scoring
  - Retry logic implementation

### Health Check Route
- [ ] Create `backend/app/api/routes/health.py`:
  - GET `/api/health` endpoint
  - Service connectivity checks

### File Handler Utilities
- [ ] Create `backend/app/utils/file_handler.py`:
  - `validate_pdf()` function
  - `get_file_size()` function
  - `generate_session_id()` function

## Day 3-4: AI Integration & Processing

### Mistral OCR Service
- [ ] Create `backend/app/services/mistral_service.py`:
  - `MistralService` class
  - `extract_with_ocr()` method
  - Confidence threshold logic
  - Integration with extraction pipeline

### OpenAI Service
- [ ] Create `backend/app/services/openai_service.py`:
  - `OpenAIService` class
  - `create_field_mapping_prompt()` method
  - `extract_and_map_fields()` method
  - Response parsing logic

### Field Mapping Engine
- [ ] Create `backend/app/services/field_mapper.py`:
  - `FieldMapper` class
  - `normalize_field_name()` method
  - `match_patient_name()` method
  - `normalize_date_format()` method
  - `extract_insurance_id()` method
  - `calculate_confidence_score()` method

### Widget Detection Service
- [ ] Create `backend/app/services/widget_detector.py`:
  - `WidgetDetector` class
  - `detect_form_fields()` method using pdfforms
  - `extract_field_properties()` method
  - `create_field_template()` method

### Concurrent Processing
- [ ] Create `backend/app/services/concurrent_processor.py`:
  - `AsyncProcessor` class
  - `process_pages_concurrently()` method
  - `create_worker_pool()` method
  - Progress tracking implementation

### Processing Pipeline
- [ ] Create `backend/app/services/processing_pipeline.py`:
  - `ProcessingPipeline` class
  - `process_documents()` main method
  - `extract_referral_data()` method
  - `detect_pa_fields()` method
  - `map_data_to_fields()` method
  - `apply_confidence_thresholds()` method

### Process Route
- [ ] Create `backend/app/api/routes/process.py`:
  - POST `/api/process/{session_id}` endpoint
  - GET `/api/process/{session_id}/status` endpoint
  - Background task integration

### Data Extraction Models
- [ ] Update `backend/app/models/schemas.py`:
  - Add `FieldMapping` model
  - Add `ConfidenceScore` model
  - Add `ExtractionResult` model

## Day 5: Backend Finalization

### Form Filling Service
- [ ] Create `backend/app/services/form_filler.py`:
  - `FormFiller` class
  - `fill_widget_form()` method
  - `validate_field_value()` method
  - `save_filled_pdf()` method

### Report Generator
- [ ] Create `backend/app/services/report_generator.py`:
  - `ReportGenerator` class
  - `generate_missing_fields_report()` method
  - `format_as_markdown()` method
  - `include_confidence_details()` method

### Redis Cache Service
- [ ] Create `backend/app/services/cache.py`:
  - `CacheService` class
  - `connect_redis()` method
  - `cache_result()` method
  - `get_cached_result()` method
  - `invalidate_cache()` method

### Download Routes
- [ ] Create `backend/app/api/routes/download.py`:
  - GET `/api/download/{session_id}/filled` endpoint
  - GET `/api/download/{session_id}/report` endpoint
  - File streaming implementation

### Middleware Setup
- [ ] Create `backend/app/core/middleware.py`:
  - Global exception handler
  - Request validation middleware
  - Logging middleware

### Logging Configuration
- [ ] Create `backend/app/core/logging.py`:
  - Configure structured logging
  - Set up log formatters
  - Request ID tracking

### Error Handlers
- [ ] Update `backend/app/main.py`:
  - Add custom exception handlers
  - Configure error responses

### Backend Tests
- [ ] Create `backend/tests/__init__.py`
- [ ] Create `backend/tests/conftest.py` with fixtures
- [ ] Create `backend/tests/test_pdf_extraction.py`
- [ ] Create `backend/tests/test_field_mapping.py`
- [ ] Create `backend/tests/test_api_endpoints.py`

## Day 6-7: Frontend Development

### Next.js Project Setup
- [ ] Navigate to frontend folder: `cd frontend`
- [ ] Run: `npx create-next-app@latest . --typescript --tailwind --app`
- [ ] Update `frontend/package.json` dependencies:
  ```json
  {
    "dependencies": {
      "react-dropzone": "^14.2.3",
      "react-pdf": "^7.5.0",
      "axios": "^1.6.0",
      "@radix-ui/react-alert": "latest",
      "@radix-ui/react-button": "latest",
      "@radix-ui/react-card": "latest",
      "@radix-ui/react-progress": "latest",
      "@radix-ui/react-tabs": "latest"
    }
  }
  ```
- [ ] Run: `npm install`
- [ ] Create `frontend/.env.local`:
  ```
  NEXT_PUBLIC_API_URL=http://localhost:8000
  ```

### Frontend Structure Setup
- [ ] Create `frontend/src/lib/` directory
- [ ] Create `frontend/src/components/` directory
- [ ] Create `frontend/src/hooks/` directory
- [ ] Create `frontend/src/types/` directory

### Type Definitions
- [ ] Create `frontend/src/types/index.ts`:
  - `UploadResponse` interface
  - `ProcessingStatus` interface
  - `ProcessingStage` enum
  - `FileUpload` interface
  - `APIError` interface

### API Client
- [ ] Create `frontend/src/lib/api-client.ts`:
  - Axios instance configuration
  - `uploadFiles()` function
  - `getProcessingStatus()` function
  - `downloadFilledForm()` function
  - `downloadReport()` function
  - Error handling wrapper

### Utility Functions
- [ ] Create `frontend/src/lib/utils.ts`:
  - `cn()` function for className merging
  - `formatFileSize()` function
  - `getFileIcon()` function

### File Upload Component
- [ ] Create `frontend/src/components/FileUpload.tsx`:
  - Dropzone implementation
  - File validation
  - Two-file requirement logic
  - Upload progress UI
  - File preview cards

### Processing Status Component
- [ ] Create `frontend/src/components/ProcessingStatus.tsx`:
  - Progress stepper UI
  - Stage indicators
  - Loading animations
  - Error state handling

### PDF Preview Component
- [ ] Create `frontend/src/components/PDFPreview.tsx`:
  - PDF.js integration
  - Page navigation
  - Zoom controls
  - Loading states

### Results Display Component
- [ ] Create `frontend/src/components/ResultsDisplay.tsx`:
  - Summary statistics
  - Confidence metrics display
  - Missing fields list
  - Download buttons

### UI Components
- [ ] Create `frontend/src/components/ui/button.tsx`
- [ ] Create `frontend/src/components/ui/card.tsx`
- [ ] Create `frontend/src/components/ui/progress.tsx`
- [ ] Create `frontend/src/components/ui/alert.tsx`

### Custom Hooks
- [ ] Create `frontend/src/hooks/useFileUpload.ts`:
  - File state management
  - Upload logic
  - Progress tracking
- [ ] Create `frontend/src/hooks/useProcessingStatus.ts`:
  - Polling implementation
  - Status updates

### Main Upload Page
- [ ] Update `frontend/src/app/page.tsx`:
  - Landing page layout
  - FileUpload integration
  - Navigation to processing

### Processing Page
- [ ] Create `frontend/src/app/process/[id]/page.tsx`:
  - Session ID handling
  - Status polling
  - Results display

### Layout Updates
- [ ] Update `frontend/src/app/layout.tsx`:
  - Add metadata
  - Configure fonts
  - Set up providers

### Global Styles
- [ ] Update `frontend/src/app/globals.css`:
  - Tailwind directives
  - Custom animations
  - Component styles

## Day 8-9: Integration & Testing

### Integration Testing
- [ ] Test complete upload flow with test PDFs
- [ ] Verify processing status updates work
- [ ] Test download functionality
- [ ] Check error handling scenarios
- [ ] Validate CORS configuration

### Performance Testing
- [ ] Test with large PDF files (30+ pages)
- [ ] Measure processing times
- [ ] Check memory usage
- [ ] Test concurrent requests

### Create Golden Outputs
- [ ] Process `test_1_referral_package.pdf` + `test_1_PA.pdf`
- [ ] Save output as `test_1_filled.pdf` and `test_1_report.md`
- [ ] Process test case 2
- [ ] Process test case 3
- [ ] Document expected vs actual results

### Bug Fixes
- [ ] Fix identified issues from testing
- [ ] Improve error messages
- [ ] Optimize slow operations
- [ ] Add missing validations

### Frontend Polish
- [ ] Add loading skeletons
- [ ] Improve responsive design
- [ ] Add tooltips for complex fields
- [ ] Enhance error displays

## Day 10: Deployment & Documentation

### Backend Deployment Prep
- [ ] Create `backend/render.yaml`:
  ```yaml
  services:
    - type: web
      name: pa-automation-backend
      env: python
      buildCommand: "pip install -r requirements.txt"
      startCommand: "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
  ```
- [ ] Update `backend/requirements.txt` with production deps
- [ ] Add `backend/Procfile` if needed

### Deploy Backend to Render
- [ ] Create Render account and project
- [ ] Connect GitHub repository
- [ ] Configure environment variables
- [ ] Deploy and test endpoints
- [ ] Note production URL

### Frontend Deployment Prep
- [ ] Update `frontend/.env.local` with production API URL
- [ ] Build and test locally: `npm run build`
- [ ] Fix any build errors

### Deploy Frontend to Vercel
- [ ] Connect GitHub repository to Vercel
- [ ] Configure build settings
- [ ] Set environment variables
- [ ] Deploy and test
- [ ] Configure domain if needed

### Update Main README.md
- [ ] Add project description
- [ ] Include architecture overview
- [ ] Add installation instructions:
  - Backend setup steps
  - Frontend setup steps
  - Environment configuration
- [ ] Document API endpoints
- [ ] Add usage examples
- [ ] Include screenshots

### Create Additional Documentation
- [ ] Create `docs/API.md` with endpoint details
- [ ] Create `docs/DEPLOYMENT.md` with deployment guide
- [ ] Create `docs/TESTING.md` with test instructions
- [ ] Create `output_examples/` directory
- [ ] Add sample outputs to `output_examples/`

### Final Production Testing
- [ ] Test complete flow on production
- [ ] Verify all endpoints work
- [ ] Check performance
- [ ] Test error scenarios
- [ ] Validate output quality

### Code Cleanup
- [ ] Remove all console.log statements
- [ ] Remove commented code
- [ ] Ensure no hardcoded values
- [ ] Format all files consistently
- [ ] Run linters

### GitHub Submission
- [ ] Ensure all files are committed
- [ ] Verify `.gitignore` excludes sensitive files
- [ ] Check branch name: `automation-pa-filling-[your-name]`
- [ ] Push all changes
- [ ] Verify deployment links work in README