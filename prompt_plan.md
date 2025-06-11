# Prior Authorization Automation - Prompt Plan

## Day 1-2: Backend Core Setup

### Project Structure Prompt: 
"Create the complete project folder structure for a FastAPI backend and Next.js frontend application. Include backend/app/, frontend/, docs/, and test directories."

### .gitignore Prompt:
"Create a comprehensive .gitignore file for a Python FastAPI backend and Next.js TypeScript frontend project. Include env files, cache, build outputs, and IDE files."

### requirements.txt Prompt:
"Create a requirements.txt file for a FastAPI project that processes PDFs using Gemini API, OpenAI, Mistral OCR, with Redis caching and async support."

### .env.example Prompt:
"Create a .env.example file showing all required environment variables for API keys (Gemini, OpenAI, Mistral), Redis connection, and file upload settings."

### main.py Prompt:
"Build the main FastAPI application file with CORS middleware, router includes for upload/process/download/health endpoints, exception handlers, and proper app configuration."

### config.py Prompt:
"Create a Pydantic Settings configuration file that loads environment variables for API keys, Redis URL, upload directory, max file size, and other app settings."

### deps.py Prompt:
"Create a FastAPI dependencies file with common dependencies like get_settings, get_db, and authentication dependencies for the API endpoints."

### schemas.py Prompt:
"Build Pydantic models for UploadResponse, ProcessingStatus (with enum), ExtractedData with confidence scores, PAFormField, MissingField, and ProcessingResult schemas."

### upload.py Route Prompt:
"Create a FastAPI upload endpoint that accepts two PDF files (referral and PA form), validates them, generates a session ID, saves files, and returns upload confirmation."

### storage.py Prompt:
"Build a FileStorage service class with methods to save_file, get_file, delete_file, create session directories, and handle temporary file management."

### pdf_extractor.py Prompt:
"Create a PDFExtractor base class with methods to extract text from PDFs, get page count, chunk large PDFs (>20MB), and extract text with coordinates using pdfplumber."

### gemini_service.py Prompt:
"Build a GeminiService class that integrates with Google's Gemini API for PDF text extraction, includes retry logic, confidence scoring, and handles chunking for large files."

### health.py Route Prompt:
"Create a health check endpoint that returns API status, version info, and checks connectivity to external services (Redis, AI APIs)."

### file_handler.py Prompt:
"Build utility functions for PDF validation, file size checking, session ID generation using UUID, and other file handling helpers."

## Day 3-4: AI Integration & Processing

### mistral_service.py Prompt:
"Create a MistralService class for OCR fallback when Gemini confidence is low. Include OCR extraction methods and confidence threshold checking."

### openai_service.py Prompt:
"Build an OpenAIService class using GPT-4 for intelligent field mapping between referral data and PA form fields. Include prompt engineering for medical data extraction."

### field_mapper.py Prompt:
"Create a FieldMapper class with methods to normalize field names, match patient names with fuzzy logic, normalize date formats, extract insurance IDs, and calculate confidence scores."

### widget_detector.py Prompt:
"Build a WidgetDetector class using pdfforms to detect form fields in PA PDFs, extract field properties (name, type, coordinates, required status), and create field templates."

### concurrent_processor.py Prompt:
"Create an AsyncProcessor class for concurrent page processing with worker pools, progress tracking, and error aggregation for parallel PDF processing."

### processing_pipeline.py Prompt:
"Build the main ProcessingPipeline class that orchestrates document extraction, field detection, data mapping, confidence thresholds, and generates final results."

### process.py Route Prompt:
"Create process endpoints: POST to start processing with background tasks, GET for status updates with progress tracking and error handling."

### Updated schemas.py Prompt:
"Add FieldMapping, ConfidenceScore, and ExtractionResult models to support the processing pipeline data structures."

## Day 5: Backend Finalization

### form_filler.py Prompt:
"Create a FormFiller class using pdfforms to fill widget-based PDF forms, validate field values, and save the filled PDF output."

### report_generator.py Prompt:
"Build a ReportGenerator class that creates markdown reports listing missing fields, low confidence extractions, and includes source page references."

### cache.py Prompt:
"Create a Redis cache service with methods to connect, cache results with 24-hour TTL, retrieve cached results, and handle cache invalidation."

### download.py Route Prompt:
"Build download endpoints for filled PDFs and reports with file streaming, proper headers, and error handling for missing files."

### middleware.py Prompt:
"Create FastAPI middleware for global exception handling, request validation, and request ID tracking for debugging."

### logging.py Prompt:
"Set up structured logging configuration with formatters, log rotation, and request ID tracking for the FastAPI application."

### Test Files Prompts:
- **conftest.py**: "Create pytest fixtures for FastAPI testing including test client, mock services, and sample PDF files."
- **test_pdf_extraction.py**: "Write unit tests for PDF extraction functions including text extraction, chunking, and error cases."
- **test_field_mapping.py**: "Create tests for field mapping logic including confidence scoring and data normalization."
- **test_api_endpoints.py**: "Build integration tests for all API endpoints including upload, process, and download flows."

## Day 6-7: Frontend Development

### Next.js Setup Prompt:
"Initialize a Next.js 14 project with TypeScript, Tailwind CSS, and App Router. Configure for a document processing application."

### package.json Dependencies Prompt:
"Update package.json with dependencies for react-dropzone, react-pdf, axios, and Radix UI components for a PDF processing frontend."

### .env.local Prompt:
"Create environment variables file for Next.js with API URL configuration for local and production environments."

### types/index.ts Prompt:
"Define TypeScript interfaces for UploadResponse, ProcessingStatus, ProcessingStage enum, FileUpload, and APIError types."

### api-client.ts Prompt:
"Build an Axios-based API client with methods for file upload, status checking, and file downloads. Include error handling and auth headers."

### utils.ts Prompt:
"Create utility functions including className merger (cn), file size formatter, and file type icon selector for the UI."

### FileUpload.tsx Prompt:
"Build a drag-and-drop file upload component using react-dropzone that accepts exactly 2 PDFs (referral and PA form) with validation and preview."

### ProcessingStatus.tsx Prompt:
"Create a processing status component with progress stepper, stage indicators, loading animations, and error state handling."

### PDFPreview.tsx Prompt:
"Build a PDF preview component using react-pdf with page navigation, zoom controls, and loading states for document viewing."

### ResultsDisplay.tsx Prompt:
"Create a results display component showing processing summary, confidence metrics, missing fields list, and download buttons."

### UI Component Prompts:
- **button.tsx**: "Create a reusable Button component using Radix UI and Tailwind with variants for primary, secondary, and destructive styles."
- **card.tsx**: "Build a Card component with header and content sections using Tailwind for the document display UI."
- **progress.tsx**: "Create a Progress bar component using Radix UI Progress with custom styling for processing status."
- **alert.tsx**: "Build an Alert component for error and success messages with icon support and dismissible functionality."

### Custom Hook Prompts:
- **useFileUpload.ts**: "Create a custom hook for file upload state management, validation, and upload progress tracking."
- **useProcessingStatus.ts**: "Build a hook that polls processing status, handles updates, and manages completion states."

### page.tsx (Main) Prompt:
"Create the main landing page with file upload interface, project description, and navigation to processing page after upload."

### process/[id]/page.tsx Prompt:
"Build the processing page that shows real-time status updates, handles completion, and displays results for a given session ID."

### layout.tsx Prompt:
"Create the root layout with metadata, font configuration, and provider setup for the Next.js application."

### globals.css Prompt:
"Set up global styles with Tailwind directives, custom animations for loading states, and component-specific styles."

## Day 8-9: Integration & Testing

### Integration Testing Prompt:
"Create a comprehensive integration test plan covering file upload, processing, and download flows with multiple test scenarios."

### Performance Testing Prompt:
"Design performance tests for large PDF processing, concurrent requests, and memory usage monitoring."

### Golden Output Creation Prompt:
"Process the test PDFs and create expected output files with filled forms and missing field reports for regression testing."

### Bug Fix Documentation Prompt:
"Create a bug tracking document template for issues found during testing with severity, steps to reproduce, and fix status."

### Frontend Polish Prompt:
"Add loading skeletons, improve responsive design, add helpful tooltips, and enhance error messaging throughout the UI."

## Day 10: Deployment & Documentation

### render.yaml Prompt:
"Create a Render deployment configuration for the FastAPI backend with Python environment, build commands, and start script."

### Deployment Scripts Prompt:
"Write deployment scripts for backend (Render) and frontend (Vercel) with environment variable setup and health checks."

### README.md Update Prompt:
"Write a comprehensive README with project overview, architecture diagram, installation steps, API documentation, and usage examples with screenshots."

### API.md Prompt:
"Create detailed API documentation with all endpoints, request/response examples, error codes, and authentication details."

### DEPLOYMENT.md Prompt:
"Write a deployment guide covering local setup, Render backend deployment, Vercel frontend deployment, and environment configuration."

### TESTING.md Prompt:
"Create testing documentation explaining how to run tests, create golden outputs, and validate processing results."

### Output Examples Prompt:
"Generate sample output files showing filled PA forms and missing field reports for documentation purposes."

### Code Cleanup Prompt:
"Review all code to remove console.logs, commented code, hardcoded values, and ensure consistent formatting with proper linting."

### Final Submission Checklist Prompt:
"Create a pre-submission checklist ensuring all files are present, deployments work, documentation is complete, and branch naming is correct."