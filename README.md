# Medical Form Processing Pipeline

An automated pipeline that helps process prior authorization (PA) forms using AI-powered OCR, data extraction, and form filling.

## 🎯 Overview

This pipeline streamlines the workflow of filling medical prior authorization forms:
1. **Parsing** - Systematically extracts fillable widgets from PDF forms
2. **OCR Processing** - Leverages Mistral AI OCR to extract structured text from medical documents 
3. **Populating** - Uses OpenAI to enrich widget metadata with contextual information from referral packages
4. **Filling** - Analyzes referral packages and intelligently fills PDF forms using AI reasoning

## 🏗️ Architecture

The pipeline uses a modular, step-based architecture that's easy to understand and maintain:

```
┌─────────────┐    ┌─────────────┐    ┌───────────────┐    ┌─────────────┐
│    Parse    │───>│     OCR     │───>│   Populate    │───>│    Fill     │
│             │    │             │    │               │    │             │
│ Extract PDF │    │ Text from   │    │ Add context   │    │ AI form     │
│ widgets     │    │ documents   │    │ to widgets    │    │ completion  │
└─────────────┘    └─────────────┘    └───────────────┘    └─────────────┘
```

### Step Details

1. **Parse Step** (`src/steps/parse.py`)
   - Extracts fillable form widgets from PDF documents
   - Identifies field types, positions, and metadata
   - Sorts widgets by reading order (top-to-bottom, left-to-right)
   - Saves widget metadata to JSON for later processing

2. **OCR Step** (`src/steps/ocr.py`)
   - Uses Mistral AI's OCR API to extract text from PDFs
   - Processes both prior authorization forms and referral packages
   - Creates clean, structured markdown that preserves the document structure
   - Handles various PDF formats and layouts

3. **Populate Step** (`src/steps/populate.py`)
   - Uses OpenAI o4-mini with structured JSON Schema responses
   - Enriches widget metadata by adding contextual information from referral packages
   - Maps relevant data from referral packages to form field contexts
   - Prepares widgets with enhanced information for the filling step

4. **Fill Step** (`src/steps/fill.py`)
   - Uses OpenAI o4-mini to analyze referral packages and determine appropriate field values
   - Employs sophisticated prompt engineering for medical reasoning
   - Handles various field types (text, checkboxes, dropdowns, dates)
   - Provides confidence scores and reasoning for each filled value
   - Creates completed PDFs with high accuracy

## 📋 Prerequisites

- Python 3.8 or higher
- PDF processing libraries (PyMuPDF, pypdf)
- API keys for OpenAI and Mistral AI
- Python virtual environment (recommended)

## 🚀 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd headstarter-mandolin-project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 5. Verify Installation
```bash
python main.py --help
```

## 🎮 Usage

### Basic Usage
Run the pipeline:
```bash
python main.py
```

### Input Structure
Put your input files in this structure:
```
input/
├── patient_name/
│   ├── prior_authorization.pdf
│   └── referral_package.pdf
```

### Output Structure
The pipeline creates these output files:
```
output/
├── patient_name/
│   ├── prior_authorization.md          # OCR text extraction
│   ├── prior_authorization_filled.pdf  # Final filled form
│   ├── referral_package.md            # OCR text extraction
│   └── widgets.json                   # Extracted form widgets
```

## 🧠 Implementation Approach

### Design Philosophy
I built this pipeline with a few key ideas in mind:

1. **Modularity**: Each step works independently so you can run them separately if needed
2. **Extensibility**: It's easy to add new steps or modify existing ones
3. **Error Handling**: Comprehensive logging and graceful error recovery
4. **AI Integration**: Uses multiple AI services to get the best results
5. **Structured Outputs**: Leverages OpenAI's JSON Schema for consistent, reliable responses

### Key Implementation Decisions

#### 1. Step-Based Architecture
- **Why**: Makes debugging easier, allows for testing individual parts, and enables parallel processing
- **Benefits**: You can restart from any step and easily modify individual components
- **Trade-offs**: A bit more complex than a single monolithic script

#### 2. Dual AI Provider Strategy
- **Mistral AI**: Great for OCR and document processing
- **OpenAI**: Better for understanding context and reasoning about medical forms
- **Benefits**: Each AI does what it's best at

#### 3. Markdown Intermediate Format
- **Why**: Easy to read and debug, preserves document structure
- **Benefits**: You can manually check OCR results and it works well with AI processing
- **Alternative**: Could use JSON, but markdown is more readable for complex forms

#### 4. Two-Stage Processing Strategy
- **Populate Stage**: Enriches widget metadata with contextual information
- **Fill Stage**: Makes final decisions on field values with confidence scoring
- **Benefits**: Better accuracy through focused AI tasks

#### 5. Structured JSON Schema Responses
- **Why**: Ensures consistent, parseable AI outputs
- **Benefits**: Eliminates parsing errors and enables reliable automation
- **Implementation**: Uses OpenAI's structured output feature for guaranteed JSON compliance

### AI Prompt Engineering

#### OCR Processing
- Uses structured prompts to get consistent markdown formatting
- Detects field labels and preserves formatting
- Recognizes checkboxes and form elements

#### Widget Population
- **Context Enrichment**: Adds relevant information from referral packages to widget metadata
- **Field Matching**: Intelligently maps referral data to appropriate form fields
- **Structured Output**: Uses JSON Schema to ensure consistent response format

#### Form Filling
- **Medical Reasoning**: Uses clinical knowledge for yes/no questions and medical decisions
- **Confidence Scoring**: Provides reliability metrics for each filled value
- **Field Type Handling**: Specialized logic for different form field types
- **Contextual Analysis**: Considers both widget context and referral package content

## 🔬 Experimentation and Development Journey

During development, I experimented with several approaches to optimize accuracy and reliability:

### Initial Approaches (What Didn't Work Well)

1. **Single-Step Processing**: 
   - **Approach**: Tried to do everything in one step - pass the whole referral package and prior authorization to AI and ask it to fill out the form
   - **Result**: AI would miss important information and make inconsistent decisions
   - **Problem**: Even larger, newer models struggled with the complexity of processing everything at once

2. **Direct JSON Processing**:
   - **Approach**: Initially tried using JSON format for all intermediate data
   - **Result**: Less reliable than markdown for complex medical documents
   - **Problem**: JSON structure didn't preserve document formatting well

3. **Static PDF Pattern Matching**:
   - **Approach**: For PDFs without fillable fields, tried having AI look for field patterns like ": - /"
   - **Result**: Unreliable and brittle for real-world forms
   - **Problem**: Too dependent on specific formatting conventions

### Successful Iterations

4. **Two-Stage Architecture**:
   - **Innovation**: Split processing into Populate (context enrichment) and Fill (decision making) stages
   - **Result**: Significantly improved accuracy and reliability
   - **Why it works**: Each AI task is focused and manageable

5. **Markdown + Structured JSON**:
   - **Innovation**: Use Mistral OCR for markdown generation, OpenAI with JSON Schema for structured decisions
   - **Result**: Best of both worlds - readable intermediate format with reliable structured outputs
   - **Why it works**: Leverages each AI's strengths

6. **Widget Sorting and Context**:
   - **Innovation**: Sort widgets by reading order and enrich with contextual information
   - **Result**: Better field matching and more intuitive form filling
   - **Why it works**: Mimics human form-filling behavior

### Current Optimizations

- **Confidence Scoring**: Each filled value includes a confidence score for quality assessment
- **Field Type Specialization**: Different handling logic for text fields, checkboxes, dates, etc.
- **Medical Domain Knowledge**: Prompts include medical reasoning for clinical decisions
- **Error Recovery**: Graceful handling of truncated responses and parsing errors

### Lessons Learned

1. **Complexity Management**: Breaking complex tasks into focused steps dramatically improves AI performance
2. **Format Selection**: Markdown is superior for document representation, JSON Schema for structured outputs
3. **Domain Expertise**: Medical form filling requires specialized prompting and domain knowledge
4. **Reliability**: Structured outputs and confidence scoring are essential for production use
5. **Widget Ordering**: Preserving spatial relationships between form fields improves accuracy

## ⚙️ Configuration

### Logging
Logging is configured in `main.py` with multiple levels:
- **INFO**: Step progress and completion
- **ERROR**: Failures and exceptions
- **DEBUG**: Detailed processing information

### AI Model Settings
- **OpenAI Model**: GPT-4o and o4 for optimal reasoning capabilities
- **Mistral Model**: Latest OCR model for document processing
- **Temperature**: Set to 0.1 for consistent, deterministic outputs

## 🔍 Assumptions and Limitations

### Current Limitations
1. **Processing Speed**: AI API calls can be slow for large documents

### Known Issues
1. **Static PDFs**: Doesn't work with PDFs that don't have fillable widgets
2. **Token Limits**: AI models might hit token limits on very complex forms in the **Fill** step

### Sample Data
The repository includes sample data for testing:
- `input/adbulla/`
- `input/akshay/`
- `input/amy/`

### Running Tests
```bash
# Run the pipeline
python main.py
# Verify outputs
ls -la output/*/
```

