# MediLink AI

An intelligent medical form automation system that uses AI to automatically fill Prior Authorization (PA) forms from referral packages, streamlining healthcare administrative processes.

## 🎯 Overview

MediLink AI leverages Google's Gemini AI to intelligently extract patient information from medical referral packages and automatically populate Prior Authorization forms. The system uses advanced PDF processing, OCR capabilities, and natural language processing to understand medical documents and map relevant information to form fields.

## ✨ Features

- **Intelligent Document Processing**: Extracts patient information from complex medical referral packages
- **Automated Form Filling**: Maps extracted data to appropriate fields in Prior Authorization forms
- **Context-Aware Field Mapping**: Uses AI to understand field context and make intelligent mapping decisions
- **PDF Handling**: Robust PDF processing with fallback mechanisms for complex forms
- **Streamlit Interface**: User-friendly web interface for document upload and processing

## 🏗️ Project Structure

```
MediLink-AI/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (API keys)
├── .gitignore           # Git ignore file
└── README.md            # Project documentation
```

## 🚀 How It Works

### 1. Document Upload

Users upload two PDF files through the Streamlit interface:

- **Referral Package**: Contains patient medical information, history, and treatment details
- **Prior Authorization Form**: The form that needs to be filled out

### 2. Information Extraction

The system uses Gemini AI to:

- Analyze the referral package and extract structured patient information
- Identify form fields in the PA form with their positions and types
- Generate contextual understanding of what each field represents

### 3. Intelligent Mapping

For each page of the PA form:

- **Context Analysis**: AI determines the context and meaning of each form field
- **Smart Mapping**: Matches extracted patient data to appropriate form fields based on context
- **Validation**: Ensures only relevant information is mapped to appropriate fields

### 4. Form Filling

- Automatically populates text fields and checkboxes
- Handles complex PDF forms with robust error handling
- Generates a filled PDF ready for download

## 🔧 Technical Implementation

### AI Models Used

- **Gemini 2.5 Flash**: Used for context extraction and field mapping
- **Multi-modal Processing**: Handles both text and visual elements in PDFs

### Key Technologies

- **Streamlit**: Web interface framework
- **PyMuPDF (fitz)**: PDF processing and manipulation
- **Google GenAI**: AI-powered document understanding
- **Python-dotenv**: Environment variable management

### Error Handling

- Robust PDF processing with fallback mechanisms
- XRef error handling for complex form widgets
- Image-based PDF generation when direct copying fails

## 📋 Prerequisites

- Python 3.8 or higher
- Google API key for Gemini AI
- Python virtual environment (recommended)

## 🚀 Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MisbahAN/MediLink-AI.git
cd MediLink-AI
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.env` file in the root directory and add your Gemini API key:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

**To get a Gemini API key:**

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy and paste it into the `.env` file

### 5. Run the Application

```bash
# Make sure virtual environment is activated
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Quick Start Commands

```bash
# Complete setup in one go:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Create .env file with GOOGLE_API_KEY=your_api_key_here
streamlit run app.py
```

## 💻 Usage

1. **Upload Documents**: Use the file uploaders to select your PA form and referral package PDFs
2. **Process**: Click "Process and Fill" to start the AI-powered form filling
3. **Download**: Once processing is complete, download the filled PA form

## 🔍 How the Processing Works

The application follows a sophisticated multi-step process:

1. **Patient Information Extraction**: Analyzes both documents to extract structured patient data
2. **Field Discovery**: Identifies all fillable fields in the PA form with their positions and types
3. **Context Generation**: For each page, determines what each field represents and its context
4. **Intelligent Mapping**: Matches patient information to appropriate form fields based on context understanding
5. **Form Population**: Fills the PDF with mapped data and generates the completed form

## 👨‍💻 Author

**Misbah Ahmed Nauman**

- Portfolio: [misbahan.com](https://misbahan.com)
- Built during Headstarter SWE Residency Sprint 3
