# Enterprise Document Search

A GenAI-powered solution that enables intelligent search across various documents, including SOPs, design documents, incident reports, policies, and any customer-requested documentation. It focuses on extracting relevant content from within documents to support faster and more accurate information retrieval.

## Technology Stack

- **Gemma**: For sentence transformation embedding
- **Vector Database**: For storing and querying embeddings
- **Adaptive LLM Selection**: 
  - Primary: Google Gemini API 
  - Fallback: Local Hugging Face models (Mistral-7B or TinyLlama-1.1B)
- **Streamlit**: Frontend interface

## Features

- Intelligent search across various document types
- Privacy protection with automatic filtering of sensitive information
- Document processing for various file formats (PDF, DOCX, TXT, etc.)
- Relevance-based search results
- GPU acceleration support with UI toggle
- Modern, responsive user interface

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file from the template:
   ```
   cp .env.example .env
   ```
4. Update the `.env` file with your API keys and configuration

### Model Selection Configuration

In the `.env` file, you can configure the model selection behavior:

```
# Model Settings
PREFERRED_MODEL=gemini  # Options: gemini, local
FORCE_LOCAL_MODEL=false  # Set to true to force using local model even if API key is present
LOCAL_MODEL_NAME=facebook/opt-350m  # Open source model that doesn't require HF login
```

- If `GOOGLE_API_KEY` is missing or invalid, the system will automatically fall back to local models
- Set `FORCE_LOCAL_MODEL=true` to always use local models regardless of API key
- You can specify any Hugging Face model as the `LOCAL_MODEL_NAME`
- The application includes a device selector in the UI to switch between CPU and GPU

## Usage

Run the application:

```
streamlit run app.py
```

Or use the provided batch script:

```
run_app.bat
```

## Project Structure

```
enterprise_doc_search/
│
├── app.py                   # Main Streamlit application (all-in-one interface)
├── pages/                   # Additional pages (optional, for multi-page mode)
│   ├── 01_Upload_Documents.py  # Document upload and processing
│   ├── 02_Search_Documents.py  # Search interface
│   ├── 03_Analytics.py         # Analytics dashboard
│   └── 04_Settings.py          # Settings and admin panel
├── document_processor.py    # Document processing utilities
├── vector_db.py             # Vector database management
├── llm_access_control.py    # LLM with access control
├── pia_access_control.py    # Personally Identifiable Access control
├── model_manager.py         # Adaptive model selection with fallback
├── requirements.txt         # Project dependencies
├── run_app.bat              # Script to run the application
└── .env.example             # Environment variables template
```

## Access Control

The system implements a role-based access control system with three default roles:
- **Admin**: Full access to all documents and features
- **Analyst**: Can access public, internal, and confidential documents, and upload new documents
- **Viewer**: Can only access public and internal documents

## Vector Database

The solution uses a FAISS vector database for storing document embeddings. The database can be persisted to disk and loaded on startup.

## Document Processing

The document processor supports multiple file formats:
- PDF files
- Word documents (DOCX, DOC)
- Text files (TXT, MD)
- CSV files
- Excel files (XLSX, XLS)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
