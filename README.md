# Enterprise Document Search

A simple, powerful AI-powered document search system that enables intelligent search across PDFs, Word documents, and text files. Get precise, human-readable answers from your documents.

## Features

- **Simple Document Upload**: Drag and drop PDF, DOCX, TXT files
- **Intelligent Search**: Ask questions in natural language
- **AI-Powered Answers**: Get comprehensive, human-readable responses
- **Local & Cloud Models**: Uses Google Gemini API with local model fallbacks
- **Fast Vector Search**: FAISS-powered semantic search
- **Clean Interface**: Simple Streamlit web interface

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run app.py
   ```

3. **Upload Documents**: Use the upload interface to add your documents

4. **Ask Questions**: Search using natural language queries like:
   - "What are John's skills?"
   - "What experience does Sarah have?"
   - "What technologies are mentioned?"

## Configuration

Create a `.env` file for optional settings:

```env
# Optional: Use Google Gemini API for better responses
GOOGLE_API_KEY=your_api_key_here

# Optional: Force local model usage
FORCE_LOCAL_MODEL=false

# Optional: Specify local model
LOCAL_MODEL_NAME=microsoft/DialoGPT-medium
```

**Models Available**:
- **Primary**: Google Gemini API (best quality)
- **Fallback**: DialoGPT-Large/Medium, OPT-350m, DistilGPT2

## How It Works

1. **Document Processing**: Splits documents into chunks, creates embeddings
2. **Vector Search**: Finds relevant document sections using semantic similarity  
3. **AI Generation**: Uses LLM to create human-readable answers from retrieved content
4. **Smart Caching**: Stores processed documents for fast subsequent searches

## Files Structure

```
enterprise_doc_search/
├── app.py                   # Main application
├── model_manager.py         # AI model management
├── vector_db.py            # Document storage & search
├── document_processor.py   # File processing
├── requirements.txt        # Dependencies
├── documents/              # Your uploaded documents
└── cache/                  # Processed document cache
```

## Technical Details

- **Embeddings**: `sentence-transformers/all-mpnet-base-v2` (768 dimensions)
- **Vector Store**: FAISS with cosine similarity
- **Text Splitting**: Recursive character splitter (1000 chars, 200 overlap)
- **LLM Integration**: LangChain with HuggingFace Transformers
- **Caching**: Automatic document processing cache

## Supported Formats

- PDF documents
- Word documents (.docx, .doc) 
- Text files (.txt, .md)

## Requirements

- Python 3.8+
- 4GB+ RAM recommended
- Optional: GPU for faster local model inference

## Enhanced Features (Previous Updates)

- **Hybrid Search**: Combines semantic and keyword matching
- **Better Models**: Upgraded from basic models to DialoGPT and OPT series
- **Improved Prompting**: Advanced templates for human-readable responses
- **Smart Extraction**: Specialized logic for different query types
- **Response Formatting**: Clean, structured answers with bullet points and sections
- PDF files
- Word documents (DOCX, DOC)
- Text files (TXT, MD)
- CSV files
- Excel files (XLSX, XLS)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
