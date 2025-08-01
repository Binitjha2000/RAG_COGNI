import streamlit as st
import os
from config import config
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.chains import RetrievalQA
from vector_db import EnhancedVectorDatabaseManager, SearchConfig
from model_manager import ModelManager
import time

# Configure Streamlit page using centralized config
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state=config.SIDEBAR_STATE
)

# Advanced Sidebar with Navigation and Settings
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); 
                padding: 2rem 1rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
        <h2 style="color: white; margin: 0; font-family: 'Inter', sans-serif; font-weight: 700;">
            🔍 RAG System
        </h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            Document Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Menu
    st.markdown("### 🧭 Navigation")
    nav_option = st.selectbox(
        "Choose Section:",
        ["🏠 Main Search", "📊 Analytics", "⚙️ Settings", "📚 Documentation"],
        index=0
    )
    
    # Quick Stats if documents are processed
    if st.session_state.get("processed", False):
        st.markdown("### 📈 Quick Stats")
        vector_db_manager = st.session_state.get("vector_db_manager")
        if vector_db_manager:
            doc_stats = vector_db_manager.get_document_stats()
            search_stats = vector_db_manager.get_search_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Documents", doc_stats.get('total_documents', 0))
                st.metric("🔍 Searches", search_stats.get('total_searches', 0))
            with col2:
                st.metric("👤 CV Docs", doc_stats.get('cv_documents', 0))
                st.metric("🎯 Person Queries", search_stats.get('person_searches', 0))
    
    # System Configuration
    st.markdown("### ⚙️ System Configuration")
    
    # Model Selection
    model_option = st.selectbox(
        "AI Model:",
        ["🤖 Simple Processor (Default)", "🧠 Gemini (API Required)", "🔧 Auto-Select"],
        help="Choose the AI model for response generation"
    )
    
    # Search Configuration
    search_config = st.expander("🔧 Search Settings")
    with search_config:
        default_alpha = st.slider("Semantic vs Keyword Balance", 0.0, 1.0, 0.5, 0.1,
                                 help="0.0 = Pure keyword, 1.0 = Pure semantic")
        enable_person_filter = st.checkbox("Enable Person Filtering", value=True)
        max_results = st.slider("Max Results", 1, 20, 5)
    
    # Performance Monitoring
    st.markdown("### 📊 Performance")
    perf_expander = st.expander("View System Performance")
    with perf_expander:
        if st.session_state.get("processed", False):
            st.success("✅ System Ready")
            st.info("🚀 Vector DB: Active")
            st.info("🧠 AI Model: Loaded")
        else:
            st.warning("⏳ Waiting for documents")
    
    # Help & Documentation
    st.markdown("### 💡 Help & Support")
    help_expander = st.expander("Quick Help")
    with help_expander:
        st.markdown("""
        **🔍 Search Tips:**
        - Use person names for targeted searches
        - Ask specific questions about skills/experience
        - Use natural language queries
        
        **📄 Document Support:**
        - PDF, DOCX, TXT, MD files
        - Automatic content extraction
        - Metadata preservation
        
        **🤖 AI Features:**
        - Semantic understanding
        - Person-specific filtering
        - Contextual responses
        """)
    
    # About Section
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                border-radius: 10px; margin-top: 1rem;">
        <p style="margin: 0; font-size: 0.8rem; color: #64748b;">
            <strong>Enhanced RAG System</strong><br>
            Version 2.0 • AI-Powered<br>
            Built with ❤️ using Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

# Function to check file extension
def get_file_loader(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".pdf":
        return PyPDFLoader(file_path)
    elif file_extension in [".docx", ".doc"]:
        return Docx2txtLoader(file_path)
    elif file_extension in [".txt", ".md"]:
        return TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

# Enhanced function to process documents from folder with better metadata
def process_documents_from_folder(folder_path, chunk_size, chunk_overlap):
    """Process all documents from the specified folder with enhanced metadata"""
    documents = []
    
    # Walk through the folder to find all files
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip non-document files
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension not in [".pdf", ".docx", ".doc", ".txt", ".md"]:
                continue
            
            try:
                # Get appropriate loader
                loader = get_file_loader(file_path)
                
                # Load documents
                loaded_docs = loader.load()
                
                # Enhanced metadata addition
                for doc in loaded_docs:
                    # Add comprehensive metadata
                    doc.metadata.update({
                        'source': file_path,
                        'filename': os.path.basename(file_path),
                        'file_extension': file_extension,
                        'file_size': os.path.getsize(file_path),
                        'processed_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                documents.extend(loaded_docs)
                st.info(f"Processed: {file}")
                
            except Exception as e:
                st.warning(f"Error processing {file}: {str(e)}")
    
    if not documents:
        raise ValueError("No valid documents found in the folder")
    
    # Split documents into chunks using centralized config
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    
    # Enhanced search configuration
    search_config = SearchConfig(
        alpha=0.4,  # Slightly favor keyword search for better person matching
        enable_person_filtering=True,
        person_name_threshold=0.7,  # Lower threshold for better matching
        cv_document_boost=3.0,  # Higher boost for CV documents
        enable_reranking=True,
        max_final_results=5
    )
    
    # Create enhanced vector database manager
    vector_db_manager = EnhancedVectorDatabaseManager(
        embedding_model=config.EMBEDDING_MODEL_NAME,
        config=search_config
    )
    
    try:
        # Create vector store with document classification
        vectorstore = vector_db_manager.create_vectorstore(texts)
        
        # Display document classification stats
        doc_stats = vector_db_manager.get_document_stats()
        st.success(f"🚀 Enhanced search system initialized!")
        st.info(f"📊 Found {doc_stats['cv_documents']} CV documents out of {doc_stats['total_documents']} total documents")
        
        return vector_db_manager, documents
        
    except Exception as e:
        st.warning(f"Enhanced search initialization failed, using fallback: {str(e)}")
        # Fallback to basic FAISS
        vectorstore = FAISS.from_documents(texts, HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': config.EMBEDDING_DEVICE}
        ))
        return vectorstore, documents

# Enhanced function to process uploaded documents
def process_documents(uploaded_files, chunk_size, chunk_overlap):
    # Use the documents folder to store uploaded files
    docs_dir = os.path.join(os.getcwd(), "documents")
    os.makedirs(docs_dir, exist_ok=True)
    
    documents = []
    for uploaded_file in uploaded_files:
        # Save uploaded file to documents directory
        file_path = os.path.join(docs_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Load document
            loader = get_file_loader(file_path)
            loaded_docs = loader.load()
            
            # Enhanced metadata addition
            for doc in loaded_docs:
                doc.metadata.update({
                    'source': file_path,
                    'filename': uploaded_file.name,
                    'file_extension': os.path.splitext(uploaded_file.name)[1].lower(),
                    'file_size': uploaded_file.size,
                    'processed_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            documents.extend(loaded_docs)
            
        except Exception as e:
            st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    
    # Enhanced search configuration
    search_config = SearchConfig(
        alpha=0.4,  # Slightly favor keyword search for better person matching
        enable_person_filtering=True,
        person_name_threshold=0.7,
        cv_document_boost=3.0,
        enable_reranking=True,
        max_final_results=5
    )
    
    # Create enhanced vector database manager
    vector_db_manager = EnhancedVectorDatabaseManager(
        embedding_model=config.EMBEDDING_MODEL_NAME,
        config=search_config
    )
    
    try:
        # Create vector store with document classification
        vectorstore = vector_db_manager.create_vectorstore(texts)
        
        # Display document classification stats
        doc_stats = vector_db_manager.get_document_stats()
        st.success(f"🚀 Enhanced search system initialized!")
        st.info(f"📊 Found {doc_stats['cv_documents']} CV documents out of {doc_stats['total_documents']} total documents")
        
        return vector_db_manager, documents
        
    except Exception as e:
        st.warning(f"Enhanced search initialization failed, using fallback: {str(e)}")
        # Fallback to basic FAISS
        vectorstore = FAISS.from_documents(texts, HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': config.EMBEDDING_DEVICE}
        ))
        return vectorstore, documents

# Advanced CSS styling with professional color scheme and rich content
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* Main app styling with dark gradient */
.stApp {
    background: linear-gradient(135deg, #1e293b 0%, #334155 50%, #475569 100%);
    min-height: 100vh;
}

/* Override Streamlit's default styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 25px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.15);
    margin: 1rem;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #1e293b;
}

/* Ensure all text has proper contrast */
.main .block-container * {
    color: #1e293b;
}

.main .block-container h1, .main .block-container h2, .main .block-container h3, .main .block-container h4 {
    color: #1e293b !important;
}

.main .block-container p, .main .block-container span, .main .block-container div {
    color: #475569;
}

/* Header styling with modern gradient */
.main-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    padding: 4rem 2rem;
    border-radius: 25px;
    margin-bottom: 3rem;
    color: white;
    text-align: center;
    box-shadow: 0 20px 50px rgba(99, 102, 241, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: glow 4s ease-in-out infinite alternate;
}

@keyframes glow {
    0% { transform: scale(1) rotate(0deg); }
    100% { transform: scale(1.1) rotate(5deg); }
}

.main-header h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    margin-bottom: 1rem;
    text-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    position: relative;
    z-index: 2;
}

.main-header p {
    font-family: 'Inter', sans-serif;
    font-size: 1.2rem;
    opacity: 0.95;
    font-weight: 400;
    position: relative;
    z-index: 2;
    max-width: 800px;
    margin: 0 auto;
}

/* Feature cards with vibrant colors */
.feature-card {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    padding: 2.5rem;
    border-radius: 20px;
    border: 1px solid rgba(99, 102, 241, 0.1);
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    transition: all 0.4s ease;
    font-family: 'Inter', sans-serif;
    position: relative;
    overflow: hidden;
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7);
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(99, 102, 241, 0.2);
    border-color: rgba(99, 102, 241, 0.3);
}

.feature-card h3 {
    color: #1e293b;
    font-weight: 700;
    margin-bottom: 1rem;
    font-size: 1.3rem;
}

.feature-card p {
    color: #475569;
    font-weight: 400;
    line-height: 1.7;
    font-size: 1rem;
}

/* RAG Architecture Info Box */
.rag-info {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    padding: 2rem;
    border-radius: 20px;
    border: 1px solid rgba(245, 158, 11, 0.3);
    margin: 2rem 0;
    box-shadow: 0 10px 30px rgba(245, 158, 11, 0.1);
    font-family: 'Inter', sans-serif;
}

.rag-info h3 {
    color: #92400e;
    font-weight: 700;
    margin-bottom: 1rem;
}

.rag-info p {
    color: #451a03;
    line-height: 1.6;
}

/* Section headers */
.section-header {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    padding: 2rem;
    border-radius: 20px;
    margin: 2rem 0;
    border: 1px solid rgba(99, 102, 241, 0.1);
    position: relative;
}

.section-header h2 {
    color: #1e293b;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    margin: 0;
    font-size: 1.8rem;
}

/* Search results with better styling */
.search-result {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    padding: 2rem;
    border-radius: 15px;
    border: 1px solid rgba(99, 102, 241, 0.15);
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
    font-family: 'Inter', sans-serif;
    transition: all 0.3s ease;
}

.search-result:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.12);
}

.search-result strong {
    color: #6366f1;
    font-weight: 600;
    font-size: 1.1rem;
}

.search-result small {
    color: #64748b;
    font-size: 0.9rem;
    background: #f1f5f9;
    padding: 0.5rem;
    border-radius: 8px;
    display: inline-block;
    margin-top: 0.5rem;
}

/* Debug info with tech styling */
.debug-info {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 2rem;
    border-radius: 15px;
    border-left: 4px solid #06b6d4;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9em;
    color: #e2e8f0;
    box-shadow: 0 10px 30px rgba(6, 182, 212, 0.2);
}

.debug-info strong {
    color: #06b6d4;
}

/* Answer section with premium styling */
.answer-section {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    padding: 3rem;
    border-radius: 20px;
    border: 1px solid rgba(6, 182, 212, 0.2);
    margin: 2rem 0;
    box-shadow: 0 15px 35px rgba(6, 182, 212, 0.1);
    font-family: 'Inter', sans-serif;
    position: relative;
    color: #1e293b;
}

.answer-section * {
    color: #1e293b !important;
}

.answer-section::before {
    content: '💡';
    position: absolute;
    top: 1rem;
    right: 1rem;
    font-size: 2rem;
}

/* Enhanced buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: none;
    border-radius: 12px;
    color: white;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    padding: 0.8rem 2rem;
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
    transition: all 0.3s ease;
    font-size: 1rem;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(99, 102, 241, 0.5);
    background: linear-gradient(135deg, #5b5bf6 0%, #7c3aed 100%);
}

/* Text inputs with modern styling */
.stTextInput > div > div > input {
    border-radius: 12px;
    border: 2px solid rgba(99, 102, 241, 0.2);
    font-family: 'Inter', sans-serif;
    padding: 1rem;
    font-size: 1rem;
    transition: all 0.3s ease;
}

.stTextInput > div > div > input:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

/* Metrics with colorful styling */
.css-1xarl3l {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-radius: 15px;
    border: 1px solid rgba(6, 182, 212, 0.2);
    padding: 1.5rem;
    transition: all 0.3s ease;
}

.css-1xarl3l:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(6, 182, 212, 0.15);
}

/* Success/Error messages with better colors */
.stSuccess {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 2px solid rgba(34, 197, 94, 0.3);
    border-radius: 12px;
    padding: 1rem;
}

.stError {
    background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
    border: 2px solid rgba(239, 68, 68, 0.3);
    border-radius: 12px;
    padding: 1rem;
}

.stWarning {
    background: linear-gradient(135deg, #fffbeb 0%, #fed7aa 100%);
    border: 2px solid rgba(245, 158, 11, 0.3);
    border-radius: 12px;
    padding: 1rem;
}

.stInfo {
    background: linear-gradient(135deg, #f0f9ff 0%, #dbeafe 100%);
    border: 2px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1rem;
}

/* Footer with premium styling */
.footer {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    color: white;
    padding: 3rem;
    border-radius: 20px;
    text-align: center;
    margin-top: 3rem;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.2);
}

/* Stats cards */
.stats-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    padding: 1.5rem;
    border-radius: 15px;
    border: 1px solid rgba(99, 102, 241, 0.1);
    text-align: center;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.stats-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
}

.stats-card h4 {
    color: #1e293b !important;
    font-weight: 700;
    margin-bottom: 1rem;
}

.stats-card p {
    color: #475569 !important;
    line-height: 1.6;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    border-radius: 10px;
    padding: 0.5rem 1rem;
    border: 1px solid rgba(99, 102, 241, 0.1);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
}

/* Advanced Sidebar styling */
.css-1d391kg {
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    border-right: 2px solid rgba(99, 102, 241, 0.1);
}

.css-1d391kg .stSelectbox label {
    color: #1e293b !important;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
}

.css-1d391kg .stMetric {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    padding: 0.8rem;
    border-radius: 10px;
    border: 1px solid rgba(99, 102, 241, 0.1);
    margin: 0.3rem 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.css-1d391kg .stExpander {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 12px;
    border: 1px solid rgba(99, 102, 241, 0.1);
    margin: 0.8rem 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

/* Sidebar text contrast */
.css-1d391kg * {
    color: #1e293b !important;
}

.css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
    color: #1e293b !important;
}
</style>
""", unsafe_allow_html=True)

features = [
    {"icon": "🔍", "title": "Intelligent Semantic Search", "desc": "Advanced vector-based similarity search using state-of-the-art embeddings for contextual understanding and precise document retrieval"},
    {"icon": "📄", "title": "Multi-Format Document Support", "desc": "Seamlessly process PDF, DOCX, TXT, and Markdown files with intelligent content extraction and metadata preservation"},
    {"icon": "💾", "title": "High-Performance Vector Storage", "desc": "Lightning-fast similarity search powered by FAISS (Facebook AI Similarity Search) with optimized indexing for large document collections"},
    {"icon": "🤖", "title": "AI-Powered RAG Architecture", "desc": "Retrieval-Augmented Generation combining document retrieval with large language models for contextually aware, accurate responses"},
    {"icon": "👤", "title": "Person-Specific Intelligence", "desc": "Advanced named entity recognition and person-specific filtering for targeted searches across CVs, resumes, and professional documents"},
    {"icon": "⚡", "title": "Hybrid Search Technology", "desc": "Combines semantic search with keyword matching for optimal retrieval performance across different query types and document structures"}
]

# Main title with enhanced styling
st.markdown("""
<div class="main-header">
    <h1>🔍 Enhanced Document Search System</h1>
    <p>Next-generation AI-powered document intelligence platform leveraging Retrieval-Augmented Generation (RAG) architecture for enterprise-grade document search, analysis, and knowledge extraction</p>
</div>
""", unsafe_allow_html=True)

# RAG Architecture Information Section
st.markdown("""
<div class="rag-info">
    <h3>🧠 About RAG (Retrieval-Augmented Generation) Architecture</h3>
    <p><strong>RAG</strong> represents a revolutionary approach that combines the power of information retrieval with generative AI. Our system first retrieves the most relevant documents using advanced vector similarity search, then uses this context to generate accurate, contextually-aware responses. This hybrid approach ensures factual accuracy while maintaining the flexibility of large language models.</p>
</div>
""", unsafe_allow_html=True)

# Technical Architecture Overview
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="stats-card">
        <h4 style="color: #1e293b !important;">🔄 Document Processing Pipeline</h4>
        <p style="color: #475569 !important;">• Text extraction & chunking<br>
        • Vector embedding generation<br>
        • FAISS index optimization<br>
        • Metadata enrichment</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stats-card">
        <h4 style="color: #1e293b !important;">🎯 Search & Retrieval Engine</h4>
        <p style="color: #475569 !important;">• Semantic similarity matching<br>
        • Hybrid keyword + vector search<br>
        • Person-specific filtering<br>
        • Relevance score ranking</p>
    </div>
    """, unsafe_allow_html=True)

# Display features in a comprehensive grid layout
st.markdown("""
<div class="section-header">
    <h2>🚀 Platform Capabilities & Features</h2>
</div>
""", unsafe_allow_html=True)

cols = st.columns(3)
for i, feature in enumerate(features):
    col_idx = i % 3
    with cols[col_idx]:
        st.markdown(f"""
        <div class="feature-card">
            <h3>{feature["icon"]} {feature["title"]}</h3>
            <p>{feature["desc"]}</p>
        </div>
        """, unsafe_allow_html=True)

# Document upload section with enhanced styling
st.markdown("""
<div class="section-header">
    <h2>📤 Document Upload & Processing</h2>
</div>
""", unsafe_allow_html=True)

# Tabs for different upload methods
tab1, tab2 = st.tabs(["Upload Files", "Use Documents Folder"])

with tab1:
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "docx", "doc", "txt", "md"],
        accept_multiple_files=True,
        help="Upload PDF, Word documents, or text files"
    )

with tab2:
    use_folder = st.checkbox("Use documents from 'documents' folder", value=False)
    if use_folder:
        docs_folder = os.path.join(os.getcwd(), "documents")
        if os.path.exists(docs_folder):
            files_in_folder = [f for f in os.listdir(docs_folder) 
                             if f.lower().endswith(('.pdf', '.docx', '.doc', '.txt', '.md'))]
            st.info(f"Found {len(files_in_folder)} documents in folder: {', '.join(files_in_folder[:5])}")
        else:
            st.warning("Documents folder not found. Please create a 'documents' folder and add your files.")

# Document processing
if uploaded_files or (use_folder and 'docs_folder' in locals()):
    with st.expander("⚙️ Processing Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk Size", 500, 2000, config.DEFAULT_CHUNK_SIZE, 100)
        with col2:
            chunk_overlap = st.slider("Chunk Overlap", 50, 500, config.DEFAULT_CHUNK_OVERLAP, 50)

    if st.button("🚀 Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            try:
                if uploaded_files:
                    vector_db_manager, documents = process_documents(uploaded_files, chunk_size, chunk_overlap)
                else:
                    vector_db_manager, documents = process_documents_from_folder(docs_folder, chunk_size, chunk_overlap)
                
                st.session_state.vector_db_manager = vector_db_manager
                st.session_state.documents = documents
                st.session_state.processed = True
                
                st.success(f"✅ Successfully processed {len(documents)} documents!")
                
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")

# Document search section with enhanced styling
if st.session_state.get("processed", False):
    st.markdown("""
    <div class="section-header">
        <h2>🔍 Intelligent Document Search & Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Search interface
    search_query = st.text_input(
        "Enter your search query",
        placeholder="e.g., what is SHAILENDRA RAJ SI.NGH skills & experience & education?",
        help="Ask questions about specific people, skills, or document content"
    )
    
    # Advanced search settings
    with st.expander("🔧 Advanced Search Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            num_results = st.slider("Number of results", 1, 10, 2)
        with col2:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        with col3:
            search_method = st.selectbox(
                "Search Method", 
                ["Adaptive", "Person-Specific", "Hybrid", "Semantic", "Keyword"],
                help="Adaptive automatically chooses the best method"
            )
    
    # Search execution
    if search_query and st.button("🔍 Search", type="primary"):
        with st.spinner("Searching documents..."):
            try:
                vector_db_manager = st.session_state.vector_db_manager
                
                # Debug: Display query analysis
                query_analysis = vector_db_manager.query_processor.analyze_query(search_query)
                
                with st.expander("🔍 Debug: Query Analysis", expanded=True):
                    st.markdown(f"""
                    <div class="debug-info">
                    <strong>Query Type:</strong> {query_analysis.get('query_type', 'unknown')}<br>
                    <strong>Is Person Query:</strong> {query_analysis.get('is_person_query', False)}<br>
                    <strong>Target Person:</strong> {query_analysis.get('target_person', 'None')}<br>
                    <strong>Person Names Detected:</strong> {query_analysis.get('person_names', [])}<br>
                    <strong>Suggested Alpha:</strong> {query_analysis.get('suggested_alpha', 0.5)}<br>
                    <strong>Keywords:</strong> {', '.join(query_analysis.get('keywords', []))}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Execute search based on selected method
                if search_method == "Adaptive":
                    search_results = vector_db_manager.adaptive_search(search_query, num_results)
                elif search_method == "Person-Specific" and query_analysis.get('target_person'):
                    search_results = vector_db_manager.person_specific_search(
                        search_query, query_analysis['target_person'], num_results
                    )
                elif search_method == "Hybrid":
                    search_results = vector_db_manager.hybrid_search(search_query, num_results)
                elif search_method == "Semantic":
                    search_results = vector_db_manager.semantic_search(search_query, num_results)
                elif search_method == "Keyword":
                    search_results = vector_db_manager.keyword_search(search_query, num_results)
                else:
                    search_results = vector_db_manager.adaptive_search(search_query, num_results)
                
                if not search_results:
                    st.warning("🔍 No relevant documents found. Try rephrasing your query or check if documents contain the requested information.")
                else:
                    # Display search results with enhanced formatting
                    st.subheader("📋 Relevant Sources")
                    
                    relevant_docs = []
                    for i, result in enumerate(search_results):
                        doc = result.document
                        filename = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        
                        st.markdown(f"""
                        <div class="search-result">
                        <strong>Source {i+1}:</strong> {filename}<br>
                        <small>
                        Search Method: {result.search_method} | 
                        Combined Score: {result.combined_score:.3f} | 
                        Semantic: {result.semantic_score:.3f} | 
                        Keyword: {result.keyword_score:.3f}
                        {f" | Person Match: {result.person_match_score:.3f}" if hasattr(result, 'person_match_score') and result.person_match_score > 0 else ""}
                        </small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        relevant_docs.append(doc)
                    
                    # Generate answer using LLM
                    if relevant_docs:
                        try:
                            model_manager = ModelManager()
                            llm, model_info = model_manager.get_model(temperature=temperature)
                            
                            # Create context from search results
                            context_parts = []
                            for i, doc in enumerate(relevant_docs):
                                context_parts.append(f"Document {i+1}: {doc.page_content}")
                            
                            context = "\n\n".join(context_parts)
                            
                            # Create prompt
                            prompt = f"""Question: {search_query}

Context:
{context}

Please provide a comprehensive answer based on the context above. Focus on the specific person or topic mentioned in the question."""
                            
                            # Debug: Show prompt being sent
                            with st.expander("🔍 Debug: Prompt being sent to model"):
                                st.text(prompt[:500] + "..." if len(prompt) > 500 else prompt)
                            
                            # Generate response
                            response = llm.invoke(prompt)
                            
                            # Display answer with enhanced styling
                            st.subheader("💬 Answer")
                            st.markdown(f"""
                            <div class="answer-section">
                            {response}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display model information
                            with st.expander("🤖 Model Information"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Model", model_info.get('name', 'Unknown'))
                                with col2:
                                    st.metric("Type", model_info.get('type', 'Unknown'))
                                with col3:
                                    if 'device' in model_info:
                                        st.metric("Device", model_info.get('device', 'Unknown'))
                        
                        except Exception as e:
                            st.error(f"❌ Error generating answer: {str(e)}")
                            st.info("📄 Showing raw document content instead:")
                            for i, doc in enumerate(relevant_docs):
                                st.markdown(f"**Document {i+1}:**")
                                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    
                    # Display search statistics
                    search_stats = vector_db_manager.get_search_stats()
                    with st.expander("📊 Search Statistics"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Searches", search_stats['total_searches'])
                        with col2:
                            st.metric("Person Searches", search_stats.get('person_searches', 0))
                        with col3:
                            st.metric("Hybrid Searches", search_stats['hybrid_searches'])
                        with col4:
                            st.metric("Semantic Searches", search_stats['semantic_searches'])
                        
            except Exception as e:
                st.error(f"❌ Search error: {str(e)}")
                st.info("💡 Try uploading documents first or check your query format.")

else:
    st.info("📤 Please upload documents or use the documents folder to start searching.")

# Enhanced footer with comprehensive information
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>🔍 Enhanced Document Search System</h3>
    <p><strong>Enterprise-Grade RAG Architecture</strong></p>
    <p>Powered by Advanced Vector Search • AI Language Models • Person-Specific Intelligence • Real-time Processing</p>
</div>
""", unsafe_allow_html=True)

# Technology stack information in columns
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin-top: 1rem;">
        <h4 style="color: #06b6d4; margin-bottom: 1rem;">🧠 AI Models</h4>
        <p style="margin: 0; line-height: 1.6;">
            HuggingFace Transformers<br>
            Sentence Transformers<br>
            LangChain Integration
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin-top: 1rem;">
        <h4 style="color: #10b981; margin-bottom: 1rem;">⚡ Performance</h4>
        <p style="margin: 0; line-height: 1.6;">
            FAISS Vector DB<br>
            Real-time Search<br>
            Scalable Architecture
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin-top: 1rem;">
        <h4 style="color: #f59e0b; margin-bottom: 1rem;">🔒 Enterprise Features</h4>
        <p style="margin: 0; line-height: 1.6;">
            Document Classification<br>
            Metadata Enrichment<br>
            Advanced Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #64748b; margin-top: 2rem; padding: 1rem;">
    <p style="font-size: 0.9rem; margin: 0;">
        Built with Streamlit • FAISS • HuggingFace • LangChain • Python
    </p>
</div>
""", unsafe_allow_html=True)
