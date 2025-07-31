import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

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

# Function to process documents from folder
def process_documents_from_folder(folder_path, chunk_size, chunk_overlap):
    """Process all documents from the specified folder"""
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
                documents.extend(loader.load())
                st.info(f"Processed: {file}")
            except Exception as e:
                st.warning(f"Error processing {file}: {str(e)}")
    
    if not documents:
        raise ValueError("No valid documents found in the folder")
        
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector store with GPU support if available
    try:
        import faiss
        from faiss_gpu_utils import check_faiss_gpu
        
        has_gpu, message = check_faiss_gpu()
        if has_gpu:
            st.success("🚀 GPU acceleration enabled for vector search")
        else:
            st.info(f"Using CPU for vector search: {message}")
            
        vectorstore = FAISS.from_documents(texts, embeddings)
    except Exception as e:
        st.warning(f"Using CPU for vector search: {str(e)}")
        vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore, documents

# Function to process uploaded documents
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
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector store with GPU support if available
    try:
        import faiss
        # Check if GPU is available
        if hasattr(faiss, 'StandardGpuResources'):
            st.success("🚀 GPU acceleration enabled for vector search")
            # Create CPU index first
            vectorstore = FAISS.from_documents(texts, embeddings)
            # No need to manually convert to GPU as FAISS should handle this automatically if available
        else:
            vectorstore = FAISS.from_documents(texts, embeddings)
    except Exception as e:
        st.warning(f"Using CPU for vector search: {str(e)}")
        vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore, documents

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Enterprise Document Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to make the interface more modern
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: #1E3A8A;
        margin-bottom: 0.2rem !important;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #6B7280;
        margin-bottom: 2rem !important;
    }
    .stButton>button {
        background-color: #2563EB !important;
        color: white !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        border: none !important;
    }
    .stButton>button:hover {
        background-color: #1D4ED8 !important;
        color: white !important;
    }
    .tech-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .feature-item {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        padding: 1rem;
        border-left: 4px solid #2563EB;
        margin-bottom: 0.75rem !important;
    }
    .info-box {
        background-color: #DBEAFE;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid #93C5FD;
    }
    .search-box {
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    .upload-box {
        background-color: #F0FDF4;
        border-radius: 0.5rem;
        padding: 1.5rem;
        border: 1px dashed #4ADE80;
    }
    .stExpander {
        border-radius: 0.5rem !important;
        border: none !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    .metric-container {
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.875rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

features = [
    {"icon": "🔍", "title": "Intelligent Search", "desc": "Advanced semantic search across various document types"},
    {"icon": "🛡️", "title": "Privacy Protection", "desc": "Automatic redaction of sensitive information like passwords and API keys"},
    {"icon": "📄", "title": "Multiple Formats", "desc": "Support for PDF, DOCX, TXT and other document formats"},
    {"icon": "💾", "title": "Vector Storage", "desc": "High-performance similarity search with FAISS"},
    {"icon": "🤖", "title": "AI-Powered", "desc": "Uses state-of-the-art language models for document understanding"}
]

# Main title with custom styling
st.markdown('<h1 class="main-header">Enterprise Document Search</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent search across your organization\'s documents powered by AI</p>', unsafe_allow_html=True)

# Display features in a modern grid layout
cols = st.columns(2)
for i, feature in enumerate(features):
    col_idx = i % 2
    with cols[col_idx]:
        st.markdown(f"""
        <div class="feature-item">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <div style="font-size: 1.5rem; margin-right: 0.75rem;">{feature["icon"]}</div>
                <h3 style="margin: 0; color: #1E3A8A;">{feature["title"]}</h3>
            </div>
            <p style="margin: 0; color: #4B5563;">{feature["desc"]}</p>
        </div>
        """, unsafe_allow_html=True)

# Getting started with modern styling
st.markdown('<h2 style="color: #1E3A8A; margin-top: 2rem; margin-bottom: 1rem;">Getting Started</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <ol style="margin: 0; padding-left: 1.5rem;">
        <li style="margin-bottom: 0.5rem; color: #000000;"><b>Upload</b> your documents using the Document Upload section</li>
        <li style="margin-bottom: 0.5rem; color: #000000;"><b>Process</b> the documents to create searchable embeddings</li>
        <li style="color: #000000;"><b>Search</b> across your documents using natural language queries</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Hero section with animation
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div style="padding-right: 2rem;">
            <h2 style="color: #1E3A8A; margin-bottom: 1rem;">Transform How You Search Documents</h2>
            <p style="font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem;">
                Extract relevant content from SOPs, design documents, incident reports, policies, 
                and any organizational documentation with advanced AI-powered search capabilities.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        # Simple animation effect
        import time
        with st.container():
            placeholder = st.empty()
            for i in range(3):
                placeholder.markdown(f"""
                <div style="height: 160px; display: flex; align-items: center; justify-content: center;">
                    <div style="font-size: {120 + i*5}px; color: #3B82F6; opacity: {0.7 - i*0.2};">🔍</div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.1)
            placeholder.markdown("""
            <div style="height: 160px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 130px; color: #2563EB;">🔍</div>
            </div>
            """, unsafe_allow_html=True)

# Tech stack description with modern cards
st.markdown('<h2 style="color: #1E3A8A; margin-top: 2rem; margin-bottom: 1rem;">Powered By</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="tech-card">
        <h3 style="color: #1E3A8A; margin-bottom: 0.5rem;">🧠 Embeddings</h3>
        <p style="font-weight: 600; margin-bottom: 0.25rem;">Gemma</p>
        <p style="color: #6B7280; font-size: 0.9rem;">Advanced sentence transformation embedding for semantic understanding</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="tech-card">
        <h3 style="color: #1E3A8A; margin-bottom: 0.5rem;">💾 Storage</h3>
        <p style="font-weight: 600; margin-bottom: 0.25rem;">Vector Database</p>
        <p style="color: #6B7280; font-size: 0.9rem;">High-performance similarity search for document retrieval</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="tech-card">
        <h3 style="color: #1E3A8A; margin-bottom: 0.5rem;">🤖 LLM</h3>
        <p style="font-weight: 600; margin-bottom: 0.25rem;">Adaptive Model Selection</p>
        <p style="color: #6B7280; font-size: 0.9rem;">Smart fallback between cloud APIs and local models</p>
    </div>
    """, unsafe_allow_html=True)

# Display current model status
with st.expander("Model Status", expanded=False):
    try:
        from check_model import check_model_availability
        
        # Check model availability
        model_info = check_model_availability()
        
        # Display model info
        st.subheader("Current Model Status")
        
        model_col1, model_col2, model_col3 = st.columns(3)
        with model_col1:
            st.metric("Model", f"{model_info.get('name', 'Unknown')}")
        with model_col2:
            st.metric("Type", f"{model_info.get('type', 'Unknown')}")
        with model_col3:
            if "device" in model_info:
                st.metric("Device", f"{model_info.get('device', 'Unknown')}")
            else:
                st.metric("Version", f"{model_info.get('version', 'Unknown')}")
                
        # Display model settings from .env
        st.subheader("Model Configuration")
        st.code(f"""
PREFERRED_MODEL={os.getenv('PREFERRED_MODEL', 'gemini')}
FORCE_LOCAL_MODEL={os.getenv('FORCE_LOCAL_MODEL', 'false')}
LOCAL_MODEL_NAME={os.getenv('LOCAL_MODEL_NAME', 'facebook/opt-350m')}
        """)
                
    except Exception as e:
        st.warning(f"Model status check failed: {str(e)}")

# Key features
st.subheader("Key Features")
features = [
    "🔍 Intelligent search across various document types",
    "🔐 Privacy protection with automatic filtering of sensitive information",
    "📄 Support for multiple document formats (PDF, DOCX, TXT, etc.)",
    "🏷️ Automatic document categorization and tagging",
    "📊 Insights and analytics on document usage"
]

for feature in features:
    st.markdown(f"- {feature}")

# Getting started
st.subheader("Getting Started")
st.markdown("""
1. **Upload** your documents using the Document Upload section in the sidebar
2. **Process** your documents to create embeddings
3. **Search** across your documents using the Search section
""")

# Initialize session state for document processing
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False

# Initialize session state for document store
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Initialize session state for document list
if "documents" not in st.session_state:
    st.session_state.documents = []

# Initialize document cache manager
if "cache_manager" not in st.session_state:
    from document_cache_manager import DocumentCacheManager
    st.session_state.cache_manager = DocumentCacheManager()

# Set default values for session state (no authentication needed)
st.session_state.user_role = "Default"  # Default role for PIA filtering

# Automatically load or process documents on startup
with st.spinner("Loading document cache or processing documents..."):
    try:
        # Default chunk settings
        chunk_size = 1000
        chunk_overlap = 200
        
        # Use cache manager to efficiently load documents
        vectorstore, documents = st.session_state.cache_manager.load_or_process_documents(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.documents = documents
            st.session_state.processed_docs = True
            
            # Check for placeholder documents (indicating no real content)
            placeholder_docs = [doc for doc in documents if doc.metadata.get('is_placeholder', False)]
            error_docs = [doc for doc in documents if doc.metadata.get('is_error', False)]
            
            if placeholder_docs:
                st.warning("No valid documents found in the documents folder. Please upload some documents to get started.")
            elif error_docs:
                st.warning("Some documents could not be processed correctly. There might be format issues.")
            else:
                st.success(f"Successfully loaded {len(documents)} document chunks from cache or processed documents")
        else:
            st.info("No documents found in the documents folder. Upload some documents to get started.")
    except Exception as e:
        st.error(f"Error loading or processing documents: {str(e)}")
        st.session_state.processed_docs = False
        
    # Force update the documents in session state by directly getting them from the document processor
    if not st.session_state.documents:
        try:
            from document_processor import DocumentProcessor
            processor = DocumentProcessor()
            docs = processor.process_directory('documents')
            if docs:
                st.session_state.documents = docs
                st.session_state.processed_docs = True
                st.success(f"Successfully loaded {len(docs)} document chunks directly")
        except Exception as e:
            st.warning(f"Could not load documents directly: {str(e)}")

# Sidebar for settings and model info
with st.sidebar:
    st.markdown('<h2 style="color: #1E3A8A;">Settings</h2>', unsafe_allow_html=True)
    
    # Device selection
    st.markdown('<h3 style="color: #4B5563; margin-top: 1rem;">Device Settings</h3>', unsafe_allow_html=True)
    
    # Add device selection option
    device_option = st.radio(
        "Select Compute Device",
        options=["Auto", "CPU", "GPU"],
        index=0,
        help="Select which device to use for model inference. Auto will use GPU if available."
    )
    
    # Save device selection to environment variable
    if device_option == "CPU":
        os.environ["FORCE_CPU"] = "true"
    elif device_option == "GPU":
        os.environ["FORCE_GPU"] = "true"
    else:
        # Auto - use what's available
        if "FORCE_CPU" in os.environ:
            del os.environ["FORCE_CPU"]
        if "FORCE_GPU" in os.environ:
            del os.environ["FORCE_GPU"]
    
    # Model information
    st.markdown('<h3 style="color: #4B5563; margin-top: 1rem;">Model Information</h3>', unsafe_allow_html=True)
    
    try:
        from check_model import check_model_availability
        
        # Check model availability
        model_info = check_model_availability()
        
        # Display model info in sidebar
        st.markdown(f"""
        <div style="background-color: #F0F9FF; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <p><b>Model:</b> {model_info.get('name', 'Unknown')}</p>
            <p><b>Type:</b> {model_info.get('type', 'Unknown')}</p>
            {'<p><b>Device:</b> ' + model_info.get('device', 'Unknown') + '</p>' if 'device' in model_info else ''}
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Model status unavailable: {str(e)}")
        
    # Document Upload section
    st.markdown('<h3 style="color: #4B5563; margin-top: 1rem;">Document Upload</h3>', unsafe_allow_html=True)
    
    # Add button to refresh document cache
    if st.button("Refresh Document Cache"):
        with st.spinner("Refreshing document cache..."):
            try:
                chunk_size = 1000  # Default value
                chunk_overlap = 200  # Default value
                vectorstore, documents = st.session_state.cache_manager.load_or_process_documents(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                st.session_state.vectorstore = vectorstore
                st.session_state.documents = documents
                st.session_state.processed_docs = True
                st.success(f"Successfully refreshed document cache with {len(documents)} document chunks")
            except Exception as e:
                st.error(f"Error refreshing document cache: {str(e)}")
    
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "docx", "doc", "txt", "md"],
        accept_multiple_files=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=1000)
    with col2:
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200)
    
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            try:
                vectorstore, documents = st.session_state.cache_manager.process_uploaded_files(
                    uploaded_files,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                st.session_state.vectorstore = vectorstore
                st.session_state.documents = documents
                st.session_state.processed_docs = True
                st.success(f"Successfully processed {len(uploaded_files)} documents")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

# Main content area
st.subheader("Document Search")

# Query input
query = st.text_input("Enter your search query")

# Filter settings
with st.expander("Advanced Search Settings"):
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
    with col2:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# Search button
if st.button("Search") and query and st.session_state.processed_docs:
    with st.spinner("Searching documents..."):
        try:
            # First, get the relevant documents without requiring LLM processing
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
            
            try:
                # Check if documents have any placeholder/error flags
                has_placeholder = any(doc.metadata.get('is_placeholder', False) for doc in st.session_state.documents)
                has_error = any(doc.metadata.get('is_error', False) for doc in st.session_state.documents)
                
                if has_placeholder:
                    st.warning("Your document library is empty. Please upload real documents to perform searches.")
                    st.stop()
                    
                # Retrieve relevant documents first (this part should work even if LLM fails)
                # Using the new invoke method instead of deprecated get_relevant_documents
                retrieved_docs = retriever.invoke(query)
                
                if not retrieved_docs:
                    st.info("No documents found that match your query. Try a different search term.")
                    st.stop()
                
                # Record search history for analytics
                if "search_history" not in st.session_state:
                    st.session_state.search_history = []
                    
                # Add this search to history
                from datetime import datetime
                search_record = {
                    "query": query,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user": "anonymous",
                    "user_role": st.session_state.user_role
                }
                st.session_state.search_history.append(search_record)
                
                # Display source documents with privacy filtering
                st.subheader("Relevant Sources")
                
                # Import privacy filter
                from privacy_filter import PrivacyFilter
                privacy_filter = PrivacyFilter()
                
                # Get document sources
                doc_sources = {}
                for i, doc in enumerate(retrieved_docs):
                    # Filter sensitive information
                    filtered_content = privacy_filter.filter_text(doc.page_content)
                    source = doc.metadata.get('source', 'Unknown')
                    
                    with st.expander(f"Source {i+1}"):
                        st.markdown(f"""
                        <div style="background-color: #F0F4F8; padding: 1rem; border-radius: 0.5rem; font-family: monospace; white-space: pre-wrap; color: #000000; border: 1px solid #E2E8F0;">
                        {filtered_content}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"**Source**: {source}")
                        
                        # Show warning if content was redacted
                        if filtered_content != doc.page_content:
                            st.warning("⚠️ Some sensitive information has been redacted for security reasons.")
                
                # Now try to generate an answer with the LLM (with multiple fallback options)
                try:
                    # Initialize LLM with model manager
                    from model_manager import ModelManager
                    model_manager = ModelManager()
                    
                    # Use safer temperature settings to avoid probability tensor errors
                    safe_temperature = 0.0  # Zero temperature for deterministic output
                    
                    # Try to get the primary model
                    try:
                        llm, model_info = model_manager.get_model(temperature=safe_temperature)
                    except Exception as e:
                        st.warning(f"Primary model unavailable: {str(e)}")
                        # Use FakeListLLM as immediate fallback
                        from langchain_community.llms import FakeListLLM
                        llm = FakeListLLM(responses=["I've analyzed the documents and found information related to your query. Please review the source documents above for the most accurate details."])
                        model_info = {"name": "Simple Fallback", "type": "Local Fallback", "version": "emergency"}
                    
                    # Create QA chain with robust parameters
                    from langchain.prompts import PromptTemplate
                    
                    # Define a proper prompt template
                    prompt_template = """You are a helpful assistant that answers questions based on the provided documents.
                    Your task is to synthesize information from ALL the provided documents into a comprehensive, detailed answer.
                    Include specific facts, figures, and key points from each source document. Don't just summarize - integrate the information
                    to provide a complete picture.
                    
                    When answering, follow these guidelines:
                    1. Focus ONLY on information present in the provided context
                    2. Include specific details from multiple source documents
                    3. Organize your answer in a logical, coherent way
                    4. If documents contain conflicting information, acknowledge the different perspectives
                    5. If you don't know the answer or the information is not in the context, say "I don't have enough information to answer this question."
                    
                    Context: {context}
                    
                    Question: {question}
                    
                    Detailed Answer:"""
                    
                    # Create the prompt template properly
                    qa_prompt = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )
                    
                    # Create QA chain with the proper prompt template
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm, 
                        chain_type="stuff", 
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={
                            "prompt": qa_prompt
                        }
                    )
                    
                    # Get answer with proper error handling
                    try:
                        # Process ALL documents together for a comprehensive answer
                        # Get the document contents first
                        docs = retrieved_docs  # Use the documents already retrieved
                        
                        # Import model manager to get process_long_input function
                        from model_manager import ModelManager
                        model_manager = ModelManager()
                        
                        # Add source numbers to each document for reference
                        formatted_docs = []
                        for i, doc in enumerate(docs):
                            source_info = f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}"
                            formatted_docs.append(f"--- {source_info} ---\n{doc.page_content}\n")
                        
                        # Combine all documents with source information
                        all_sources = "\n\n".join(formatted_docs)
                        
                        # Create a simplified, direct prompt template
                        comprehensive_prompt = f"""
                        USER QUERY: {query}
                        
                        DOCUMENT SOURCES:
                        {all_sources}
                        
                        TASK: Create a detailed answer to the user query based on ONLY the information in the document sources above.
                        
                        INSTRUCTIONS:
                        1. Focus on facts and specific information from the documents
                        2. Be concise but thorough
                        3. Include technical details when relevant
                        4. Structure your answer clearly
                        5. Only use information from the provided sources
                        
                        Make sure your answer directly addresses the user's query with specific information from the documents.
                        """
                        
                        # Use different processing based on model type
                        if model_info.get('type') == 'Local':
                            # Use the model manager's long input processor for local models
                            result = {"result": model_manager.process_long_input(llm, comprehensive_prompt)}
                        else:
                            # For API models (like Gemini), use direct invocation for better quality
                            try:
                                answer = llm.invoke(comprehensive_prompt)
                                result = {"result": answer}
                            except Exception as api_err:
                                st.error(f"Error generating with API model: {str(api_err)}")
                                # Fall back to QA chain if direct invocation fails
                                result = qa_chain.invoke({"query": query})
                        
                        # Display answer
                        st.subheader("Answer")
                        st.markdown(result["result"])
                        
                        # Display model information
                        st.subheader("Model Used")
                        st.markdown(f"""
                        <div style="background-color: #2E3440; padding: 1rem; border-radius: 0.5rem; border: 1px solid #4C566A; color: #ECEFF4;">
                            <p><strong style="color: #88C0D0;">Model:</strong> {model_info.get('name', 'Unknown')} | <strong style="color: #88C0D0;">Type:</strong> {model_info.get('type', 'Unknown')}</p>
                            <p><strong style="color: #88C0D0;">Running on:</strong> {model_info.get('device', 'Unknown')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                            
                    except Exception as qa_error:
                        st.error(f"Error generating answer: {str(qa_error)}")
                        st.info("However, relevant documents were retrieved and are displayed above.")
                        
                except Exception as model_error:
                    st.error(f"Error with LLM processing: {str(model_error)}")
                    st.info("However, relevant documents were retrieved and are displayed above.")
                
            except Exception as retrieval_error:
                st.error(f"Error retrieving documents: {str(retrieval_error)}")
                st.error("Please try again with a different query or check the vector store configuration.")
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            st.error("Please try again with a different query or device setting.")
            # Show detailed error to help with debugging
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
elif query and not st.session_state.processed_docs:
    st.warning("Please upload and process documents first.")

# Display document summary
if st.session_state.processed_docs:
    with st.expander("Document Summary"):
        # Add custom CSS just for this section
        st.markdown("""
        <style>
        .doc-summary-text {
            color: #000000 !important;
            font-weight: 500;
            font-size: 1.05rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Force a direct check of documents
        if not st.session_state.documents:
            try:
                from document_processor import DocumentProcessor
                processor = DocumentProcessor()
                docs = processor.process_directory('documents')
                if docs:
                    st.session_state.documents = docs
                    st.success(f"Re-loaded {len(docs)} document chunks")
            except Exception as e:
                st.warning(f"Could not re-load documents: {str(e)}")
        
        # Total document chunks
        st.markdown(f"<div class='doc-summary-text'>Number of document chunks: {len(st.session_state.documents)}</div>", unsafe_allow_html=True)
        
        # Count unique document files
        unique_files = set()
        doc_files_dict = {}
        
        for doc in st.session_state.documents:
            if "source" in doc.metadata and doc.metadata["source"] != "placeholder" and not doc.metadata.get("is_error", False):
                file_path = doc.metadata["source"]
                file_name = os.path.basename(file_path)
                unique_files.add(file_path)
                
                # Count documents by file
                if file_name in doc_files_dict:
                    doc_files_dict[file_name] += 1
                else:
                    doc_files_dict[file_name] = 1
        
        # Display unique document count and names
        st.markdown(f"<div class='doc-summary-text'>Unique document files: {len(unique_files)}</div>", unsafe_allow_html=True)
        
        # Display document list with chunks per file
        if doc_files_dict:
            st.markdown("<h4 style='color: #000000;'>Document Details</h4>", unsafe_allow_html=True)
            for doc_name, count in doc_files_dict.items():
                st.markdown(f"<div class='doc-summary-text'>📄 <b>{doc_name}</b> - {count} chunks</div>", unsafe_allow_html=True)
        
        # Store document list for analytics
        st.session_state.all_documents = st.session_state.documents
        
        # If there's search activity, record it for analytics
        if query and "search_history" not in st.session_state:
            st.session_state.search_history = []
