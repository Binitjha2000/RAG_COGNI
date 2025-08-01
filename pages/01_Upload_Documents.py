import streamlit as st
import os
from dotenv import load_dotenv
import time

from document_processor import DocumentProcessor
from vector_db import VectorDatabaseManager

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Upload Documents",
    page_icon="📄",
    layout="wide"
)

# Main title
st.title("Upload Documents")
st.markdown("Upload and process documents to make them searchable.")

# Initialize session state for document processing
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "all_documents" not in st.session_state:
    st.session_state.all_documents = []

# Authentication check
if not st.session_state.get("user_role"):
    st.warning("Please login on the home page to access this functionality.")
    st.stop()

# Access control
if st.session_state.user_role == "Viewer":
    st.error("You do not have permission to upload documents. Please contact an administrator or analyst.")
    st.stop()

# Document upload section
st.subheader("Upload New Documents")

# Document settings
with st.expander("Document Processing Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input(
            "Chunk Size", 
            min_value=100, 
            max_value=2000, 
            value=int(os.getenv("DEFAULT_CHUNK_SIZE", 800))
        )
    with col2:
        chunk_overlap = st.number_input(
            "Chunk Overlap", 
            min_value=0, 
            max_value=500, 
            value=int(os.getenv("DEFAULT_CHUNK_OVERLAP", 100))
        )
    
    # Document metadata options
    st.subheader("Default Document Metadata")
    col1, col2 = st.columns(2)
    with col1:
        default_classification = st.selectbox(
            "Classification",
            ["Public", "Internal", "Confidential", "Restricted"],
            index=1
        )
    with col2:
        default_department = st.selectbox(
            "Department",
            ["IT", "HR", "Finance", "Marketing", "Operations", "Legal"],
            index=["IT", "HR", "Finance", "Marketing", "Operations", "Legal"].index(
                st.session_state.get("user_department", "IT")
            )
        )
    
    shared_with_all = st.checkbox("Share with all departments", value=False)

# File uploader
uploaded_files = st.file_uploader(
    "Upload Documents", 
    type=["pdf", "docx", "doc", "txt", "md", "csv", "xlsx", "xls"],
    accept_multiple_files=True
)

if uploaded_files:
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(os.getcwd(), "temp_documents")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Display uploaded files
    st.write(f"Uploaded {len(uploaded_files)} files:")
    for file in uploaded_files:
        st.markdown(f"- **{file.name}** ({file.type})")
    
    # Process documents button
    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            try:
                # Save uploaded files to temporary directory
                saved_files = []
                for uploaded_file in uploaded_files:
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_files.append(temp_file_path)
                
                # Initialize document processor
                doc_processor = DocumentProcessor(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process documents
                all_docs = []
                for i, file_path in enumerate(saved_files):
                    status_text.text(f"Processing {os.path.basename(file_path)}...")
                    
                    # Process file
                    docs = doc_processor.process_file(file_path)
                    
                    # Add metadata
                    for doc in docs:
                        doc.metadata["classification"] = default_classification
                        doc.metadata["department"] = default_department
                        doc.metadata["shared"] = shared_with_all
                        doc.metadata["upload_user"] = st.session_state.user_name
                        doc.metadata["upload_role"] = st.session_state.user_role
                        doc.metadata["upload_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    all_docs.extend(docs)
                    progress_bar.progress((i + 1) / len(saved_files))
                
                # Initialize vector database manager
                vector_db_manager = VectorDatabaseManager()
                
                # Create vector store
                status_text.text("Creating vector store...")
                vectorstore = vector_db_manager.create_vectorstore(all_docs)
                
                # Save vector store
                vector_db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
                os.makedirs(vector_db_path, exist_ok=True)
                vector_db_manager.save_vectorstore(vector_db_path)
                
                # Update session state
                if st.session_state.vectorstore:
                    # Merge with existing vector store if it exists
                    st.session_state.vectorstore.merge_from(vectorstore)
                    st.session_state.all_documents.extend(all_docs)
                else:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.all_documents = all_docs
                
                st.session_state.processed_docs = True
                
                # Complete
                status_text.text("Processing complete!")
                progress_bar.progress(1.0)
                
                st.success(f"Successfully processed {len(all_docs)} document chunks from {len(uploaded_files)} files.")
            
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

# Document statistics
if st.session_state.processed_docs and st.session_state.all_documents:
    st.subheader("Document Statistics")
    
    # Count documents by type
    doc_types = {}
    for doc in st.session_state.all_documents:
        source = doc.metadata.get("source", "Unknown")
        file_ext = os.path.splitext(source)[1].lower()
        doc_types[file_ext] = doc_types.get(file_ext, 0) + 1
    
    # Display statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Documents", len(st.session_state.all_documents))
    with col2:
        st.metric("Document Types", len(doc_types))
    
    # Display document types
    st.markdown("#### Document Types")
    for ext, count in doc_types.items():
        st.write(f"- **{ext}**: {count} chunks")
    
    # Display recent documents
    st.markdown("#### Recent Documents")
    recent_docs = {}
    for doc in st.session_state.all_documents:
        source = doc.metadata.get("source", "Unknown")
        if source not in recent_docs:
            recent_docs[source] = {
                "classification": doc.metadata.get("classification", "Unknown"),
                "department": doc.metadata.get("department", "Unknown"),
                "upload_user": doc.metadata.get("upload_user", "Unknown"),
                "upload_timestamp": doc.metadata.get("upload_timestamp", "Unknown")
            }
    
    for source, metadata in list(recent_docs.items())[:5]:
        st.markdown(f"**{os.path.basename(source)}**")
        st.markdown(f"Classification: {metadata['classification']} | "
                   f"Department: {metadata['department']} | "
                   f"Uploaded by: {metadata['upload_user']} | "
                   f"Date: {metadata['upload_timestamp']}")
        st.markdown("---")
