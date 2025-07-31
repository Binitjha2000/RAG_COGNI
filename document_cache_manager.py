import os
import pickle
import hashlib
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional, Set
import logging
from langchain_core.documents import Document  # Updated from langchain.docstore.document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentCacheManager:
    """Class to handle document caching and vectorstore management"""
    
    def __init__(self, documents_folder: str = "documents", cache_folder: str = "cache"):
        """
        Initialize the document cache manager
        
        Args:
            documents_folder: Path to folder containing documents
            cache_folder: Path to folder for caching vector stores
        """
        self.documents_folder = os.path.join(os.getcwd(), documents_folder)
        self.cache_folder = os.path.join(os.getcwd(), cache_folder)
        self.embeddings = None
        self.vectorstore = None
        self.documents = []
        self.file_hashes = {}
        self.last_processed_files = set()
        
        # Create folders if they don't exist
        os.makedirs(self.documents_folder, exist_ok=True)
        os.makedirs(self.cache_folder, exist_ok=True)
        
        # Initialize embeddings model
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize the embeddings model"""
        try:
            device = self._get_device()
            logger.info(f"Attempting to initialize embeddings model on device: {device}")
            
            # Try using a more robust initialization approach to handle meta tensors
            try:
                # Using a more powerful embedding model for better retrieval with safer initialization
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    model_kwargs={'device': 'cpu'}  # Initialize on CPU first to avoid meta tensor issues
                )
                logger.info("Successfully initialized MPNet embeddings model on CPU")
            except Exception as e:
                logger.warning(f"Failed to initialize MPNet model: {str(e)}")
                # Fallback to smaller model with safer initialization
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}  # Always start on CPU
                )
                logger.info("Successfully initialized MiniLM embeddings model on CPU")
                
        except Exception as e:
            logger.error(f"Critical error initializing any embeddings model: {str(e)}")
            # Create a minimal fallback embedding that won't fail
            from langchain_community.embeddings import FakeEmbeddings
            self.embeddings = FakeEmbeddings(size=384)  # Standard embedding size
            logger.warning("Using FakeEmbeddings as emergency fallback due to initialization errors")
    
    def _get_device(self) -> str:
        """Determine which device to use based on settings and availability"""
        import os
        
        force_cpu = os.getenv("FORCE_CPU", "false").lower() == "true"
        force_gpu = os.getenv("FORCE_GPU", "false").lower() == "true"
        
        # Default to CPU for safety
        device = "cpu"
        
        # Check if GPU is available only if not forcing CPU
        if not force_cpu:
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                
                # Additional check to avoid meta tensor issues
                if gpu_available:
                    try:
                        # Try to create a small tensor on GPU to verify it works
                        test_tensor = torch.zeros(1, device="cuda")
                        _ = test_tensor + 1  # Try a simple operation
                        
                        # If we get here, GPU is truly available
                        if force_gpu or not force_cpu:
                            device = "cuda"
                            logger.info("GPU verification successful, will use CUDA")
                    except Exception as gpu_test_error:
                        logger.warning(f"GPU detected but test failed: {str(gpu_test_error)}")
                        device = "cpu"
                        logger.info("Falling back to CPU despite GPU being detected")
            except Exception as e:
                logger.warning(f"Error checking GPU availability: {str(e)}")
                device = "cpu"
        
        logger.info(f"Selected device for embeddings: {device}")
        return device
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute hash of file to detect changes"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        
        # Include file modification time in hash
        mtime = os.path.getmtime(file_path)
        hasher.update(str(mtime).encode())
        
        return hasher.hexdigest()
    
    def _get_cache_path(self) -> str:
        """Get path to cache file"""
        return os.path.join(self.cache_folder, "vectorstore.pickle")
    
    def _get_metadata_path(self) -> str:
        """Get path to metadata file"""
        return os.path.join(self.cache_folder, "metadata.pickle")
    
    def _save_cache(self):
        """Save vectorstore and metadata to cache"""
        try:
            # Save vector store
            with open(self._get_cache_path(), 'wb') as f:
                pickle.dump(self.vectorstore, f)
            
            # Save metadata (file hashes, etc.)
            metadata = {
                'file_hashes': self.file_hashes,
                'last_processed_files': self.last_processed_files,
                'timestamp': datetime.now().isoformat()
            }
            with open(self._get_metadata_path(), 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info(f"Cache saved successfully with {len(self.last_processed_files)} documents")
            return True
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
            return False
    
    def _load_cache(self) -> bool:
        """Load vectorstore and metadata from cache if available"""
        try:
            # Check if cache files exist
            if not os.path.exists(self._get_cache_path()) or not os.path.exists(self._get_metadata_path()):
                logger.info("No cache files found")
                return False
            
            # Load vector store
            with open(self._get_cache_path(), 'rb') as f:
                self.vectorstore = pickle.load(f)
            
            # Load metadata
            with open(self._get_metadata_path(), 'rb') as f:
                metadata = pickle.load(f)
                self.file_hashes = metadata.get('file_hashes', {})
                self.last_processed_files = metadata.get('last_processed_files', set())
            
            logger.info(f"Cache loaded successfully with {len(self.last_processed_files)} documents")
            return True
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            # Reset cache state
            self.vectorstore = None
            self.file_hashes = {}
            self.last_processed_files = set()
            return False
    
    def scan_for_changes(self) -> Tuple[Set[str], Set[str]]:
        """
        Scan documents folder for new, modified, or deleted files
        
        Returns:
            Tuple of (modified_files, deleted_files)
        """
        current_files = set()
        modified_files = set()
        
        for root, _, files in os.walk(self.documents_folder):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file_path)[1].lower()
                
                # Skip non-document files
                if file_extension not in [".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".xlsx", ".xls"]:
                    continue
                
                current_files.add(file_path)
                
                # Check if file is new or modified
                current_hash = self._compute_file_hash(file_path)
                if file_path not in self.file_hashes or self.file_hashes[file_path] != current_hash:
                    modified_files.add(file_path)
                    self.file_hashes[file_path] = current_hash
        
        # Check for deleted files
        deleted_files = self.last_processed_files - current_files
        
        return modified_files, deleted_files
    
    def load_or_process_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[Optional[FAISS], List[Document]]:
        """
        Load vectorstore from cache or process documents if needed
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Tuple of (vectorstore, documents)
        """
        # Try loading from cache first
        cache_loaded = self._load_cache()
        
        if cache_loaded:
            # Check for new or modified files
            modified_files, deleted_files = self.scan_for_changes()
            
            # If no changes, return the cached vectorstore
            if not modified_files and not deleted_files:
                logger.info("No document changes detected, using cached vectorstore")
                return self.vectorstore, self.documents
            
            # Log changes
            if modified_files:
                logger.info(f"Found {len(modified_files)} new or modified files")
            if deleted_files:
                logger.info(f"Found {len(deleted_files)} deleted files")
                
            # If there are deleted files, we need to reprocess everything
            # This is because FAISS doesn't support removing specific documents
            if deleted_files:
                logger.info("Deleted files detected, reprocessing all documents")
                self.last_processed_files = self.last_processed_files - deleted_files
                return self._process_all_documents(chunk_size, chunk_overlap)
            
            # Process only the modified files and update the vectorstore
            if modified_files:
                # Import DocumentProcessor to process the files
                from document_processor import DocumentProcessor
                processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                
                new_docs = []
                for file_path in modified_files:
                    try:
                        docs = processor.process_file(file_path)
                        new_docs.extend(docs)
                        # Update last processed files
                        self.last_processed_files.add(file_path)
                    except Exception as e:
                        logger.warning(f"Error processing {file_path}: {str(e)}")
                
                if new_docs:
                    # Add new documents to existing vectorstore
                    if self.vectorstore:
                        self.vectorstore.add_documents(new_docs)
                        self.documents.extend(new_docs)
                        # Save updated cache
                        self._save_cache()
                        logger.info(f"Added {len(new_docs)} new document chunks to vectorstore")
                
                return self.vectorstore, self.documents
        
        # If no cache or cache loading failed, process all documents
        return self._process_all_documents(chunk_size, chunk_overlap)
    
    def _process_all_documents(self, chunk_size: int, chunk_overlap: int) -> Tuple[Optional[FAISS], List[Document]]:
        """Process all documents in the folder from scratch"""
        try:
            # Import DocumentProcessor to process the files
            from document_processor import DocumentProcessor
            processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # First check if there are any valid documents
            any_valid_docs = False
            for root, _, files in os.walk(self.documents_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_extension = os.path.splitext(file_path)[1].lower()
                    supported_extensions = [".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".xlsx", ".xls"]
                    if file_extension in supported_extensions:
                        any_valid_docs = True
                        break
                if any_valid_docs:
                    break
                    
            if not any_valid_docs:
                logger.warning("No valid document files found in the documents folder")
                # Create a dummy document to prevent errors
                dummy_doc = Document(
                    page_content="This is a placeholder document. Please upload real documents to enable search.",
                    metadata={"source": "placeholder", "is_placeholder": True}
                )
                self.documents = [dummy_doc]
                self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
                return self.vectorstore, self.documents
            
            # Process all files in the documents folder
            all_docs = processor.process_directory(self.documents_folder)
            
            # Update processed files
            self.last_processed_files = set()
            for root, _, files in os.walk(self.documents_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_extension = os.path.splitext(file_path)[1].lower()
                    supported_extensions = [".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".xlsx", ".xls"]
                    if file_extension in supported_extensions:
                        self.last_processed_files.add(file_path)
                        self.file_hashes[file_path] = self._compute_file_hash(file_path)
            
            if not all_docs:
                logger.warning("No content could be extracted from any documents")
                # Create a dummy document to prevent errors
                dummy_doc = Document(
                    page_content="No content could be extracted from your documents. Please check file formats and try again.",
                    metadata={"source": "extraction_failed", "is_error": True}
                )
                self.documents = [dummy_doc]
                self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
                return self.vectorstore, self.documents
            
            # Create vector store with GPU support if available
            try:
                # Import FAISS utilities for GPU support
                from faiss_gpu_utils import check_faiss_gpu
                
                has_gpu, message = check_faiss_gpu()
                if has_gpu:
                    st.success("🚀 GPU acceleration enabled for vector search")
                else:
                    st.info(f"Using CPU for vector search: {message}")
                
                # Create vectorstore with additional error handling for meta tensor issues
                try:
                    self.vectorstore = FAISS.from_documents(all_docs, self.embeddings)
                except NotImplementedError as meta_error:
                    if "Cannot copy out of meta tensor" in str(meta_error):
                        # Switch to CPU for embeddings and try again
                        logger.warning("Meta tensor error detected, switching to CPU embeddings")
                        # Reinitialize embeddings on CPU
                        self.embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'}
                        )
                        # Try again with CPU embeddings
                        self.vectorstore = FAISS.from_documents(all_docs, self.embeddings)
                    else:
                        # Re-raise if it's a different NotImplementedError
                        raise
                
                self.documents = all_docs
                
                # Save cache
                self._save_cache()
                
            except Exception as e:
                logger.warning(f"Using CPU for vector search: {str(e)}")
                self.vectorstore = FAISS.from_documents(all_docs, self.embeddings)
                self.documents = all_docs
                
                # Save cache
                self._save_cache()
            
            return self.vectorstore, self.documents
        
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return None, []
    
    def process_uploaded_files(self, uploaded_files, chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[Optional[FAISS], List[Document]]:
        """Process uploaded files and add them to the vectorstore"""
        try:
            # Import DocumentProcessor to process the files
            from document_processor import DocumentProcessor
            processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            all_new_docs = []
            
            for uploaded_file in uploaded_files:
                # Save uploaded file to documents directory
                file_path = os.path.join(self.documents_folder, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Process file
                    docs = processor.process_file(file_path)
                    all_new_docs.extend(docs)
                    
                    # Update file hash and processed files
                    self.file_hashes[file_path] = self._compute_file_hash(file_path)
                    self.last_processed_files.add(file_path)
                    
                except Exception as e:
                    logger.warning(f"Error processing {uploaded_file.name}: {str(e)}")
            
            if not all_new_docs:
                return self.vectorstore, self.documents
            
            # Add to existing vectorstore or create new one
            if self.vectorstore:
                try:
                    self.vectorstore.add_documents(all_new_docs)
                except NotImplementedError as meta_error:
                    if "Cannot copy out of meta tensor" in str(meta_error):
                        # Switch to CPU for embeddings and recreate the vectorstore
                        logger.warning("Meta tensor error detected while adding documents, recreating vectorstore with CPU embeddings")
                        # Reinitialize embeddings on CPU
                        self.embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'}
                        )
                        # Create new vectorstore with all documents
                        combined_docs = self.documents + all_new_docs
                        self.vectorstore = FAISS.from_documents(combined_docs, self.embeddings)
                    else:
                        # Re-raise if it's a different NotImplementedError
                        raise
                
                self.documents.extend(all_new_docs)
            else:
                try:
                    self.vectorstore = FAISS.from_documents(all_new_docs, self.embeddings)
                except NotImplementedError as meta_error:
                    if "Cannot copy out of meta tensor" in str(meta_error):
                        # Switch to CPU for embeddings and try again
                        logger.warning("Meta tensor error detected, switching to CPU embeddings")
                        # Reinitialize embeddings on CPU
                        self.embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'}
                        )
                        # Try again with CPU embeddings
                        self.vectorstore = FAISS.from_documents(all_new_docs, self.embeddings)
                    else:
                        # Re-raise if it's a different NotImplementedError
                        raise
                
                self.documents = all_new_docs
            
            # Save updated cache
            self._save_cache()
            
            return self.vectorstore, self.documents
            
        except Exception as e:
            logger.error(f"Error processing uploaded files: {str(e)}")
            return self.vectorstore, self.documents
