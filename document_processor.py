import os
import logging
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Class to process documents from various sources"""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        metadata_extractor: Optional[callable] = None
    ):
        """
        Initialize the document processor
        
        Args:
            chunk_size: Size of each document chunk
            chunk_overlap: Overlap between chunks
            metadata_extractor: Optional function to extract additional metadata
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadata_extractor = metadata_extractor
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def get_file_loader(self, file_path: str) -> Any:
        """
        Get the appropriate document loader for a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document loader
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == ".pdf":
            try:
                # Try using PyPDFLoader first
                return PyPDFLoader(file_path)
            except Exception as e:
                # Log the error
                logger.warning(f"PyPDFLoader failed: {str(e)}. Trying alternate PDF loader...")
                # Import and try an alternative PDF loader
                from langchain_community.document_loaders import PyMuPDFLoader
                return PyMuPDFLoader(file_path)
        elif file_extension in [".docx", ".doc"]:
            return Docx2txtLoader(file_path)
        elif file_extension == ".csv":
            return CSVLoader(file_path)
        elif file_extension in [".xlsx", ".xls"]:
            return UnstructuredExcelLoader(file_path)
        elif file_extension in [".txt", ".md"]:
            return TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    
    def process_file(self, file_path: str) -> List[Document]:
        """
        Process a single file into document chunks
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of document chunks
        """
        try:
            logger.info(f"Processing file: {file_path}")
            try:
                loader = self.get_file_loader(file_path)
                documents = loader.load()
            except Exception as loader_error:
                logger.error(f"Primary loader failed: {str(loader_error)}")
                # As a fallback for any loader error, try with UnstructuredFileLoader
                logger.info("Attempting with UnstructuredFileLoader as fallback...")
                from langchain_community.document_loaders import UnstructuredFileLoader
                loader = UnstructuredFileLoader(file_path)
                documents = loader.load()
            
            # Check if documents were loaded successfully
            if not documents:
                logger.warning(f"No content extracted from {file_path}")
                # Create a simple document with filename as fallback
                documents = [Document(
                    page_content=f"File: {os.path.basename(file_path)}\nNo content could be extracted from this file.",
                    metadata={"source": file_path, "extraction_error": True}
                )]
            
            # Extract additional metadata if metadata extractor is provided
            if self.metadata_extractor:
                for doc in documents:
                    additional_metadata = self.metadata_extractor(file_path, doc)
                    doc.metadata.update(additional_metadata)
            
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Split {file_path} into {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            # Don't raise the exception, just return an empty list
            # This allows processing to continue with other files
            return [Document(
                page_content=f"Error processing file: {os.path.basename(file_path)}",
                metadata={"source": file_path, "error": str(e)}
            )]
    
    def process_directory(self, directory_path: str, recursive: bool = True) -> List[Document]:
        """
        Process all files in a directory
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to recursively process subdirectories
            
        Returns:
            List of document chunks
        """
        all_docs = []
        try:
            for root, _, files in os.walk(directory_path):
                if not recursive and root != directory_path:
                    continue
                    
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # Skip files with unsupported extensions
                        file_extension = os.path.splitext(file)[1].lower()
                        supported_extensions = [".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".xlsx", ".xls"]
                        if file_extension not in supported_extensions:
                            continue
                            
                        docs = self.process_file(file_path)
                        all_docs.extend(docs)
                    except Exception as e:
                        logger.warning(f"Skipping file {file_path} due to error: {str(e)}")
            
            logger.info(f"Processed {len(all_docs)} documents from directory {directory_path}")
            return all_docs
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            raise
    
    def process_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[Document]:
        """
        Process raw text strings into document chunks
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of document chunks
        """
        try:
            if metadatas is None:
                metadatas = [{} for _ in texts]
                
            documents = [Document(page_content=text, metadata=metadata) 
                        for text, metadata in zip(texts, metadatas)]
            
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(texts)} texts into {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            logger.error(f"Error processing texts: {str(e)}")
            raise
