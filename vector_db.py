import os
from typing import List, Dict, Any
import logging

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    """Class to manage vector database operations"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the vector database manager
        
        Args:
            embedding_model: The embedding model name to use
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Create a vector store from documents
        
        Args:
            documents: List of documents to add to vector store
            
        Returns:
            FAISS vector store
        """
        try:
            logger.info(f"Creating vector store with {len(documents)} documents")
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            return self.vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def save_vectorstore(self, path: str) -> None:
        """
        Save the vector store to disk
        
        Args:
            path: Path to save the vector store
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            logger.info(f"Saving vector store to {path}")
            self.vectorstore.save_local(path)
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_vectorstore(self, path: str) -> FAISS:
        """
        Load a vector store from disk
        
        Args:
            path: Path to load the vector store from
            
        Returns:
            FAISS vector store
        """
        try:
            logger.info(f"Loading vector store from {path}")
            self.vectorstore = FAISS.load_local(path, self.embeddings)
            return self.vectorstore
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform a similarity search on the vector store
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        try:
            logger.info(f"Performing similarity search for '{query}' with k={k}")
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    def get_retriever(self, search_kwargs: Dict[str, Any] = None) -> Any:
        """
        Get a retriever from the vector store
        
        Args:
            search_kwargs: Search kwargs to pass to the retriever
            
        Returns:
            Retriever
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        if search_kwargs is None:
            search_kwargs = {"k": 5}
            
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
