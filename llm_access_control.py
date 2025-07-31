import logging
from typing import Dict, Any, List, Optional
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccessControlledLLM:
    """Class to manage LLM access with PIA (Personally Identifiable Access) controls"""
    
    def __init__(self, temperature: float = 0.7):
        """
        Initialize the LLM with access control
        
        Args:
            temperature: Temperature for the LLM
        """
        self.temperature = temperature
        self.model_manager = ModelManager()
        self.llm, self.model_info = self.model_manager.get_model(temperature=temperature)
        logger.info(f"Using model: {self.model_info['name']} ({self.model_info['type']})")
        
    def create_qa_chain(self, retriever: Any, chain_type: str = "stuff") -> RetrievalQA:
        """
        Create a QA chain with the LLM
        
        Args:
            retriever: Document retriever
            chain_type: Type of chain to create
            
        Returns:
            QA chain
        """
        try:
            logger.info(f"Creating {chain_type} QA chain with {self.model_info['name']}")
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type=chain_type,
                retriever=retriever,
                return_source_documents=True
            )
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model
        
        Returns:
            Dictionary with model information
        """
        return self.model_info
        
    def query_with_access_control(
        self, 
        qa_chain: RetrievalQA, 
        query: str, 
        user_role: str,
        restricted_keywords: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Query the LLM with access control
        
        Args:
            qa_chain: QA chain to use
            query: Query string
            user_role: Role of the user making the query
            restricted_keywords: Dictionary mapping roles to restricted keywords
            
        Returns:
            Query result
        """
        # Default restricted keywords if none provided
        if restricted_keywords is None:
            restricted_keywords = {
                "Viewer": ["confidential", "restricted", "private", "sensitive", "classified"],
                "Analyst": ["top secret", "classified"],
                "Admin": []  # Admin has full access
            }
        
        # Check if the query contains restricted keywords for the user's role
        if user_role in restricted_keywords:
            for keyword in restricted_keywords[user_role]:
                if keyword.lower() in query.lower():
                    logger.warning(f"Access denied for {user_role} attempting to query restricted content: {keyword}")
                    return {
                        "result": f"Access denied: You do not have permission to query information related to '{keyword}'",
                        "source_documents": []
                    }
        
        try:
            logger.info(f"Executing query as {user_role}: {query}")
            result = qa_chain({"query": query})
            
            # Add model info to result
            result["model_info"] = self.model_info
            
            # Apply post-processing filters based on user role if needed
            # For example, you could filter out sensitive information from results
            
            return result
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
