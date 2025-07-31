import re
from typing import List, Dict, Any, Optional

class PrivacyFilter:
    """Class to filter out sensitive information from document content"""
    
    def __init__(self):
        """Initialize the privacy filter with regex patterns for sensitive data"""
        # Regular expressions for different types of sensitive information
        self.patterns = {
            # Passwords
            'password': [
                r'password[s]?\s*[=:]\s*[\'"]?([^\s\'"]+)[\'"]?',
                r'pwd[s]?\s*[=:]\s*[\'"]?([^\s\'"]+)[\'"]?',
                r'passcode[s]?\s*[=:]\s*[\'"]?([^\s\'"]+)[\'"]?',
            ],
            
            # API Keys, Secret Keys, Access Tokens
            'api_key': [
                r'api[_\-\s]?key[s]?\s*[=:]\s*[\'"]?([^\s\'"]+)[\'"]?',
                r'secret[_\-\s]?key[s]?\s*[=:]\s*[\'"]?([^\s\'"]+)[\'"]?',
                r'access[_\-\s]?token[s]?\s*[=:]\s*[\'"]?([^\s\'"]+)[\'"]?',
                r'auth[_\-\s]?token[s]?\s*[=:]\s*[\'"]?([^\s\'"]+)[\'"]?',
                r'bearer[_\-\s]?token[s]?\s*[=:]\s*[\'"]?([^\s\'"]+)[\'"]?',
            ],
            
            # Credit Card Numbers (basic pattern)
            'credit_card': [
                r'\b(?:\d{4}[- ]?){3}\d{4}\b',
                r'\b\d{16}\b',
            ],
            
            # Social Security Numbers
            'ssn': [
                r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            ],
            
            # Email addresses
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            ],
            
            # Phone numbers
            'phone': [
                r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            ],
            
            # IP addresses
            'ip_address': [
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            ],
            
            # Database connection strings
            'connection_string': [
                r'(?:mongodb|mysql|postgresql|jdbc|redis|ldap):\/\/[^\s"\']+',
                r'(?:Server|Data Source|Host)=[^;]+;(?:Database|Initial Catalog)=[^;]+;(?:User Id|UID)=[^;]+;(?:Password|PWD)=[^;]+'
            ]
        }
        
        # Compile all patterns for efficiency
        self.compiled_patterns = {}
        for category, patterns in self.patterns.items():
            self.compiled_patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def filter_text(self, text: str) -> str:
        """
        Filter sensitive information from text
        
        Args:
            text: The input text to filter
            
        Returns:
            Text with sensitive information replaced by placeholders
        """
        if not text:
            return text
        
        filtered_text = text
        
        # Apply all regex patterns and replace matches
        for category, compiled_patterns in self.compiled_patterns.items():
            for pattern in compiled_patterns:
                if category in ['email', 'connection_string', 'api_key', 'password']:
                    # For these categories, we replace the entire match
                    filtered_text = pattern.sub(f"[REDACTED {category.upper()}]", filtered_text)
                else:
                    # For other categories, replace with a more specific marker
                    filtered_text = pattern.sub(f"[REDACTED {category.upper()}]", filtered_text)
        
        return filtered_text
    
    def filter_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter sensitive information from a document
        
        Args:
            document: Document dictionary with 'page_content' and 'metadata'
            
        Returns:
            Document with filtered content
        """
        if not document:
            return document
        
        filtered_doc = document.copy()
        if 'page_content' in filtered_doc:
            filtered_doc['page_content'] = self.filter_text(filtered_doc['page_content'])
        
        return filtered_doc
    
    def filter_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter sensitive information from a list of documents
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of documents with filtered content
        """
        return [self.filter_document(doc) for doc in documents]
