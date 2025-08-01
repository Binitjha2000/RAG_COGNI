import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from functools import lru_cache
from difflib import SequenceMatcher

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever
from langchain.schema import BaseRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchConfig:
    """Configuration for search parameters"""
    # Hybrid search parameters
    alpha: float = 0.3  # Favor keyword search for person matching
    keyword_weight: float = 0.6
    semantic_weight: float = 0.4
    
    # Search result limits
    max_initial_results: int = 15
    max_final_results: int = 5
    
    # Similarity thresholds (lowered for better recall)
    semantic_threshold: float = 0.3
    keyword_threshold: float = 0.1
    
    # Person-specific search (more lenient)
    person_name_threshold: float = 0.4
    enable_person_filtering: bool = True
    cv_document_boost: float = 5.0
    
    # Reranking parameters
    enable_reranking: bool = True
    rerank_top_k: int = 10

@dataclass
class SearchResult:
    """Enhanced search result with scoring details"""
    document: Document
    semantic_score: float
    keyword_score: float
    combined_score: float
    person_match_score: float
    document_type_score: float
    rank: int
    search_method: str

class PersonDetector:
    """Robust person name detection with strict filtering"""
    
    def __init__(self):
        self._organizational_terms = self._get_organizational_terms()
        self._location_terms = self._get_location_terms()
    
    def _get_organizational_terms(self) -> set:
        """Terms that indicate organizational entities, not people"""
        return {
            'district', 'education', 'office', 'officer', 'department', 'ministry',
            'company', 'limited', 'ltd', 'pvt', 'corporation', 'enterprises',
            'services', 'solutions', 'technologies', 'systems', 'group',
            'foundation', 'institute', 'university', 'college', 'school',
            'hospital', 'clinic', 'bank', 'financial', 'insurance'
        }
    
    def _get_location_terms(self) -> set:
        """Location names that are not person names"""
        return {
            'patna', 'bihar', 'jharkhand', 'mumbai', 'delhi', 'chennai', 'bangalore',
            'hyderabad', 'pune', 'kolkata', 'gurgaon', 'noida', 'gaya', 'muzaffarpur',
            'bengaluru', 'karnataka', 'telangana', 'maharashtra', 'tamil nadu',
            'west bengal', 'uttar pradesh', 'gujarat', 'rajasthan'
        }
    
    def extract_person_names(self, text: str) -> List[str]:
        """Extract person names with strict filtering"""
        if not text or len(text.strip()) < 3:
            return []
        
        # Check if text contains organizational indicators
        text_lower = text.lower()
        if any(term in text_lower for term in self._organizational_terms):
            logger.info(f"Skipping organizational text: {text[:50]}...")
            return []
        
        # Extract potential names using multiple patterns
        potential_names = []
        
        # Pattern 1: Capitalized words (2-4 words)
        words = text.split()
        for i in range(len(words) - 1):
            if i + 2 < len(words):
                three_word = f"{words[i]} {words[i+1]} {words[i+2]}"
                if self._is_valid_person_name(three_word):
                    potential_names.append(three_word)
            
            two_word = f"{words[i]} {words[i+1]}"
            if self._is_valid_person_name(two_word):
                potential_names.append(two_word)
        
        # Pattern 2: Direct extraction from queries
        name_patterns = [
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:skills?|experience|education|cv|resume)',
            r'(?:skills?|experience|education|cv|resume)\s+(?:of|for)\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'what\s+(?:are|is)\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if self._is_valid_person_name(match):
                    potential_names.append(match.title())
        
        # Clean and validate names
        valid_names = []
        for name in potential_names:
            cleaned_name = self._clean_name(name)
            if cleaned_name and self._is_valid_person_name(cleaned_name):
                valid_names.append(cleaned_name)
        
        # Remove duplicates and return
        unique_names = list(set(valid_names))
        logger.info(f"Extracted person names: {unique_names}")
        return unique_names
    
    def _clean_name(self, name: str) -> str:
        """Clean and normalize a name"""
        if not name:
            return ""
        
        # Remove extra spaces and special characters
        name = re.sub(r'\s+', ' ', name.strip())
        name = re.sub(r'[^\w\s]', '', name)
        
        # Title case
        return ' '.join(word.capitalize() for word in name.split() if word)
    
    def _is_valid_person_name(self, name: str) -> bool:
        """Validate if text is likely a person name"""
        if not name or len(name.strip()) < 3:
            return False
        
        words = name.lower().split()
        
        # Must have 2-3 words
        if len(words) < 2 or len(words) > 4:
            return False
        
        # Check against organizational terms
        if any(word in self._organizational_terms for word in words):
            return False
        
        # Check against location terms
        if any(word in self._location_terms for word in words):
            return False
        
        # Each word should be reasonable length and alphabetic
        for word in words:
            if len(word) < 2 or len(word) > 15 or not word.isalpha():
                return False
        
        # Should not contain common non-name words
        non_name_words = {
            'the', 'and', 'or', 'but', 'with', 'from', 'what', 'where', 'when',
            'skills', 'experience', 'education', 'tell', 'about', 'give'
        }
        if any(word in non_name_words for word in words):
            return False
        
        return True
    
    def calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names"""
        if not name1 or not name2:
            return 0.0
        
        # Normalize names
        norm_name1 = self._clean_name(name1).upper()
        norm_name2 = self._clean_name(name2).upper()
        
        # Exact match
        if norm_name1 == norm_name2:
            return 1.0
        
        # Word-level matching
        words1 = set(norm_name1.split())
        words2 = set(norm_name2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate overlap
        common_words = words1.intersection(words2)
        total_words = words1.union(words2)
        word_similarity = len(common_words) / len(total_words)
        
        # Sequence similarity
        sequence_similarity = SequenceMatcher(None, norm_name1, norm_name2).ratio()
        
        # Partial name matching (first name or last name match)
        partial_match = 0.0
        for word1 in words1:
            for word2 in words2:
                if word1 == word2 and len(word1) > 2:
                    partial_match = max(partial_match, 0.7)
        
        # Return best similarity
        final_score = max(word_similarity, sequence_similarity, partial_match)
        logger.debug(f"Name similarity '{name1}' vs '{name2}': {final_score:.3f}")
        return final_score

class DocumentClassifier:
    """Enhanced document classifier with strict CV detection"""
    
    def __init__(self):
        self._cv_patterns = self._compile_cv_patterns()
        self._organizational_patterns = self._compile_organizational_patterns()
    
    def _compile_cv_patterns(self) -> List[re.Pattern]:
        """Strong CV/resume indicators"""
        patterns = [
            r'\b(?:curriculum\s+vitae|resume|cv)\b',
            r'\b(?:work\s+experience|employment\s+history|professional\s+experience)\b',
            r'\b(?:education|educational\s+background|academic\s+background)\b',
            r'\b(?:skills|technical\s+skills|core\s+competencies|expertise)\b',
            r'\b(?:projects|key\s+projects|major\s+projects)\b',
            r'\b(?:certifications|professional\s+certifications|achievements)\b',
            r'\b(?:bachelor|master|degree|university|college|institute)\b',
            r'\b(?:manager|engineer|developer|analyst|consultant|specialist)\b',
            r'\b(?:programming|software|technology|technical)\b'
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _compile_organizational_patterns(self) -> List[re.Pattern]:
        """Strong organizational document indicators"""
        patterns = [
            r'\bEST\s+ID\b|\bEST\s+NAME\b|\bZONE\s+NAME\b',
            r'\bDISTRICT\s+EDUCATION\s+OFFICE\b',
            r'\bOFFICE\s+NAME\b|\bESTABLISHMENT\b',
            r'\bSR\s+OFFICE\s+NAME\b',
            r'\b(?:establishments|companies|organizations)\s+in\s+terms\s+of\b'
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def classify_document(self, document: Document) -> Dict[str, Any]:
        """Classify document with strict rules"""
        content = document.page_content
        filename = document.metadata.get('source', '').lower()
        
        classification = {
            'type': 'unknown',
            'confidence': 0.0,
            'is_cv': False,
            'is_organizational': False,
            'cv_score': 0.0,
            'organizational_score': 0.0,
            'person_names': []
        }
        
        # Check for organizational patterns first (higher priority)
        org_matches = 0
        for pattern in self._organizational_patterns:
            org_matches += len(pattern.findall(content))
        
        # Strong organizational indicators in filename
        org_filename_indicators = ['establishment', 'office', 'district', 'company', 'organization']
        if any(indicator in filename for indicator in org_filename_indicators):
            org_matches += 5
        
        org_score = min(org_matches / 2.0, 1.0)
        
        # Check for CV patterns
        cv_matches = 0
        for pattern in self._cv_patterns:
            cv_matches += len(pattern.findall(content))
        
        # Strong CV indicators in filename
        cv_filename_indicators = ['cv', 'resume', 'curriculum']
        if any(indicator in filename for indicator in cv_filename_indicators):
            cv_matches += 5
        
        cv_score = min(cv_matches / 3.0, 1.0)
        
        # Determine document type (organizational takes precedence)
        if org_score > 0.4:
            classification['type'] = 'organizational'
            classification['is_organizational'] = True
            classification['confidence'] = org_score
        elif cv_score > 0.3:
            classification['type'] = 'cv_resume'
            classification['is_cv'] = True
            classification['confidence'] = cv_score
            
            # Extract person names only for CV documents
            person_detector = PersonDetector()
            content_names = person_detector.extract_person_names(content)
            filename_names = person_detector.extract_person_names(
                os.path.basename(filename).replace('.pdf', '').replace('_', ' ').replace('-', ' ')
            )
            
            all_names = content_names + filename_names
            classification['person_names'] = list(set(all_names))
        
        classification['cv_score'] = cv_score
        classification['organizational_score'] = org_score
        
        logger.info(f"Document {os.path.basename(filename)}: "
                   f"type={classification['type']}, "
                   f"cv_score={cv_score:.2f}, "
                   f"org_score={org_score:.2f}, "
                   f"names={classification['person_names']}")
        
        return classification

class QueryProcessor:
    """Enhanced query processing with robust person detection"""
    
    def __init__(self):
        self._stop_words = self._get_stop_words()
        self.person_detector = PersonDetector()
    
    def _get_stop_words(self) -> set:
        """Comprehensive stop words"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'under', 'over', 'is', 'are', 'was', 'were',
            'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'shall', 'what', 'who', 'where', 'when',
            'why', 'how', 'which', 'tell', 'me', 'please'
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Robust query analysis with person detection"""
        analysis = {
            'query_type': 'general',
            'is_person_query': False,
            'target_person': None,
            'person_names': [],
            'keywords': self._extract_keywords(query),
            'suggested_alpha': 0.5
        }
        
        # Extract person names from query
        person_names = self.person_detector.extract_person_names(query)
        analysis['person_names'] = person_names
        
        # Multiple approaches to extract target person
        target_person = None
        
        # Approach 1: Regex patterns for person queries
        person_patterns = [
            r'what\s+(?:are|is)\s+([A-Za-z]+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:skills?|experience|education)',
            r'([A-Za-z]+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:skills?|experience|education)',
            r'(?:skills?|experience|education)\s+(?:of|for)\s+([A-Za-z]+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*)',
            r'tell\s+me\s+about\s+([A-Za-z]+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*)',
            r'\b([A-Za-z]+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*)\s+experience'
        ]
        
        for pattern in person_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                potential_name = match.group(1).strip()
                if self.person_detector._is_valid_person_name(potential_name):
                    target_person = self.person_detector._clean_name(potential_name)
                    break
        
        # Approach 2: Use detected names if pattern didn't work
        if not target_person and person_names:
            target_person = person_names[0]
        
        # Set person query flags
        if target_person:
            analysis['is_person_query'] = True
            analysis['query_type'] = 'person_specific'
            analysis['target_person'] = target_person
            analysis['suggested_alpha'] = 0.2  # Heavily favor keyword search
        
        logger.info(f"Query analysis: type={analysis['query_type']}, "
                   f"person={analysis['target_person']}, "
                   f"names_detected={analysis['person_names']}")
        
        return analysis
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        words = re.findall(r'\b\w+\b', query.lower())
        return [word for word in words if word not in self._stop_words and len(word) > 2]

class EnhancedVectorDatabaseManager:
    """RAG system with strict person-document matching"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-mpnet-base-v2", config: Optional[SearchConfig] = None):
        """Initialize with enhanced search capabilities"""
        self.config = config or SearchConfig()
        self.query_processor = QueryProcessor()
        self.person_detector = PersonDetector()
        self.document_classifier = DocumentClassifier()
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Storage
        self.vectorstore = None
        self.bm25_retriever = None
        self.documents = []
        self.document_classifications = {}
        

        self.search_stats = {
            'total_searches': 0,
            'person_searches': 0,
            'person_specific': 0,  # Added this
            'organizational_searches': 0,  # Added this
            'organizational_specific': 0,  # Added this
            'semantic_searches': 0,
            'keyword_searches': 0,
            'hybrid_searches': 0,
            'general_searches': 0,  # Added this
            'avg_response_time': 0.0,
            'successful_person_matches': 0,
            'failed_person_matches': 0
    }
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """Create vector store with enhanced document classification"""
        try:
            logger.info(f"Creating vector store with {len(documents)} documents")
            
            self.documents = documents
            
            # Classify all documents
            cv_count = 0
            org_count = 0
            
            for doc in documents:
                classification = self.document_classifier.classify_document(doc)
                self.document_classifications[id(doc)] = classification
                
                # Add metadata
                doc.metadata.update({
                    'document_type': classification['type'],
                    'is_cv': classification['is_cv'],
                    'is_organizational': classification['is_organizational'],
                    'cv_score': classification['cv_score'],
                    'organizational_score': classification['organizational_score'],
                    'person_names': classification['person_names']
                })
                
                if classification['is_cv']:
                    cv_count += 1
                if classification['is_organizational']:
                    org_count += 1
            
            logger.info(f"Document classification complete: {cv_count} CVs, {org_count} organizational docs")
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Create BM25 retriever
            self._create_bm25_retriever(documents)
            
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def _create_bm25_retriever(self, documents: List[Document]):
        """Create BM25 retriever for keyword search"""
        try:
            texts = [doc.page_content for doc in documents]
            self.bm25_retriever = BM25Retriever.from_texts(
                texts,
                metadatas=[doc.metadata for doc in documents]
            )
            self.bm25_retriever.k = self.config.max_initial_results
        except Exception as e:
            logger.warning(f"Failed to create BM25 retriever: {str(e)}")
            self.bm25_retriever = None
    
    def person_specific_search(self, query: str, target_person: str, k: int = None) -> List[SearchResult]:
        """Enhanced person-specific search with strict filtering"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        k = k or self.config.max_final_results
        
        logger.info(f"Person-specific search for: '{target_person}'")
        
        # Step 1: Find CV documents that match the person
        matching_cv_docs = []
        
        for doc in self.documents:
            classification = self.document_classifications.get(id(doc), {})
            
            # Must be a CV document
            if not classification.get('is_cv', False):
                continue
            
            # Calculate person match score
            person_match_score = 0.0
            
            # Check against extracted person names
            for doc_person_name in classification.get('person_names', []):
                similarity = self.person_detector.calculate_name_similarity(target_person, doc_person_name)
                person_match_score = max(person_match_score, similarity)
            
            # Check content directly
            content_upper = doc.page_content.upper()
            target_upper = target_person.upper()
            
            if target_upper in content_upper:
                person_match_score = max(person_match_score, 1.0)
            else:
                # Check individual words
                target_words = target_upper.split()
                matched_words = sum(1 for word in target_words if word in content_upper)
                if matched_words >= len(target_words) * 0.5:  # At least 50% of name words
                    person_match_score = max(person_match_score, 0.8)
            
            # Check filename
            filename = doc.metadata.get('source', '')
            if any(word.lower() in filename.lower() for word in target_person.split()):
                person_match_score = max(person_match_score, 0.9)
            
            logger.debug(f"Doc {os.path.basename(filename)}: person_score={person_match_score:.3f}")
            
            # Use lower threshold for better recall
            if person_match_score >= self.config.person_name_threshold:
                matching_cv_docs.append((doc, person_match_score))
        
        logger.info(f"Found {len(matching_cv_docs)} matching CV documents")
        
        if not matching_cv_docs:
            logger.warning(f"No CV documents found for '{target_person}'")
            self.search_stats['failed_person_matches'] += 1
            return []
        
        # Step 2: Search within matching documents
        self.search_stats['successful_person_matches'] += 1
        matching_cv_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Create temporary vector store with only matching docs
        person_docs = [doc for doc, _ in matching_cv_docs]
        temp_vectorstore = FAISS.from_documents(person_docs, self.embeddings)
        
        # Perform search
        docs_and_scores = temp_vectorstore.similarity_search_with_score(query, k=k*2)
        
        results = []
        for rank, (doc, distance) in enumerate(docs_and_scores[:k]):
            similarity_score = 1.0 / (1.0 + distance)
            person_score = next((score for d, score in matching_cv_docs if id(d) == id(doc)), 0.0)
            
            classification = self.document_classifications.get(id(doc), {})
            doc_type_score = classification.get('cv_score', 0.0)
            
            combined_score = similarity_score + (person_score * self.config.cv_document_boost) + doc_type_score
            
            result = SearchResult(
                document=doc,
                semantic_score=similarity_score,
                keyword_score=0.0,
                combined_score=combined_score,
                person_match_score=person_score,
                document_type_score=doc_type_score,
                rank=rank + 1,
                search_method="person_specific"
            )
            results.append(result)
        
        logger.info(f"Person-specific search returned {len(results)} results")
        return results
    
    def adaptive_search(self, query: str, k: int = None) -> List[SearchResult]:
        """Main search entry point with person query handling"""
        k = k or self.config.max_final_results
        
        # Analyze query
        analysis = self.query_processor.analyze_query(query)
        self.search_stats['total_searches'] += 1
        
        # Handle person-specific queries
        if analysis['is_person_query'] and analysis['target_person']:
            self.search_stats['person_searches'] += 1
            logger.info(f"Person query detected: '{analysis['target_person']}'")
            
            results = self.person_specific_search(query, analysis['target_person'], k)
            
            if results:
                return results
            else:
                logger.warning(f"No results for person '{analysis['target_person']}', trying fallback")
                return self._fallback_search(query, k)
        
        # General search for non-person queries
        return self._general_search(query, k)
    
    def _fallback_search(self, query: str, k: int) -> List[SearchResult]:
        """Fallback search when person-specific search fails"""
        logger.info("Executing fallback search")
        
        # Try to find any CV documents
        cv_docs = [doc for doc in self.documents 
                  if self.document_classifications.get(id(doc), {}).get('is_cv', False)]
        
        if cv_docs:
            logger.info(f"Searching within {len(cv_docs)} CV documents")
            temp_vectorstore = FAISS.from_documents(cv_docs, self.embeddings)
            docs_and_scores = temp_vectorstore.similarity_search_with_score(query, k=k)
            
            results = []
            for rank, (doc, distance) in enumerate(docs_and_scores):
                similarity_score = 1.0 / (1.0 + distance)
                
                result = SearchResult(
                    document=doc,
                    semantic_score=similarity_score,
                    keyword_score=0.0,
                    combined_score=similarity_score,
                    person_match_score=0.0,
                    document_type_score=0.5,
                    rank=rank + 1,
                    search_method="fallback_cv"
                )
                results.append(result)
            
            return results
        
        # Last resort: general search
        return self._general_search(query, k)
    
    def _general_search(self, query: str, k: int) -> List[SearchResult]:
        """General search across all documents"""
        if not self.vectorstore:
            return []
        
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        results = []
        for rank, (doc, distance) in enumerate(docs_and_scores):
            similarity_score = 1.0 / (1.0 + distance)
            
            result = SearchResult(
                document=doc,
                semantic_score=similarity_score,
                keyword_score=0.0,
                combined_score=similarity_score,
                person_match_score=0.0,
                document_type_score=0.0,
                rank=rank + 1,
                search_method="general"
            )
            results.append(result)
        
        return results
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return self.search_stats.copy()
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get document classification statistics"""
        total_docs = len(self.document_classifications)
        cv_docs = len([c for c in self.document_classifications.values() if c['is_cv']])
        org_docs = len([c for c in self.document_classifications.values() if c['is_organizational']])
        
        return {
            'total_documents': total_docs,
            'cv_documents': cv_docs,
            'organizational_documents': org_docs,
            'other_documents': total_docs - cv_docs - org_docs
        }
    
    # Legacy compatibility methods
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Legacy compatibility method"""
        results = self.adaptive_search(query, k)
        return [result.document for result in results]
    
    def get_retriever(self, search_kwargs: Dict[str, Any] = None) -> BaseRetriever:
        """Get retriever for legacy compatibility"""
        search_kwargs = search_kwargs or {"k": self.config.max_final_results}
        
        class EnhancedRetriever(BaseRetriever):
            def __init__(self, vector_db_manager):
                super().__init__()
                self.vector_db_manager = vector_db_manager
            
            def get_relevant_documents(self, query: str) -> List[Document]:
                results = self.vector_db_manager.adaptive_search(query)
                return [result.document for result in results]
        
        return EnhancedRetriever(self)
    
    def save_vectorstore(self, path: str) -> None:
        """Save vector store to disk"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.vectorstore.save_local(path)
            
            # Save classifications
            import pickle
            with open(os.path.join(path, 'document_classifications.pkl'), 'wb') as f:
                pickle.dump(self.document_classifications, f)
                
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_vectorstore(self, path: str) -> FAISS:
        """Load vector store from disk"""
        try:
            self.vectorstore = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load classifications
            import pickle
            classifications_path = os.path.join(path, 'document_classifications.pkl')
            if os.path.exists(classifications_path):
                with open(classifications_path, 'rb') as f:
                    self.document_classifications = pickle.load(f)
            
            return self.vectorstore
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

# Alias for backward compatibility
VectorDatabaseManager = EnhancedVectorDatabaseManager
