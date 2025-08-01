import os
import logging
from typing import Dict, Any, Tuple, List, Optional, Set, Union
from functools import lru_cache
from dataclasses import dataclass
import re
from config import config

# Configure logging using centralized config
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Log configuration on startup
config.log_config(logger)

# ===== CONSTANTS AND CONFIGURATION =====
@dataclass
class LLMConstants:
    """Constants for LLM processing"""
    MIN_CONTEXT_LENGTH: int = 10
    MIN_MEANINGFUL_CONTENT_LENGTH: int = 5
    MIN_CHUNK_LENGTH: int = 50
    MIN_LINE_LENGTH: int = 15
    MAX_RELEVANT_SENTENCES: int = 10
    MAX_SKILLS_PER_CATEGORY: int = 8
    MIN_RESPONSE_LENGTH: int = 100
    MAX_SENTENCES_FOR_SHORT_RESPONSE: int = 6
    MAX_SENTENCES_FOR_LONG_RESPONSE: int = 12
    MIN_COMMA_COUNT_FOR_LIST: int = 3
    TECHNICAL_BOOST_SCORE: int = 3
    EXACT_MATCH_BOOST: int = 2
    PARTIAL_MATCH_BOOST: int = 1

@dataclass
class SkillCategories:
    """Categorized skills for better response organization"""
    programming: List[str]
    tools_frameworks: List[str]
    concepts_methods: List[str]
    databases: List[str]
    cloud_platforms: List[str]

class SimpleLLM:
    """Enhanced fallback LLM with improved text processing and response generation"""
    
    def __init__(self, test_mode: bool = False):
        self.model_name = "Enhanced Text Processor"
        self.test_mode = test_mode
        self._stop_words = self._get_stop_words()
        self._technical_patterns = self._compile_technical_patterns()
        self._skill_categories = self._initialize_skill_categories()
        
        if not test_mode:
            logger.info(f"Initialized Enhanced SimpleLLM: {self.model_name}")
    
    @lru_cache(maxsize=1)
    def _get_stop_words(self) -> frozenset:
        """Get stop words (cached for performance)"""
        return frozenset([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'about', 'what', 'how', 'why', 'when', 'where', 'which', 'who', 'is', 'are',
            'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'tell', 'me', 'please', 'from',
            'his', 'her', 'their', 'our', 'your', 'my', 'this', 'that', 'these', 'those'
        ])
    
    def _compile_technical_patterns(self) -> Dict[str, re.Pattern]:
        """Pre-compile regex patterns for technical content detection"""
        return {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*)?(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?'),
            'version': re.compile(r'\bv?\d+\.\d+(?:\.\d+)?\b'),
            'technical_terms': re.compile(r'\b(?:API|SDK|REST|JSON|XML|SQL|NoSQL|CI/CD|DevOps|ML|AI|UI/UX)\b', re.IGNORECASE),
            'programming_languages': re.compile(r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|Ruby|PHP|Swift|Kotlin|Scala)\b', re.IGNORECASE),
            'frameworks': re.compile(r'\b(?:React|Angular|Vue|Django|Flask|Spring|Express|Laravel|TensorFlow|PyTorch|Keras)\b', re.IGNORECASE)
        }
    
    def _initialize_skill_categories(self) -> Dict[str, List[str]]:
        """Initialize skill categorization mappings"""
        return {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 
                'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql'
            ],
            'web_technologies': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 
                'flask', 'spring', 'laravel', 'bootstrap', 'jquery'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 
                'dynamodb', 'oracle', 'sqlite', 'mariadb'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean', 
                'linode', 'kubernetes', 'docker'
            ],
            'data_science': [
                'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 
                'matplotlib', 'seaborn', 'plotly', 'jupyter'
            ],
            'tools_and_frameworks': [
                'git', 'jenkins', 'docker', 'kubernetes', 'terraform', 'ansible', 
                'webpack', 'babel', 'eslint', 'prettier'
            ]
        }

    def invoke(self, prompt: str) -> str:
        """Generate intelligent response with enhanced error handling and validation"""
        # Input validation
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            logger.warning("Empty, invalid, or whitespace-only prompt received")
            return "Please provide a valid question with context."
        
        try:
            logger.debug(f"Processing prompt of length: {len(prompt)}")
            
            # Extract and validate question and context
            question_part, context_part = self._parse_prompt(prompt)
            
            if not self._is_valid_context(context_part):
                logger.warning(f"Invalid context: length={len(context_part) if context_part else 0}")
                return self._handle_insufficient_context()
            
            # Process and generate enhanced response
            cleaned_context = self._clean_context(context_part)
            response = self._generate_intelligent_response(question_part, cleaned_context)
            
            logger.info(f"Successfully generated response of length: {len(response)}")
            return response
            
        except ValueError as ve:
            logger.error(f"Validation error in SimpleLLM.invoke: {str(ve)}")
            return "There was an issue with the input format. Please check your question and try again."
        except Exception as e:
            logger.error(f"Unexpected error in SimpleLLM.invoke: {str(e)}", exc_info=True)
            return "I encountered an error processing your request. Please try again."

    def _is_valid_context(self, context: str) -> bool:
        """Enhanced context validation"""
        if not context:
            return False
        
        context_stripped = context.strip()
        if len(context_stripped) < LLMConstants.MIN_CONTEXT_LENGTH:
            return False
        
        # Check for meaningful content (not just whitespace or minimal text)
        meaningful_lines = [line.strip() for line in context_stripped.split('\n') 
                          if line.strip() and len(line.strip()) > 3]
        
        return len(meaningful_lines) > 0

    def _handle_insufficient_context(self) -> str:
        """Handle cases with insufficient context"""
        return ("I couldn't find sufficient relevant information in the documents to answer your question. "
                "Please ensure your documents contain the information you're looking for, or try rephrasing your question.")

    def _parse_prompt(self, prompt: str) -> Tuple[str, str]:
        """Enhanced prompt parsing with multiple format support"""
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")
        
        # Handle structured prompts (Question: ... Context: ...)
        if "Question:" in prompt and "Context:" in prompt:
            return self._parse_structured_prompt(prompt)
        
        # Handle other common formats
        return self._parse_unstructured_prompt(prompt)

    def _parse_structured_prompt(self, prompt: str) -> Tuple[str, str]:
        """Parse structured prompt with Question: and Context: format"""
        lines = prompt.split("\n")
        question_line = ""
        context_start_idx = None
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.startswith("Question:"):
                question_line = stripped_line.replace("Question:", "").strip()
            elif stripped_line.startswith("Context:"):
                context_start_idx = i + 1
                break
        
        if context_start_idx is not None:
            context_lines = self._filter_context_lines(lines[context_start_idx:])
            context_part = "\n".join(context_lines).strip()
        else:
            context_part = ""
            
        return question_line, context_part

    def _filter_context_lines(self, lines: List[str]) -> List[str]:
        """Filter out unwanted lines from context"""
        filtered_lines = []
        for line in lines:
            stripped_line = line.strip()
            # Skip lines that are instructions or empty
            if (not stripped_line.startswith("Please provide") and 
                not stripped_line.startswith("Based on the above") and
                stripped_line):
                filtered_lines.append(line)
        return filtered_lines

    def _parse_unstructured_prompt(self, prompt: str) -> Tuple[str, str]:
        """Parse unstructured prompts"""
        # Handle "Based on" format
        if "Based on" in prompt:
            parts = prompt.split("Based on", 1)
            return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
        
        # Handle "Given the following" format
        if "Given the following" in prompt:
            parts = prompt.split("Given the following", 1)
            return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
        
        # Default: treat entire prompt as question
        return prompt.strip(), ""

    def _clean_context(self, context: str) -> str:
        """Enhanced context cleaning with better prefix removal"""
        if not context:
            return ""
        
        cleaned_lines = []
        prefix_patterns = {"doc ", "page ", "section ", "chapter ", "file ", "document "}
        
        for line in context.split("\n"):
            clean_line = line.strip()
            
            if not clean_line or len(clean_line) <= LLMConstants.MIN_MEANINGFUL_CONTENT_LENGTH:
                continue
            
            # Enhanced prefix removal
            clean_line = self._remove_document_prefixes(clean_line, prefix_patterns)
            
            # Remove redundant spacing and special characters
            clean_line = re.sub(r'\s+', ' ', clean_line)
            clean_line = re.sub(r'^[-•\*\+]\s*', '', clean_line)  # Remove bullet points
            
            if clean_line:
                cleaned_lines.append(clean_line)
        
        return "\n".join(cleaned_lines)

    def _remove_document_prefixes(self, line: str, prefix_patterns: Set[str]) -> str:
        """Enhanced document prefix removal"""
        if ":" not in line:
            return line
        
        line_lower = line.lower()
        for pattern in prefix_patterns:
            if line_lower.startswith(pattern):
                # More sophisticated pattern matching
                colon_index = line.find(':')
                if colon_index != -1:
                    potential_content = line[colon_index + 1:].strip()
                    if len(potential_content) > 5:  # Ensure meaningful content remains
                        return potential_content
                break
        
        return line

    def _generate_intelligent_response(self, question: str, context: str) -> str:
        """Enhanced response generation with improved accuracy"""
        try:
            question_lower = question.lower()
            
            # Extract most relevant content
            relevant_sentences = self._extract_relevant_sentences(question, context)
            
            if not relevant_sentences:
                return self._handle_no_relevant_content()
            
            # Enhanced query type detection
            query_type = self._detect_query_type(question_lower)
            
            if query_type == "list":
                return self._create_narrative_list_response(question, relevant_sentences, context)
            elif query_type == "comparison":
                return self._create_comparison_response(question, relevant_sentences)
            elif query_type == "technical":
                return self._create_technical_response(question, relevant_sentences)
            else:
                return self._create_narrative_response(question, relevant_sentences)
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I encountered an error while generating the response. Please try again."

    def _detect_query_type(self, question_lower: str) -> str:
        """Enhanced query type detection"""
        list_keywords = [
            "skill", "skills", "technology", "technologies", "tool", "tools", 
            "framework", "frameworks", "library", "libraries", "certification", 
            "certifications", "expertise", "languages", "experience"
        ]
        
        comparison_keywords = [
            "compare", "versus", "vs", "difference", "differences", "better", 
            "best", "advantages", "disadvantages", "pros", "cons"
        ]
        
        technical_keywords = [
            "algorithm", "implementation", "architecture", "design", "pattern",
            "formula", "equation", "calculation", "method", "approach"
        ]
        
        if any(keyword in question_lower for keyword in list_keywords):
            return "list"
        elif any(keyword in question_lower for keyword in comparison_keywords):
            return "comparison"
        elif any(keyword in question_lower for keyword in technical_keywords):
            return "technical"
        else:
            return "general"

    def _handle_no_relevant_content(self) -> str:
        """Enhanced handling when no relevant content is found"""
        return ("I couldn't find specific information related to your question in the provided documents. "
                "This could be because:\n"
                "• The information isn't present in the uploaded documents\n"
                "• The question might need to be more specific\n"
                "• Try rephrasing your question with different keywords")

    def _create_narrative_list_response(self, question: str, relevant_sentences: List[str], context: str) -> str:
        """Enhanced narrative list response with better categorization"""
        # Extract person/entity name from question
        person_name = self._extract_entity_name(question)
        
        # Extract and categorize skills/items
        extracted_items = self._extract_and_categorize_items(context)
        
        # Generate introduction
        intro = self._generate_intro_for_list_response(person_name, extracted_items)
        
        # Create categorized narrative
        narrative_parts = [intro]
        narrative_parts.extend(self._build_skill_narratives(extracted_items))
        
        # Join and clean up the response
        full_narrative = self._finalize_narrative(narrative_parts)
        
        return f"**Skills and Technical Expertise:**\n\n{full_narrative}\n\n*Source: Retrieved from uploaded documents*"

    def _extract_entity_name(self, question: str) -> str:
        """Extract person or entity name from question"""
        question_words = question.split()
        exclude_words = {
            'tell', 'me', 'about', 'what', 'are', 'the', 'skills', 'of', 'from', 
            'his', 'her', 'cv', 'resume', 'experience', 'expertise', 'has', 'does'
        }
        
        name_parts = []
        for word in question_words:
            clean_word = re.sub(r'[^\w]', '', word)
            if (clean_word and len(clean_word) > 2 and 
                clean_word[0].isupper() and 
                clean_word.lower() not in exclude_words):
                name_parts.append(clean_word)
        
        return " ".join(name_parts[:2])  # Limit to first two name parts

    def _extract_and_categorize_items(self, context: str) -> SkillCategories:
        """Enhanced item extraction with categorization"""
        all_items = self._extract_list_items(context, ["skill", "skills", "technology", "technologies", "tool", "tools"])
        
        programming = []
        tools_frameworks = []
        concepts_methods = []
        databases = []
        cloud_platforms = []
        
        skill_categories = self._skill_categories
        
        for item in all_items:
            item_lower = item.lower()
            categorized = False
            
            # Check each category
            for lang in skill_categories['programming_languages']:
                if lang in item_lower:
                    programming.append(item)
                    categorized = True
                    break
            
            if not categorized:
                for db in skill_categories['databases']:
                    if db in item_lower:
                        databases.append(item)
                        categorized = True
                        break
            
            if not categorized:
                for cloud in skill_categories['cloud_platforms']:
                    if cloud in item_lower:
                        cloud_platforms.append(item)
                        categorized = True
                        break
            
            if not categorized:
                for tool in skill_categories['tools_and_frameworks'] + skill_categories['web_technologies']:
                    if tool in item_lower:
                        tools_frameworks.append(item)
                        categorized = True
                        break
            
            if not categorized:
                concepts_methods.append(item)
        
        return SkillCategories(
            programming=programming,
            tools_frameworks=tools_frameworks,
            concepts_methods=concepts_methods,
            databases=databases,
            cloud_platforms=cloud_platforms
        )

    def _generate_intro_for_list_response(self, person_name: str, skills: SkillCategories) -> str:
        """Generate appropriate introduction for list responses"""
        if person_name:
            return f"Based on the available information, {person_name}'s professional profile demonstrates comprehensive technical expertise across multiple domains."
        else:
            total_skills = (len(skills.programming) + len(skills.tools_frameworks) + 
                          len(skills.concepts_methods) + len(skills.databases) + len(skills.cloud_platforms))
            if total_skills > 10:
                return "The document reveals an extensive technical skill set spanning multiple domains and technologies."
            else:
                return "The document outlines a focused technical skill set covering key areas of expertise."

    def _build_skill_narratives(self, skills: SkillCategories) -> List[str]:
        """Build narrative sections for each skill category"""
        narratives = []
        
        if skills.programming:
            prog_text = f"Programming expertise includes {', '.join(skills.programming[:LLMConstants.MAX_SKILLS_PER_CATEGORY])}"
            if len(skills.programming) > LLMConstants.MAX_SKILLS_PER_CATEGORY:
                prog_text += f" and {len(skills.programming) - LLMConstants.MAX_SKILLS_PER_CATEGORY} additional languages"
            narratives.append(prog_text)
        
        if skills.databases:
            db_text = f"Database technologies encompass {', '.join(skills.databases[:LLMConstants.MAX_SKILLS_PER_CATEGORY])}"
            narratives.append(db_text)
        
        if skills.cloud_platforms:
            cloud_text = f"Cloud platform experience covers {', '.join(skills.cloud_platforms[:LLMConstants.MAX_SKILLS_PER_CATEGORY])}"
            narratives.append(cloud_text)
        
        if skills.tools_frameworks:
            tools_text = f"Development tools and frameworks include {', '.join(skills.tools_frameworks[:LLMConstants.MAX_SKILLS_PER_CATEGORY])}"
            narratives.append(tools_text)
        
        if skills.concepts_methods:
            concepts_text = f"Additional competencies cover {', '.join(skills.concepts_methods[:LLMConstants.MAX_SKILLS_PER_CATEGORY])}"
            narratives.append(concepts_text)
        
        return narratives

    def _finalize_narrative(self, narrative_parts: List[str]) -> str:
        """Clean up and finalize the narrative response"""
        if not narrative_parts:
            return "No specific technical skills were identified in the documents."
        
        # Join with proper punctuation
        full_narrative = ". ".join(narrative_parts) + "."
        
        # Clean up formatting
        full_narrative = re.sub(r'\s+', ' ', full_narrative)
        full_narrative = re.sub(r'([.!?])\s*([.!?])', r'\1', full_narrative)
        full_narrative = re.sub(r'\.{2,}', '.', full_narrative)
        
        return full_narrative

    def _create_comparison_response(self, question: str, relevant_sentences: List[str]) -> str:
        """Create response for comparison queries"""
        paragraph = " ".join(relevant_sentences[:8])
        paragraph = re.sub(r'\s+', ' ', paragraph).strip()
        
        return f"**Comparison Analysis:**\n\n{paragraph}\n\n*Source: Retrieved from uploaded documents*"

    def _create_technical_response(self, question: str, relevant_sentences: List[str]) -> str:
        """Create response for technical queries"""
        # Prioritize sentences with technical content
        technical_sentences = []
        general_sentences = []
        
        for sentence in relevant_sentences:
            if any(pattern.search(sentence) for pattern in self._technical_patterns.values()):
                technical_sentences.append(sentence)
            else:
                general_sentences.append(sentence)
        
        # Combine technical first, then general
        combined_sentences = technical_sentences + general_sentences
        paragraph = " ".join(combined_sentences[:8])
        paragraph = re.sub(r'\s+', ' ', paragraph).strip()
        
        return f"**Technical Information:**\n\n{paragraph}\n\n*Source: Retrieved from uploaded documents*"

    def _create_narrative_response(self, question: str, relevant_sentences: List[str]) -> str:
        """Enhanced narrative response for general queries"""
        # Select optimal number of sentences based on content length
        optimal_sentences = LLMConstants.MAX_SENTENCES_FOR_SHORT_RESPONSE
        if sum(len(s) for s in relevant_sentences[:6]) < LLMConstants.MIN_RESPONSE_LENGTH:
            optimal_sentences = LLMConstants.MAX_SENTENCES_FOR_LONG_RESPONSE
        
        paragraph = " ".join(relevant_sentences[:optimal_sentences])
        paragraph = re.sub(r'\s+', ' ', paragraph).strip()
        
        # Remove duplicate punctuation and clean up
        paragraph = re.sub(r'([.!?])\s*([.!?])', r'\1', paragraph)
        
        return f"**Answer:**\n\n{paragraph}\n\n*Source: Retrieved from uploaded documents*"

    @lru_cache(maxsize=256)
    def _extract_keywords(self, question: str) -> frozenset:
        """Extract keywords from question (cached for performance)"""
        question_words = set(re.findall(r"\w+", question.lower()))
        return frozenset(question_words - self._stop_words)

    def _extract_relevant_sentences(self, question: str, context: str) -> List[str]:
        """Enhanced sentence extraction with improved scoring"""
        question_keywords = self._extract_keywords(question)
        
        if not question_keywords:
            return []
        
        # Create chunks for better context preservation
        chunks = self._create_context_chunks(context)
        
        # Score and rank chunks
        scored_chunks = []
        for chunk in chunks:
            score = self._calculate_relevance_score(chunk, question_keywords, question)
            if score > 0:
                scored_chunks.append((chunk, score))
        
        # Sort by score and return top chunks
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored_chunks[:LLMConstants.MAX_RELEVANT_SENTENCES]]

    def _create_context_chunks(self, context: str) -> List[str]:
        """Create meaningful chunks from context"""
        chunks = []
        lines = context.split('\n')
        current_chunk = ""
        
        for line in lines:
            line = line.strip()
            if len(line) > 5:
                current_chunk += " " + line
                
                # Create chunk when it's substantial or at sentence end
                if (len(current_chunk) > LLMConstants.MIN_CHUNK_LENGTH and 
                    (line.endswith('.') or line.endswith('!') or line.endswith('?'))):
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Also include individual substantial lines
        for line in lines:
            line = line.strip()
            if len(line) > LLMConstants.MIN_LINE_LENGTH:
                chunks.append(line)
        
        return list(set(chunks))  # Remove duplicates

    def _calculate_relevance_score(self, chunk: str, question_keywords: frozenset, question: str) -> float:
        """Enhanced relevance scoring with multiple factors"""
        chunk_words = set(re.findall(r"\w+", chunk.lower()))
        
        # Basic keyword overlap
        overlap = len(question_keywords.intersection(chunk_words))
        base_score = overlap
        
        # Boost for exact phrase matches
        phrase_boost = 0
        for keyword in question_keywords:
            if keyword in chunk.lower():
                phrase_boost += LLMConstants.EXACT_MATCH_BOOST
        
        # Boost for technical content
        technical_boost = 0
        for pattern in self._technical_patterns.values():
            if pattern.search(chunk):
                technical_boost += LLMConstants.TECHNICAL_BOOST_SCORE
                break
        
        # Boost for contextual relevance
        context_boost = 0
        if any(keyword in chunk.lower() for keyword in question_keywords):
            context_boost += LLMConstants.PARTIAL_MATCH_BOOST
        
        # Length normalization (prefer substantial content)
        length_factor = min(len(chunk) / 100, 1.5)
        
        total_score = (base_score + phrase_boost + technical_boost + context_boost) * length_factor
        return total_score

    def _extract_list_items(self, context: str, list_keywords: List[str]) -> List[str]:
        """Enhanced list item extraction with better pattern matching"""
        items = set()
        
        # Enhanced regex patterns for skill extraction
        skill_patterns = [
            r'(?:skills?|technologies|tools|frameworks?|libraries|certifications?|expertise|languages)\s*[:\-]?\s*(.+)',
            r'(?:proficient|experienced|skilled)\s+(?:in|with)\s*[:\-]?\s*(.+)',
            r'(?:knowledge|experience)\s+(?:of|in|with)\s*[:\-]?\s*(.+)'
        ]
        
        for line in context.split('\n'):
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Check if line contains skill-related keywords
            line_lower = line_clean.lower()
            if any(keyword in line_lower for keyword in list_keywords):
                
                # Try each pattern
                for pattern in skill_patterns:
                    match = re.search(pattern, line_lower)
                    if match:
                        candidates = re.split(r'[,;]\s*', match.group(1))
                        for candidate in candidates:
                            cleaned_item = self._clean_skill_item(candidate)
                            if cleaned_item and len(cleaned_item) > 1:
                                items.add(cleaned_item)
                        break
            
            # Also check for comma-separated lists (even without keywords)
            elif line_clean.count(',') >= LLMConstants.MIN_COMMA_COUNT_FOR_LIST:
                candidates = re.split(r'[,;]\s*', line_clean)
                for candidate in candidates:
                    cleaned_item = self._clean_skill_item(candidate)
                    if (cleaned_item and len(cleaned_item) > 2 and 
                        self._is_likely_skill(cleaned_item)):
                        items.add(cleaned_item)
        
        return sorted(list(items))

    def _clean_skill_item(self, item: str) -> str:
        """Clean and normalize skill items"""
        # Remove parenthetical information
        item = re.sub(r'\([^)]*\)', '', item)
        
        # Remove common prefixes and suffixes
        item = re.sub(r'^(?:and|or|with|using|including)\s+', '', item, flags=re.IGNORECASE)
        item = re.sub(r'\s+(?:and|or|with|using|including)$', '', item, flags=re.IGNORECASE)
        
        # Clean up spacing and punctuation
        item = re.sub(r'\s+', ' ', item.strip())
        item = re.sub(r'^[^\w]+|[^\w]+$', '', item)
        
        return item

    def _is_likely_skill(self, item: str) -> bool:
        """Determine if an item is likely a technical skill"""
        item_lower = item.lower()
        
        # Check against known skill categories
        for category_skills in self._skill_categories.values():
            if any(skill in item_lower for skill in category_skills):
                return True
        
        # Check for technical patterns
        technical_indicators = [
            r'\b\w+\.(js|py|java|cpp|cs|php|rb|go|rs)\b',  # File extensions
            r'\bv?\d+\.\d+\b',  # Version numbers
            r'\b[A-Z]{2,}\b',   # Acronyms
            r'\w+(?:Script|Lang|DB|SQL|API|SDK)\b'  # Technical suffixes
        ]
        
        return any(re.search(pattern, item, re.IGNORECASE) for pattern in technical_indicators)

    def set_test_mode(self, enabled: bool) -> None:
        """Enable/disable test mode for unit testing"""
        self.test_mode = enabled


class ModelManager:
    """Enhanced centralized model manager with improved configuration and error handling"""
    
    def __init__(self):
        """Initialize the model manager with validation and enhanced logging"""
        self.current_model = None
        self.model_info = {}
        
        # Validate configuration
        self._validate_config()
        
        logger.info("Enhanced ModelManager initialized with configuration:")
        logger.info(f"  Preferred model: {config.PREFERRED_MODEL}")
        logger.info(f"  Force local: {config.FORCE_LOCAL_MODEL}")
        logger.info(f"  Local model: {config.LOCAL_MODEL_NAME}")
        logger.info(f"  Temperature: {config.DEFAULT_TEMPERATURE}")
        logger.info(f"  Max tokens: {config.MAX_NEW_TOKENS}")
        
    def _validate_config(self) -> None:
        """Validate configuration settings"""
        validations = [
            (hasattr(config, 'PREFERRED_MODEL') and config.PREFERRED_MODEL, "PREFERRED_MODEL must be set"),
            (0.0 <= config.DEFAULT_TEMPERATURE <= 2.0, "DEFAULT_TEMPERATURE must be between 0.0 and 2.0"),
            (config.MAX_NEW_TOKENS > 0, "MAX_NEW_TOKENS must be positive"),
            (hasattr(config, 'LOG_LEVEL'), "LOG_LEVEL must be configured")
        ]
        
        for condition, error_message in validations:
            if not condition:
                raise ValueError(error_message)
        
        logger.info("Configuration validation passed")
        
    def get_model(self, temperature: Optional[float] = None) -> Tuple[Any, Dict[str, Any]]:
        """Get the appropriate LLM model with enhanced fallback logic"""
        if temperature is None:
            temperature = config.DEFAULT_TEMPERATURE
            
        # Validate temperature
        if not (0.0 <= temperature <= 2.0):
            logger.warning(f"Invalid temperature {temperature}, using default {config.DEFAULT_TEMPERATURE}")
            temperature = config.DEFAULT_TEMPERATURE
            
        logger.info(f"Loading model with temperature: {temperature}")
        
        # Try Gemini model first (if conditions are met)
        if self._should_use_gemini():
            try:
                return self._load_gemini_model(temperature)
            except Exception as e:
                logger.warning(f"Failed to load Gemini model: {str(e)}")
        
        # Fallback to enhanced local processor
        return self._load_local_processor(temperature)
    
    def _should_use_gemini(self) -> bool:
        """Determine if Gemini model should be used"""
        return (not config.FORCE_LOCAL_MODEL and 
                "gemini" in config.PREFERRED_MODEL.lower() and
                config.GOOGLE_API_KEY and 
                config.GOOGLE_API_KEY.strip())
    
    def _load_gemini_model(self, temperature: float) -> Tuple[Any, Dict[str, Any]]:
        """Load Gemini model with enhanced error handling"""
        try:
            from langchain_google_genai import GoogleGenerativeAI
            
            model_name = config.PREFERRED_MODEL
            logger.info(f"Loading Gemini model: {model_name}")
            
            # Initialize with robust parameters
            gemini_model = GoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=config.GOOGLE_API_KEY,
                max_output_tokens=config.MAX_OUTPUT_TOKENS,
                top_p=getattr(config, 'TOP_P', 0.9)
            )
            
            # Test the model with a simple query
            test_response = gemini_model.invoke("Hello, please respond with 'OK' if you're working.")
            logger.info(f"Gemini test successful: {test_response[:50]}...")
            
            # Set model info
            self.current_model = "gemini"
            self.model_info = {
                "name": "Gemini",
                "version": model_name,
                "type": "API",
                "device": "Cloud",
                "temperature": temperature,
                "max_tokens": config.MAX_OUTPUT_TOKENS,
                "config_source": "config.py",
                "status": "active",
                "test_response": test_response[:100] if test_response else "No response"
            }
            
            logger.info("Successfully loaded and tested Gemini model")
            return gemini_model, self.model_info
            
        except ImportError as e:
            logger.error(f"Gemini dependencies not available: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise
    
    def _load_local_processor(self, temperature: float) -> Tuple[SimpleLLM, Dict[str, Any]]:
        """Load enhanced local text processor"""
        logger.info("Using enhanced local text processor")
        
        # Initialize enhanced SimpleLLM
        simple_llm = SimpleLLM()
        
        self.current_model = "enhanced_local_processor"
        self.model_info = {
            "name": "Enhanced Text Processor",
            "version": "2.0",
            "type": "Rule-based",
            "device": "Local",
            "temperature": temperature,
            "max_tokens": config.MAX_NEW_TOKENS,
            "config_source": "config.py",
            "status": "active",
            "features": [
                "Enhanced context parsing",
                "Improved skill extraction",
                "Better response categorization",
                "Technical content detection",
                "Multi-format prompt support"
            ]
        }
        
        logger.info(f"Enhanced local processor loaded: {self.model_info}")
        return simple_llm, self.model_info
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the currently loaded model"""
        if not self.model_info:
            return {"error": "No model currently loaded"}
        
        # Add runtime information
        runtime_info = self.model_info.copy()
        runtime_info.update({
            "load_time": getattr(self, 'load_time', 'Unknown'),
            "config_validation": "Passed",
            "available_features": self._get_available_features()
        })
        
        return runtime_info
    
    def _get_available_features(self) -> List[str]:
        """Get list of available features based on current model"""
        base_features = [
            "Document processing",
            "Question answering", 
            "Context extraction",
            "Response generation"
        ]
        
        if self.current_model == "enhanced_local_processor":
            base_features.extend([
                "Enhanced skill extraction",
                "Technical content detection",
                "Multi-category response generation",
                "Improved accuracy algorithms"
            ])
        elif self.current_model == "gemini":
            base_features.extend([
                "Advanced language understanding",
                "High-quality response generation",
                "Cloud-based processing"
            ])
        
        return base_features

    def reload_model(self, temperature: Optional[float] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reload the current model with new parameters"""
        logger.info("Reloading model with new parameters")
        self.current_model = None
        self.model_info = {}
        return self.get_model(temperature)

    def get_model_status(self) -> Dict[str, Any]:
        """Get detailed status of the model manager"""
        return {
            "current_model": self.current_model,
            "model_loaded": self.current_model is not None,
            "config_valid": True,  # Since we validate on init
            "available_models": ["gemini", "enhanced_local_processor"],
            "model_info": self.model_info
        }
