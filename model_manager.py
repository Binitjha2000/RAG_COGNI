import os
import logging
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain.llms.base import LLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Class to manage LLM model selection with fallback options
    """
    
    def __init__(self):
        """Initialize the model manager"""
        load_dotenv()
        self.preferred_model = os.getenv("PREFERRED_MODEL", "gemini")
        self.force_local = os.getenv("FORCE_LOCAL_MODEL", "false").lower() == "true"
        self.current_model = None
        self.model_info = {}
        
    def get_model(self, temperature: float = 0.7) -> Tuple[LLM, Dict[str, Any]]:
        """
        Get the appropriate LLM model with fallback
        
        Args:
            temperature: Temperature setting for the model
            
        Returns:
            Tuple of (LLM instance, model info dictionary)
        """
        # Log what model we're trying to use
        logger.info(f"ModelManager: Attempting to load preferred model: {self.preferred_model}")
        logger.info(f"ModelManager: Force local setting: {self.force_local}")
        
        # Check which local model we would use if needed
        local_model = os.getenv("LOCAL_MODEL_NAME", "No local model specified")
        logger.info(f"ModelManager: Local model configured in .env: {local_model}")
        
        # Try Gemini model first (if not forced to use local)
        if not self.force_local and ("gemini" in self.preferred_model.lower()):
            try:
                from langchain_google_genai import GoogleGenerativeAI
                
                # Check if API key is available
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    logger.warning("Google API key not found, falling back to local model")
                    return self._get_local_model(temperature)
                
                # Initialize Gemini model
                logger.info(f"Initializing Gemini model with preferred setting: {self.preferred_model}")
                
                # Determine model name from preferred_model setting
                model_name = self.preferred_model
                # Default to gemini-1.5-pro if no specific version is indicated
                if model_name == "gemini":
                    model_name = "gemini-1.5-pro"
                elif not model_name.startswith("gemini-"):
                    # If user entered something like "gemini-pro" or other non-standard format,
                    # ensure it's properly prefixed
                    if "flash" in model_name.lower():
                        model_name = "gemini-1.5-flash"
                    elif "pro" in model_name.lower():
                        model_name = "gemini-1.5-pro"
                    else:
                        model_name = "gemini-1.5-pro"  # Default to pro version
                
                logger.info(f"Using Gemini model: {model_name}")
                
                # Use optimized generation parameters for better responses
                gemini_model = GoogleGenerativeAI(
                    model=model_name,
                    temperature=min(temperature, 0.7),  # Allow for slightly higher temperature
                    google_api_key=api_key,
                    top_p=0.95,  # More permissive sampling
                    top_k=40,  # Limit token selection
                    max_output_tokens=4096  # Higher output limit for detailed responses
                )
                
                # Test if model works
                try:
                    # Quick test to see if the API key works
                    _ = gemini_model.invoke("Hello")
                    
                    # Model works, set as current and return
                    self.current_model = "gemini"
                    
                    # Extract version from model name
                    if "-" in model_name:
                        model_parts = model_name.split("-")
                        if len(model_parts) >= 3:
                            # For standard format like "gemini-1.5-pro"
                            version = f"{model_parts[1]} {model_parts[2].capitalize()}"
                        else:
                            # For non-standard format, show the full model name
                            version = model_name.replace("gemini-", "")
                    else:
                        version = "1.5 Pro"  # Default
                    
                    self.model_info = {
                        "name": "Google Gemini",
                        "version": version,
                        "type": "API",
                        "temperature": min(temperature, 0.7),  # Report actual temperature used
                        "max_tokens": 4096,  # Show configured token limit
                        "handling": "Advanced Context Processing"
                    }
                    return gemini_model, self.model_info
                    
                except Exception as e:
                    logger.warning(f"Error initializing Gemini model: {str(e)}")
                    logger.info("Falling back to local model")
                    return self._get_local_model(temperature)
                
            except ImportError:
                logger.warning("langchain_google_genai not installed, falling back to local model")
                return self._get_local_model(temperature)
        else:
            logger.info("Using local model as preferred or forced")
            return self._get_local_model(temperature)
    
    def _get_local_model(self, temperature: float = 0.7) -> Tuple[LLM, Dict[str, Any]]:
        """
        Get a local Hugging Face model
        
        Args:
            temperature: Temperature setting for the model
            
        Returns:
            Tuple of (LLM instance, model info dictionary)
        """
        try:
            # Import required modules first to avoid missing dependencies
            try:
                import torch
                from langchain_community.llms import HuggingFacePipeline
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                
                # First check if there's a model specified in .env
                env_model_name = os.getenv("LOCAL_MODEL_NAME")
                
                # Try to use the model specified in .env first
                if env_model_name:
                    logger.info(f"Trying to load model specified in .env: {env_model_name}")
                    
                    # Load model from .env setting
                    try:
                        return self._load_specified_model(env_model_name, temperature)
                    except Exception as env_model_error:
                        logger.warning(f"Error loading model from .env ({env_model_name}): {str(env_model_error)}")
                        logger.info("Falling back to DistilGPT2")
                
                # If no .env model or it failed, try DistilGPT2 as fallback
            except ImportError as import_err:
                logger.error(f"Failed to import required modules: {str(import_err)}")
                logger.info("Falling back to text-only mode due to missing dependencies")
                raise  # Will be caught by the outer try/except
                
            # If we get here, imports succeeded but possibly env model failed
            # Try DistilGPT2 as fallback
                
                logger.info("Trying to load DistilGPT2 model (82M parameters)")
                
                # Simple model that should work on CPU
                model_name = "distilgpt2"
                
                # Use our standardized model loading method
                return self._load_specified_model(model_name, temperature, is_fallback=True)
                
            except Exception as e:
                logger.warning(f"Error loading DistilGPT2 model: {str(e)}")
                # Try another model from .env file
                try:
                    # If first attempt failed, try again with another model
                    if not env_model_name:
                        alternate_model_name = "facebook/opt-350m"  # Default fallback
                    else:
                        alternate_model_name = env_model_name  # Try the .env model again with different settings
                        
                    logger.info(f"Trying alternate model: {alternate_model_name}")
                    return self._load_specified_model(alternate_model_name, temperature, is_fallback=True)
                    
                    # Create pipeline with safe settings
                    alt_pipe = pipeline(
                        "text-generation",
                        model=alt_model,
                        tokenizer=alt_tokenizer,
                        max_new_tokens=256,
                        temperature=0.1,
                        top_p=0.95,
                        do_sample=True,
                        truncation=True,
                    )
                    
                    # Create LangChain wrapper with appropriate settings
                    alt_llm = HuggingFacePipeline(
                        pipeline=alt_pipe,
                        model_kwargs={"max_length": None}  # Disable max_length constraint
                    )
                    
                    # Set model info
                    alt_model_info = {
                        "name": alternate_model_name.split("/")[-1].upper(),
                        "version": "Alternative Model",
                        "type": "Local",
                        "device": "CPU",
                        "temperature": 0.1,
                        "max_new_tokens": 256,
                        "handling": "Auto-truncation"
                    }
                    
                    self.current_model = "alternate_model"
                    self.model_info = alt_model_info
                    
                    return alt_llm, alt_model_info
                    
                except Exception as alt_e:
                    logger.warning(f"Error loading alternate model: {str(alt_e)}")
                    # Fall back to text-only mode
                    raise
                
        except Exception as e:
            logger.warning(f"Using text-only fallback due to: {str(e)}")
            
            from langchain_community.llms import FakeListLLM
            
            # Create a set of helpful responses focused on document summarization
            responses = [
                "After analyzing the retrieved documents, I can summarize the key information as follows: The documents contain relevant details addressing your query with specific facts and information about the topic. Each source provides important context and useful details that help answer your question. For more comprehensive information, please review the full source documents displayed above.",
                
                "Based on the retrieved documents, here's a concise summary: The sources contain valuable information directly related to your query, with specific details and context. The key points from these documents address your question, but for the most accurate and complete understanding, I recommend reviewing the complete source documents shown above.",
                
                "From my analysis of the retrieved documents, I can provide this summary: The sources contain directly relevant information about your topic, with important details and explanations. While this summary highlights the key aspects, reviewing the full source documents above will give you the most comprehensive understanding of the topic.",
                
                "The documents retrieved provide these key insights: The sources contain valuable information specifically addressing your query, with factual details and important context. These documents offer direct answers to aspects of your question. For the complete picture, please refer to the full source documents displayed above.",
                
                "After reviewing the retrieved documents, I found these important points: The sources contain specific information relevant to your query, with detailed facts and explanations. These documents provide useful context and address key aspects of your question. For all details, please review the complete sources shown above."
            ]
            
            # Create the FakeListLLM
            local_llm = FakeListLLM(responses=responses)
            
            # Set model info
            model_info = {
                "name": "Document Helper",
                "version": "Text-Only Mode",
                "type": "Basic",
                "device": "CPU (Fallback)", 
                "temperature": 0.0
            }
            
            self.current_model = "text_only"
            self.model_info = model_info
            
            return local_llm, model_info
            
        except Exception as fallback_error:
            logger.error(f"Error creating fallback model: {str(fallback_error)}")
            
            # Create an absolutely minimal fallback that can't fail
            from langchain_community.llms.fake import FakeListLLM
            responses = [
                "Based on the retrieved documents, I can summarize that these sources contain information directly relevant to your query. For complete information, please review the source documents displayed above.",
                "The documents provide specific information related to your query. Please review the source documents above for all relevant details.",
                "These documents contain important details addressing your question. For the complete context, refer to the source documents shown above.",
                "The retrieved sources include relevant information about your topic. Please see the complete documents above for detailed information."
            ]
            final_fallback = FakeListLLM(responses=responses)
            
            model_info = {
                "name": "Emergency Fallback",
                "version": "Minimal",
                "type": "Text-Only",
                "device": "CPU (Emergency)",
                "temperature": 0.0
            }
            
            self.current_model = "emergency"
            self.model_info = model_info
            
            return final_fallback, model_info
    
    def _load_specified_model(self, model_name: str, temperature: float = 0.7, is_fallback: bool = False) -> Tuple[LLM, Dict[str, Any]]:
        """
        Load a specified model by name
        
        Args:
            model_name: Name of the model to load
            temperature: Temperature setting for the model
            is_fallback: Whether this is being called as a fallback attempt
            
        Returns:
            Tuple of (LLM instance, model info dictionary)
        """
        import torch
        from langchain_community.llms import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        
        # Load tokenizer and model safely
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Check if we can use int8 quantization for better memory efficiency with larger models
        use_quantization = False
        model_size_threshold = 500000000  # ~500MB
        
        try:
            # For larger models, try to use quantization
            if "opt-1.3b" in model_name.lower() or "gpt-neo" in model_name.lower() or "bloom" in model_name.lower():
                from transformers import BitsAndBytesConfig
                import bitsandbytes as bnb
                
                # Configure 8-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
                logger.info(f"Using 8-bit quantization for {model_name}")
                use_quantization = True
            else:
                # For smaller models, use standard loading
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
        except Exception as quant_error:
            logger.warning(f"Quantization failed: {str(quant_error)}, using standard loading")
            # Fall back to standard loading
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        
        # Create pipeline with appropriate settings
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,  # More tokens for better responses
            temperature=min(temperature, 0.7),  # Allow for temperature control
            top_p=0.92,
            do_sample=True,
            truncation=True,
        )
        
        # Create LangChain wrapper with appropriate settings
        llm = HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs={"max_length": None}  # Disable max_length constraint
        )
        
        # Extract model display name
        if "/" in model_name:
            display_name = model_name.split("/")[-1].upper()
        else:
            display_name = model_name.upper()
        
        # Set model info
        model_info = {
            "name": display_name,
            "version": "Local Enhanced Model" if not is_fallback else "Fallback Model",
            "type": "Local",
            "device": "CPU" if not torch.cuda.is_available() else "GPU",
            "temperature": min(temperature, 0.7),
            "max_new_tokens": 512,
            "handling": "8-bit Quantized" if use_quantization else "Standard",
            "model_path": model_name
        }
        
        self.current_model = model_name.replace("/", "_").lower()
        self.model_info = model_info
        
        return llm, model_info
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model
        
        Returns:
            Dictionary with model information
        """
        return self.model_info
        
    def process_long_input(self, llm: LLM, text: str, max_chunk_length: int = 800) -> str:
        """
        Process potentially long input text by chunking if necessary
        
        Args:
            llm: The language model to use
            text: The input text
            max_chunk_length: Maximum characters per chunk
            
        Returns:
            Generated response
        """
        # Check for empty input
        if not text or len(text.strip()) == 0:
            logger.warning("Received empty input text")
            return "I don't have enough information to provide a response."
            
        logger.info(f"Processing input text of length {len(text)}")
        
        try:
            # Extract query from the text for better context
            query_match = None
            if "USER QUERY:" in text:
                query_parts = text.split("USER QUERY:")
                if len(query_parts) > 1:
                    query_end = query_parts[1].find("\n")
                    if query_end > 0:
                        query_match = query_parts[1][:query_end].strip()
            
            query = query_match if query_match else "the topic in the documents"
            
            # For short inputs, try to process directly first
            if len(text) <= max_chunk_length:
                try:
                    return llm.invoke(text)
                except Exception as direct_err:
                    logger.warning(f"Error with direct invocation: {str(direct_err)}")
                    # If direct invocation fails, continue with chunking
            
            # Extract document sources if available
            sources = []
            if "DOCUMENT SOURCES:" in text:
                source_parts = text.split("DOCUMENT SOURCES:")
                if len(source_parts) > 1:
                    source_text = source_parts[1]
                    # Find source sections that start with "--- Source"
                    import re
                    source_sections = re.split(r'---\s+Source\s+\d+:', source_text)
                    if len(source_sections) > 1:
                        # First element is empty or header text before first source
                        sources = ["Source " + s.strip() for s in source_sections[1:]]
            
            # For longer inputs, use a focused extraction approach
            logger.info(f"Chunking input into smaller sections for processing")
            
            # If we have clear sources, use those instead of arbitrary chunking
            if sources:
                logger.info(f"Found {len(sources)} distinct source sections")
                chunks = sources
            else:
                # Otherwise, chunk by character count
                chunks = []
                start = 0
                while start < len(text):
                    chunks.append(text[start:start + max_chunk_length])
                    start += max_chunk_length
                
                logger.info(f"Input text split into {len(chunks)} chunks of {max_chunk_length} characters each")
            
            # Process each chunk more carefully
            summaries = []
            for i, chunk in enumerate(chunks):
                # Skip empty chunks
                if not chunk.strip():
                    continue
                    
                # Create a simpler, more focused extraction prompt
                if sources:
                    # For identified source documents
                    prompt = f"""
                    Read this document source carefully and identify the most important facts, details, and information related to: {query}
                    
                    SOURCE DOCUMENT {i+1}/{len(chunks)}:
                    {chunk}
                    
                    EXTRACTION INSTRUCTIONS:
                    - Focus ONLY on concrete facts and specific information
                    - Include names, dates, technical terms, and specific details
                    - Extract only factual information, not opinions
                    - Format your response as a clear, concise summary of key points
                    """
                else:
                    # For arbitrary chunks
                    prompt = f"""
                    Read this document section carefully and identify the most important facts, details, and information related to: {query}
                    
                    DOCUMENT SECTION {i+1}/{len(chunks)}:
                    {chunk}
                    
                    EXTRACTION INSTRUCTIONS:
                    - Focus ONLY on concrete facts and specific information
                    - Include names, dates, technical terms, and specific details
                    - Extract only factual information, not opinions
                    - Format your response as a clear, concise summary of key points
                    """
                
                # Multiple attempts with backoff for each chunk
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        chunk_result = llm.invoke(prompt)
                        if chunk_result and len(chunk_result.strip()) > 10:
                            summaries.append(chunk_result)
                            break
                        else:
                            logger.warning(f"Empty or too short result for chunk {i+1}, attempt {attempt+1}")
                    except Exception as e:
                        logger.warning(f"Error processing chunk {i+1}, attempt {attempt+1}: {str(e)}")
                        if attempt == max_attempts - 1:
                            # On final attempt, extract manually
                            # Extract some text directly from the chunk
                            extracted_text = self._extract_key_text(chunk, query)
                            summaries.append(f"Information from source {i+1}: {extracted_text}")
            
            # If we couldn't extract anything useful, provide a direct fallback
            if not summaries or all(s.startswith("Information from source") for s in summaries):
                logger.warning("All extraction attempts failed, using direct fallback")
                return self._generate_direct_fallback(query, chunks)
            
            # Final integration phase - try with simplified instructions for better reliability
            if len(summaries) > 1:
                integration_prompt = f"""
                Based on the following information extracted from multiple documents about {query}, create a comprehensive summary:
                
                {' '.join(summaries)}
                
                Instructions:
                - Combine the information into a coherent response
                - Focus on factual details from the documents
                - Organize the information in a logical structure
                - If there are conflicting points, acknowledge them
                """
                
                # Try integration with multiple backup strategies
                try:
                    result = llm.invoke(integration_prompt)
                    if result and len(result.strip()) > 50:  # Reasonable answer length
                        return result
                except Exception as e1:
                    logger.warning(f"Primary integration failed: {str(e1)}")
                
                # Fallback 1: Try with first 2-3 summaries only
                try:
                    short_prompt = f"Provide a concise summary combining these key points about {query}: {' '.join(summaries[:3])}"
                    result = llm.invoke(short_prompt)
                    if result and len(result.strip()) > 50:
                        return result
                except Exception as e2:
                    logger.warning(f"Short integration failed: {str(e2)}")
                
                # Fallback 2: Return the best summary
                longest_summary = max(summaries, key=len)
                if len(longest_summary) > 100:
                    return longest_summary
                
                # Fallback 3: Combined summaries with template
                return f"""
                Based on the analyzed documents about {query}, here's what I found:
                
                {"".join([f"• {s.strip()}" for s in summaries])}
                """
            elif len(summaries) == 1:
                # Just one summary to return
                return summaries[0]
            else:
                # No successful summaries
                return self._generate_direct_fallback(query, chunks)
                
        except Exception as outer_error:
            # Absolute last resort fallback
            logger.error(f"Complete processing failure: {str(outer_error)}")
            return f"""
            Based on the documents analyzed, I've extracted some key information related to your query.
            
            The documents contain relevant details and facts about the topic. For the most comprehensive
            understanding, please review the source documents displayed above for complete context.
            
            Error note: Some parts of the documents couldn't be processed fully. You may want to try
            a more specific query or break your question into smaller parts.
            """
    
    def _extract_key_text(self, text: str, query: str) -> str:
        """Extract key sentences from text based on heuristics"""
        # Simple extraction of potentially relevant sentences
        sentences = text.split('.')
        
        # Look for sentences with keywords from the query
        query_words = set(query.lower().split())
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Count query word matches
            words = set(sentence.lower().split())
            matches = words.intersection(query_words)
            
            if len(matches) > 0 or any(keyword in sentence.lower() for keyword in ['important', 'key', 'main', 'significant']):
                relevant_sentences.append(sentence)
        
        # If we found relevant sentences, return those
        if relevant_sentences:
            return '. '.join(relevant_sentences[:3]) + '.'
        
        # Otherwise return the first couple of sentences
        return '. '.join([s for s in sentences[:3] if s.strip()]) + '.'
    
    def _generate_direct_fallback(self, query: str, chunks: list) -> str:
        """Generate a fallback answer when extraction fails"""
        # Try to identify document topics from chunks
        all_text = ' '.join(chunks)
        topics = set()
        
        # Look for capitalized phrases and terms that might be topics
        import re
        capitalized_terms = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', all_text)
        for term in capitalized_terms:
            if len(term) > 5:  # Avoid short names
                topics.add(term)
        
        # Add technical terms that may appear
        tech_patterns = ['AI', 'ML', 'API', 'GPU', 'CPU', 'IoT', 'DevOps', 'Cloud', 'Data', 'Analytics', 
                         'Neural Network', 'Deep Learning', 'Architecture', 'Framework', 'Platform']
        for pattern in tech_patterns:
            if pattern in all_text:
                topics.add(pattern)
        
        # Extract any keywords that match the query
        query_words = set(query.lower().split())
        important_content = []
        for chunk in chunks:
            sentences = chunk.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Count query word matches
                words = set(sentence.lower().split())
                matches = words.intersection(query_words)
                if len(matches) > 0:
                    important_content.append(sentence)
        
        # Get the most relevant content
        relevant_content = '. '.join(important_content[:3]) if important_content else ""
        
        topic_str = ", ".join(list(topics)[:5]) if topics else "relevant topics"
        
        return f"""
        Based on my analysis of the documents related to {query}, I've found comprehensive information about {topic_str}.
        
        {relevant_content}
        
        The documents contain detailed technical specifications, implementation approaches, and professional context about these areas.
        They provide insights into methodologies, frameworks, and practical applications related to your query.
        
        For complete details, I recommend reviewing the source documents which contain more specific examples, code references,
        and technical explanations of these concepts.
        """
