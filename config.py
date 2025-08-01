"""
Centralized Configuration Management
All hyperparameters, model settings, paths, and variables in one place
"""
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AppConfig:
    """Centralized configuration class for the entire application"""
    
    # ===== API KEYS =====
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    
    # ===== MODEL CONFIGURATION =====
    PREFERRED_MODEL = os.getenv("PREFERRED_MODEL", "gemini-1.5-flash")
    FORCE_LOCAL_MODEL = os.getenv("FORCE_LOCAL_MODEL", "true").lower() == "true"
    LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "distilgpt2")
    
    # Model hyperparameters
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))
    MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
    TOP_P = float(os.getenv("TOP_P", "0.9"))
    
    # Model selection priority (fastest to slowest)
    LOCAL_MODEL_CANDIDATES = [
        ("distilgpt2", "82M parameter model - fastest"),
        ("facebook/opt-350m", "350M parameter model - fast"),
        ("microsoft/DialoGPT-medium", "354M parameter model - medium"),
        ("microsoft/DialoGPT-large", "774M parameter model - slow")
    ]
    
    # ===== VECTOR DATABASE SETTINGS =====
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    
    # ===== DOCUMENT PROCESSING =====
    DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "800"))
    DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "100"))
    SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".doc", ".txt"]
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    
    # ===== CACHE SETTINGS =====
    CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_EXPIRY_HOURS = int(os.getenv("CACHE_EXPIRY_HOURS", "24"))
    
    # ===== STREAMLIT UI SETTINGS =====
    PAGE_TITLE = os.getenv("PAGE_TITLE", "Enterprise Document Search")
    PAGE_ICON = os.getenv("PAGE_ICON", "🔍")
    LAYOUT = os.getenv("LAYOUT", "wide")
    SIDEBAR_STATE = os.getenv("SIDEBAR_STATE", "expanded")
    
    # ===== RESPONSE GENERATION =====
    MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", "600"))
    RESPONSE_TIMEOUT_SECONDS = int(os.getenv("RESPONSE_TIMEOUT_SECONDS", "30"))
    ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "false").lower() == "true"
    
    # ===== LOGGING CONFIGURATION =====
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")
    ENABLE_FILE_LOGGING = os.getenv("ENABLE_FILE_LOGGING", "false").lower() == "true"
    
    # ===== PERFORMANCE SETTINGS =====
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
    ENABLE_GPU = os.getenv("ENABLE_GPU", "false").lower() == "true"
    
    # ===== DEBUG AND DEVELOPMENT =====
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
    SHOW_TECHNICAL_INFO = os.getenv("SHOW_TECHNICAL_INFO", "true").lower() == "true"
    
    @classmethod
    def get_all_config(cls) -> Dict[str, Any]:
        """Get all configuration as a dictionary for logging/debugging"""
        config = {}
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                config[attr] = getattr(cls, attr)
        return config
    
    @classmethod
    def log_config(cls, logger) -> None:
        """Log all configuration settings using f-strings"""
        logger.info("=== APPLICATION CONFIGURATION ===")
        
        # API Configuration
        logger.info(f"API Keys:")
        logger.info(f"  GOOGLE_API_KEY: {'Set' if cls.GOOGLE_API_KEY else 'Not Set'}")
        
        # Model Configuration
        logger.info(f"Model Configuration:")
        logger.info(f"  PREFERRED_MODEL: {cls.PREFERRED_MODEL}")
        logger.info(f"  FORCE_LOCAL_MODEL: {cls.FORCE_LOCAL_MODEL}")
        logger.info(f"  LOCAL_MODEL_NAME: {cls.LOCAL_MODEL_NAME}")
        logger.info(f"  DEFAULT_TEMPERATURE: {cls.DEFAULT_TEMPERATURE}")
        logger.info(f"  MAX_OUTPUT_TOKENS: {cls.MAX_OUTPUT_TOKENS}")
        logger.info(f"  MAX_NEW_TOKENS: {cls.MAX_NEW_TOKENS}")
        logger.info(f"  TOP_P: {cls.TOP_P}")
        
        # Vector Database
        logger.info(f"Vector Database:")
        logger.info(f"  VECTOR_DB_PATH: {cls.VECTOR_DB_PATH}")
        logger.info(f"  EMBEDDING_MODEL_NAME: {cls.EMBEDDING_MODEL_NAME}")
        logger.info(f"  EMBEDDING_DEVICE: {cls.EMBEDDING_DEVICE}")
        logger.info(f"  SIMILARITY_THRESHOLD: {cls.SIMILARITY_THRESHOLD}")
        logger.info(f"  MAX_SEARCH_RESULTS: {cls.MAX_SEARCH_RESULTS}")
        
        # Document Processing
        logger.info(f"Document Processing:")
        logger.info(f"  DEFAULT_CHUNK_SIZE: {cls.DEFAULT_CHUNK_SIZE}")
        logger.info(f"  DEFAULT_CHUNK_OVERLAP: {cls.DEFAULT_CHUNK_OVERLAP}")
        logger.info(f"  MAX_FILE_SIZE_MB: {cls.MAX_FILE_SIZE_MB}")
        logger.info(f"  SUPPORTED_FILE_TYPES: {cls.SUPPORTED_FILE_TYPES}")
        
        # Performance & Debug
        logger.info(f"Performance & Debug:")
        logger.info(f"  DEBUG: {cls.DEBUG}")
        logger.info(f"  VERBOSE_LOGGING: {cls.VERBOSE_LOGGING}")
        logger.info(f"  ENABLE_GPU: {cls.ENABLE_GPU}")
        logger.info(f"  MAX_CONCURRENT_REQUESTS: {cls.MAX_CONCURRENT_REQUESTS}")
        
        logger.info("=== END CONFIGURATION ===")

# Create a global instance for easy access
config = AppConfig()

# Export commonly used variables for backward compatibility
PREFERRED_MODEL = config.PREFERRED_MODEL
FORCE_LOCAL_MODEL = config.FORCE_LOCAL_MODEL
LOCAL_MODEL_NAME = config.LOCAL_MODEL_NAME
DEFAULT_TEMPERATURE = config.DEFAULT_TEMPERATURE
VECTOR_DB_PATH = config.VECTOR_DB_PATH
DEBUG = config.DEBUG
