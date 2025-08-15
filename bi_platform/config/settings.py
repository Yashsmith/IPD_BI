"""
Configuration management for BI Platform
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent / "config.env")

class Config:
    """Configuration class for BI Platform"""
    
    # News API Settings
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
    NEWSAPI_BASE_URL = os.getenv("NEWSAPI_BASE_URL", "https://newsapi.org/v2")
    
    # Financial Data APIs
    ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
    QUANDL_KEY = os.getenv("QUANDL_KEY")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///bi_platform.db")
    
    # ML Model Settings
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")
    MAX_NEWS_ARTICLES_PER_FETCH = int(os.getenv("MAX_NEWS_ARTICLES_PER_FETCH", "100"))
    SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "textblob")
    
    # Agent Configuration
    GRPO_POPULATION_SIZE = int(os.getenv("GRPO_POPULATION_SIZE", "50"))
    GRPO_LEARNING_RATE = float(os.getenv("GRPO_LEARNING_RATE", "0.001"))
    GRPO_P_LEARNING_RATE = float(os.getenv("GRPO_P_LEARNING_RATE", "0.001"))
    ARBITRATION_LEARNING_RATE = float(os.getenv("ARBITRATION_LEARNING_RATE", "0.01"))
    
    # Processing Settings
    NEWS_FETCH_INTERVAL_HOURS = int(os.getenv("NEWS_FETCH_INTERVAL_HOURS", "1"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "512"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/bi_platform.log")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required_keys = ["NEWSAPI_KEY"]
        missing_keys = [key for key in required_keys if not getattr(cls, key)]
        
        if missing_keys:
            raise ValueError(f"Missing required configuration: {missing_keys}")
        
        return True

# Business sectors for filtering
BUSINESS_SECTORS = [
    "technology", "healthcare", "finance", "energy", "automotive",
    "real-estate", "agriculture", "manufacturing", "retail", 
    "telecommunications", "aerospace", "biotechnology"
]

# Investment risk levels
RISK_LEVELS = ["low", "moderate", "high"]

# User roles
USER_ROLES = ["entrepreneur", "investor", "business_owner", "analyst"]

# Geographic regions
REGIONS = [
    "india", "mumbai", "delhi", "bangalore", "chennai", "hyderabad",
    "pune", "kolkata", "asia", "global"
]
