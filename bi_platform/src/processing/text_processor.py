"""
Text processing pipeline for news articles
Includes sentiment analysis, keyword extraction, and trend detection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter
import logging

# For sentiment analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Install with: pip install textblob")

# For advanced NLP (optional)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    
    # Download required NLTK data (run once)
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Install with: pip install nltk")

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Processes news articles for business intelligence analysis
    """
    
    def __init__(self):
        """Initialize text processor"""
        
        # Business sentiment keywords
        self.positive_business_words = [
            'growth', 'profit', 'gain', 'increase', 'rise', 'surge', 'boom', 
            'expansion', 'success', 'breakthrough', 'opportunity', 'bullish',
            'strong', 'robust', 'healthy', 'optimistic', 'recovery', 'upgrade'
        ]
        
        self.negative_business_words = [
            'loss', 'decline', 'fall', 'drop', 'crash', 'recession', 'crisis',
            'risk', 'threat', 'concern', 'weakness', 'bearish', 'volatile',
            'uncertainty', 'downturn', 'slowdown', 'bankruptcy', 'failure'
        ]
        
        # Market trend indicators
        self.bullish_indicators = [
            'all-time high', 'record high', 'new high', 'breakout', 'rally',
            'bull market', 'uptrend', 'momentum', 'breakthrough'
        ]
        
        self.bearish_indicators = [
            'all-time low', 'record low', 'new low', 'breakdown', 'selloff',
            'bear market', 'downtrend', 'correction', 'crash'
        ]
        
        # Initialize NLTK components if available
        if NLTK_AVAILABLE:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'])
            self.lemmatizer = None
    
    def analyze_sentiment(self, text: str, method: str = "textblob") -> Dict[str, float]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            method: Sentiment analysis method ('textblob', 'business_keywords')
            
        Returns:
            Dictionary with sentiment scores
        """
        
        if not text or not isinstance(text, str):
            return {"polarity": 0.0, "subjectivity": 0.0, "business_sentiment": 0.0}
        
        sentiment_scores = {}
        
        # TextBlob sentiment (general sentiment)
        if method == "textblob" and TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            sentiment_scores["polarity"] = blob.sentiment.polarity  # -1 to 1
            sentiment_scores["subjectivity"] = blob.sentiment.subjectivity  # 0 to 1
        else:
            # Fallback basic sentiment
            sentiment_scores["polarity"] = 0.0
            sentiment_scores["subjectivity"] = 0.5
        
        # Business-specific sentiment
        sentiment_scores["business_sentiment"] = self._calculate_business_sentiment(text)
        
        # Market trend sentiment  
        sentiment_scores["market_trend"] = self._calculate_market_trend(text)
        
        return sentiment_scores
    
    def _calculate_business_sentiment(self, text: str) -> float:
        """
        Calculate business-specific sentiment based on financial keywords
        
        Args:
            text: Text to analyze
            
        Returns:
            Business sentiment score (-1 to 1)
        """
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_business_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_business_words if word in text_lower)
        
        total_count = positive_count + negative_count
        
        if total_count == 0:
            return 0.0
        
        # Calculate weighted sentiment
        sentiment = (positive_count - negative_count) / total_count
        
        return max(-1.0, min(1.0, sentiment))
    
    def _calculate_market_trend(self, text: str) -> float:
        """
        Calculate market trend sentiment (bullish vs bearish)
        
        Args:
            text: Text to analyze
            
        Returns:
            Market trend score (-1 to 1, where 1 is bullish, -1 is bearish)
        """
        
        text_lower = text.lower()
        
        bullish_score = sum(1 for indicator in self.bullish_indicators if indicator in text_lower)
        bearish_score = sum(1 for indicator in self.bearish_indicators if indicator in text_lower)
        
        total_score = bullish_score + bearish_score
        
        if total_score == 0:
            return 0.0
        
        trend = (bullish_score - bearish_score) / total_score
        return max(-1.0, min(1.0, trend))
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Extract important keywords from text
        
        Args:
            text: Text to process
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, frequency) tuples
        """
        
        if not text or not isinstance(text, str):
            return []
        
        # Clean and tokenize text
        cleaned_text = self._clean_text(text)
        
        if NLTK_AVAILABLE:
            tokens = word_tokenize(cleaned_text.lower())
        else:
            # Basic tokenization fallback
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', cleaned_text.lower())
        
        # Remove stop words and filter
        filtered_tokens = [
            token for token in tokens 
            if token not in self.stop_words 
            and len(token) > 2 
            and token.isalpha()
        ]
        
        # Lemmatize if available
        if self.lemmatizer:
            filtered_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        
        # Count frequency
        word_freq = Counter(filtered_tokens)
        
        return word_freq.most_common(top_n)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for processing
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
        
        return text.strip()
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract business entities from text (companies, locations, etc.)
        Basic implementation - could be enhanced with spaCy or other NLP libraries
        
        Args:
            text: Text to process
            
        Returns:
            Dictionary of entity types and their values
        """
        
        entities = {
            "companies": [],
            "locations": [],
            "currencies": [],
            "numbers": []
        }
        
        # Extract potential company names (capitalized words/phrases)
        company_pattern = r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b'
        companies = re.findall(company_pattern, text)
        entities["companies"] = list(set(companies))
        
        # Extract currencies
        currency_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\₹\d+(?:,\d{3})*(?:\.\d{2})?'
        currencies = re.findall(currency_pattern, text)
        entities["currencies"] = currencies
        
        # Extract numbers (potential financial figures)
        number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion|crore|lakh)\b'
        numbers = re.findall(number_pattern, text, re.IGNORECASE)
        entities["numbers"] = numbers
        
        # Extract locations (basic - just common business locations)
        business_locations = [
            'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune',
            'New York', 'London', 'Singapore', 'Hong Kong', 'Tokyo', 'San Francisco'
        ]
        
        for location in business_locations:
            if location in text:
                entities["locations"].append(location)
        
        return entities
    
    def process_news_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire news DataFrame with text analysis
        
        Args:
            df: DataFrame with news articles
            
        Returns:
            Enhanced DataFrame with text analysis results
        """
        
        logger.info(f"Processing {len(df)} articles for text analysis")
        
        # Apply sentiment analysis
        df["sentiment"] = df["full_text"].apply(lambda x: self.analyze_sentiment(x, "textblob"))
        
        # Extract individual sentiment components
        df["polarity"] = df["sentiment"].apply(lambda x: x.get("polarity", 0.0))
        df["subjectivity"] = df["sentiment"].apply(lambda x: x.get("subjectivity", 0.0))
        df["business_sentiment"] = df["sentiment"].apply(lambda x: x.get("business_sentiment", 0.0))
        df["market_trend"] = df["sentiment"].apply(lambda x: x.get("market_trend", 0.0))
        
        # Extract keywords
        df["keywords"] = df["full_text"].apply(lambda x: self.extract_keywords(x, top_n=5))
        df["top_keywords"] = df["keywords"].apply(lambda x: [kw[0] for kw in x])
        
        # Extract entities
        df["entities"] = df["full_text"].apply(self.extract_entities)
        
        # Calculate text statistics
        df["text_length"] = df["full_text"].apply(len)
        df["word_count"] = df["full_text"].apply(lambda x: len(x.split()) if x else 0)
        
        # Create sentiment categories
        df["sentiment_category"] = df["polarity"].apply(self._categorize_sentiment)
        df["business_sentiment_category"] = df["business_sentiment"].apply(self._categorize_sentiment)
        
        logger.info("Text processing completed")
        
        return df
    
    def _categorize_sentiment(self, score: float) -> str:
        """
        Categorize sentiment score into labels
        
        Args:
            score: Sentiment score (-1 to 1)
            
        Returns:
            Sentiment category string
        """
        
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get overall sentiment summary from processed DataFrame
        
        Args:
            df: Processed DataFrame with sentiment analysis
            
        Returns:
            Dictionary with sentiment summary statistics
        """
        
        summary = {
            "total_articles": len(df),
            "avg_polarity": df["polarity"].mean(),
            "avg_business_sentiment": df["business_sentiment"].mean(),
            "avg_market_trend": df["market_trend"].mean(),
            "sentiment_distribution": df["sentiment_category"].value_counts().to_dict(),
            "business_sentiment_distribution": df["business_sentiment_category"].value_counts().to_dict(),
            "most_positive_article": df.loc[df["polarity"].idxmax(), "title"] if len(df) > 0 else None,
            "most_negative_article": df.loc[df["polarity"].idxmin(), "title"] if len(df) > 0 else None
        }
        
        return summary
    
    def detect_trends(self, df: pd.DataFrame, min_frequency: int = 3) -> Dict[str, int]:
        """
        Detect trending topics from processed articles
        
        Args:
            df: Processed DataFrame
            min_frequency: Minimum frequency for trend detection
            
        Returns:
            Dictionary of trending topics and their frequencies
        """
        
        # Collect all keywords
        all_keywords = []
        for keywords_list in df["top_keywords"]:
            all_keywords.extend(keywords_list)
        
        # Count frequency
        keyword_counts = Counter(all_keywords)
        
        # Filter by minimum frequency
        trending_keywords = {
            keyword: count for keyword, count in keyword_counts.items()
            if count >= min_frequency
        }
        
        return dict(sorted(trending_keywords.items(), key=lambda x: x[1], reverse=True))

# Example usage and testing
if __name__ == "__main__":
    # Test the TextProcessor
    processor = TextProcessor()
    
    # Test text
    sample_text = """
    Apple Inc. reported strong quarterly earnings, with revenue growing 15% year-over-year.
    The company's stock surged 5% in after-hours trading as investors showed optimism
    about the tech giant's future prospects in artificial intelligence and renewable energy.
    CEO Tim Cook expressed confidence in the company's strategic direction.
    """
    
    print("🧪 Testing TextProcessor")
    print("=" * 40)
    
    # Test sentiment analysis
    sentiment = processor.analyze_sentiment(sample_text)
    print(f"Sentiment: {sentiment}")
    
    # Test keyword extraction
    keywords = processor.extract_keywords(sample_text)
    print(f"Keywords: {keywords}")
    
    # Test entity extraction
    entities = processor.extract_entities(sample_text)
    print(f"Entities: {entities}")
    
    print("\n✅ TextProcessor test completed!")
