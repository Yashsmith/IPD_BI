"""
News data collector using NewsAPI
Fetches business and financial news for BI analysis
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsCollector:
    """
    Collects news data from NewsAPI for business intelligence analysis
    """
    
    def __init__(self, api_key: str, base_url: str = "https://newsapi.org/v2"):
        """
        Initialize News Collector
        
        Args:
            api_key: NewsAPI key
            base_url: NewsAPI base URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        
        # Business and financial keywords for filtering
        self.business_keywords = [
            "investment", "startup", "funding", "IPO", "merger", "acquisition",
            "market", "stock", "economic", "business", "financial", "revenue",
            "profit", "loss", "growth", "decline", "expansion", "partnership"
        ]
        
        # Sector-specific keywords
        self.sector_keywords = {
            "technology": ["AI", "tech", "software", "digital", "innovation", "semiconductor"],
            "healthcare": ["pharma", "medical", "health", "biotech", "drug", "treatment"],
            "energy": ["oil", "gas", "renewable", "solar", "wind", "energy", "power"],
            "finance": ["bank", "fintech", "payment", "loan", "credit", "insurance"],
            "retail": ["ecommerce", "retail", "consumer", "shopping", "marketplace"],
            "automotive": ["auto", "car", "vehicle", "electric vehicle", "EV", "mobility"]
        }
    
    def fetch_everything_news(self, 
                            query: str = None,
                            sources: str = None,
                            domains: str = None,
                            from_date: str = None,
                            to_date: str = None,
                            language: str = "en",
                            sort_by: str = "publishedAt",
                            page_size: int = 100) -> Dict:
        """
        Fetch news using everything endpoint
        
        Args:
            query: Keywords or phrases to search for
            sources: Comma-separated string of news sources
            domains: Comma-separated string of domains
            from_date: Date string (YYYY-MM-DD)
            to_date: Date string (YYYY-MM-DD)  
            language: Language code
            sort_by: Sort order (publishedAt, relevancy, popularity)
            page_size: Number of articles per page (max 100)
        
        Returns:
            JSON response from NewsAPI
        """
        
        url = f"{self.base_url}/everything"
        
        params = {
            "apiKey": self.api_key,
            "language": language,
            "sortBy": sort_by,
            "pageSize": page_size
        }
        
        # Add optional parameters
        if query:
            params["q"] = query
        if sources:
            params["sources"] = sources
        if domains:
            params["domains"] = domains
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
            
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news: {e}")
            return {"status": "error", "message": str(e)}
    
    def fetch_business_news(self, 
                           days_back: int = 1,
                           sectors: List[str] = None,
                           regions: List[str] = None) -> pd.DataFrame:
        """
        Fetch business and financial news for the last N days
        
        Args:
            days_back: Number of days back to fetch news
            sectors: List of sectors to focus on
            regions: List of regions to include
            
        Returns:
            DataFrame with processed news articles
        """
        
        # Calculate date range
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        all_articles = []
        
        # General business news query
        business_query = " OR ".join(self.business_keywords[:10])  # Limit to avoid long URLs
        
        logger.info(f"Fetching business news from {from_date} to {to_date}")
        
        # Fetch general business news
        response = self.fetch_everything_news(
            query=business_query,
            from_date=from_date,
            to_date=to_date,
            sort_by="publishedAt"
        )
        
        if response.get("status") == "ok":
            all_articles.extend(response.get("articles", []))
            logger.info(f"Fetched {len(response.get('articles', []))} general business articles")
        
        # Fetch sector-specific news if specified
        if sectors:
            for sector in sectors:
                if sector in self.sector_keywords:
                    sector_query = " OR ".join(self.sector_keywords[sector][:5])
                    
                    logger.info(f"Fetching {sector} sector news")
                    
                    response = self.fetch_everything_news(
                        query=sector_query,
                        from_date=from_date,
                        to_date=to_date,
                        sort_by="publishedAt",
                        page_size=50  # Smaller page size for sector-specific
                    )
                    
                    if response.get("status") == "ok":
                        all_articles.extend(response.get("articles", []))
                        logger.info(f"Fetched {len(response.get('articles', []))} {sector} articles")
                    
                    # Rate limiting - NewsAPI allows 1000 requests per day for free tier
                    time.sleep(0.5)
        
        # Convert to DataFrame
        df = self._process_articles_to_dataframe(all_articles)
        
        # Add sector classification
        df = self._classify_articles_by_sector(df)
        
        logger.info(f"Total articles processed: {len(df)}")
        return df
    
    def _process_articles_to_dataframe(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Process raw articles into structured DataFrame
        
        Args:
            articles: List of article dictionaries from NewsAPI
            
        Returns:
            Processed DataFrame
        """
        
        processed_articles = []
        
        for article in articles:
            # Extract and clean article data
            processed_article = {
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "content": article.get("content", ""),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
                "source_name": article.get("source", {}).get("name", ""),
                "source_id": article.get("source", {}).get("id", ""),
                "author": article.get("author", ""),
                "url_to_image": article.get("urlToImage", "")
            }
            
            # Combine title, description, and content for full text
            full_text_parts = [
                processed_article["title"],
                processed_article["description"],
                processed_article["content"]
            ]
            processed_article["full_text"] = " ".join([part for part in full_text_parts if part])
            
            # Convert published_at to datetime
            try:
                processed_article["published_datetime"] = pd.to_datetime(processed_article["published_at"])
            except:
                processed_article["published_datetime"] = pd.NaT
            
            processed_articles.append(processed_article)
        
        df = pd.DataFrame(processed_articles)
        
        # Remove duplicates based on URL
        df = df.drop_duplicates(subset=["url"], keep="first")
        
        # Remove articles with minimal content
        df = df[df["full_text"].str.len() > 100]
        
        return df
    
    def _classify_articles_by_sector(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify articles by business sector based on keywords
        
        Args:
            df: DataFrame with news articles
            
        Returns:
            DataFrame with sector classifications
        """
        
        def classify_sector(text: str) -> List[str]:
            """Classify text into business sectors"""
            text_lower = text.lower()
            matched_sectors = []
            
            for sector, keywords in self.sector_keywords.items():
                if any(keyword.lower() in text_lower for keyword in keywords):
                    matched_sectors.append(sector)
            
            return matched_sectors if matched_sectors else ["general"]
        
        # Apply sector classification
        df["sectors"] = df["full_text"].apply(classify_sector)
        df["primary_sector"] = df["sectors"].apply(lambda x: x[0] if x else "general")
        
        return df
    
    def get_trending_topics(self, df: pd.DataFrame, top_n: int = 10) -> Dict[str, int]:
        """
        Extract trending topics from news articles
        
        Args:
            df: DataFrame with news articles
            top_n: Number of top topics to return
            
        Returns:
            Dictionary of trending topics and their counts
        """
        
        # Simple keyword frequency analysis
        all_text = " ".join(df["full_text"].fillna(""))
        
        # Remove common words and extract business-relevant terms
        import re
        from collections import Counter
        
        # Extract words (basic approach - could be enhanced with NLP)
        words = re.findall(r'\b[A-Za-z]{3,}\b', all_text.lower())
        
        # Filter for business-relevant terms
        business_terms = [word for word in words if word in 
                         [kw.lower() for kw_list in self.sector_keywords.values() for kw in kw_list] + 
                         [kw.lower() for kw in self.business_keywords]]
        
        # Count frequency
        term_counts = Counter(business_terms)
        
        return dict(term_counts.most_common(top_n))

    def save_news_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save news data to CSV file
        
        Args:
            df: DataFrame to save
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Saved filename
        """
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"news_data_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(df)} articles to {filename}")
        
        return filename

# Example usage and testing
if __name__ == "__main__":
    # This is for testing - you'll need to provide your actual NewsAPI key
    
    # Example usage (commented out - requires API key)
    """
    # Initialize collector
    collector = NewsCollector(api_key="your_newsapi_key_here")
    
    # Fetch business news for last 2 days, focusing on technology and energy
    news_df = collector.fetch_business_news(
        days_back=2,
        sectors=["technology", "energy"]
    )
    
    # Print summary
    print(f"Fetched {len(news_df)} articles")
    print(f"Sectors covered: {news_df['primary_sector'].value_counts().to_dict()}")
    
    # Get trending topics
    trending = collector.get_trending_topics(news_df)
    print(f"Trending topics: {trending}")
    
    # Save data
    filename = collector.save_news_data(news_df)
    print(f"Data saved to: {filename}")
    """
    
    print("NewsCollector class created successfully!")
    print("To use it, provide your NewsAPI key and call fetch_business_news()")
