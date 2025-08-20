#!/usr/bin/env python3
"""
Debug script to test News API and examine DataFrame structure
"""
import sys
import os
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.news_collector import NewsCollector

def test_news_api():
    """Test the News API and examine the data structure"""
    
    print("🔍 Testing News API...")
    print("=" * 50)
    
    try:
        # Initialize NewsCollector
        from config.settings import Config
        api_key = Config.NEWSAPI_KEY
        print(f"📡 API Key: {api_key[:10]}..." if api_key else "⚠️  No API key found")
        
        collector = NewsCollector(api_key=api_key)
        print("✅ NewsCollector initialized successfully")
        
        # Test 1: Fetch general business news
        print("\n📰 Test 1: Fetching general business news...")
        news_data = collector.fetch_business_news(days_back=3)
        
        if news_data is None:
            print("❌ news_data is None")
            return
        elif isinstance(news_data, pd.DataFrame):
            print(f"✅ Got DataFrame with {len(news_data)} rows")
            print(f"📊 DataFrame shape: {news_data.shape}")
            print(f"📋 Column names: {list(news_data.columns)}")
            
            # Show first few rows
            print("\n🔍 First 3 rows:")
            print(news_data.head(3))
            
            # Check for specific columns we need
            required_columns = ['title', 'content', 'source_name', 'published_at', 'business_sentiment']
            print(f"\n✅ Required columns check:")
            for col in required_columns:
                exists = col in news_data.columns
                print(f"   {col}: {'✅' if exists else '❌'}")
            
            # Check data types
            print(f"\n📋 Data types:")
            print(news_data.dtypes)
            
            # Check for empty values
            print(f"\n📊 Null values:")
            print(news_data.isnull().sum())
            
        else:
            print(f"⚠️  Unexpected return type: {type(news_data)}")
            print(f"📄 Content: {news_data}")
        
        # Test 2: Fetch sector-specific news
        print("\n📰 Test 2: Fetching finance sector news...")
        finance_news = collector.fetch_business_news(days_back=3, sectors=["finance"])
        
        if finance_news is not None and isinstance(finance_news, pd.DataFrame):
            print(f"✅ Got finance news: {len(finance_news)} articles")
            if not finance_news.empty:
                print("📝 Sample finance article:")
                sample = finance_news.iloc[0]
                print(f"   Title: {sample.get('title', 'N/A')}")
                print(f"   Source: {sample.get('source_name', 'N/A')}")
                print(f"   Published: {sample.get('published_at', 'N/A')}")
                if 'business_sentiment' in finance_news.columns:
                    print(f"   Sentiment: {sample.get('business_sentiment', 'N/A')}")
        else:
            print("❌ Failed to get finance news")
        
        # Test 3: Check if business_sentiment exists and what it contains
        if news_data is not None and isinstance(news_data, pd.DataFrame) and 'business_sentiment' in news_data.columns:
            print(f"\n📊 Business sentiment analysis:")
            sentiment_col = news_data['business_sentiment']
            print(f"   Type: {sentiment_col.dtype}")
            print(f"   Min: {sentiment_col.min()}")
            print(f"   Max: {sentiment_col.max()}")
            print(f"   Mean: {sentiment_col.mean():.3f}")
            print(f"   Sample values: {sentiment_col.head().tolist()}")
        
        return news_data
        
    except Exception as e:
        print(f"❌ Error testing News API: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_text_processor(news_data):
    """Test the text processor with actual news data"""
    
    if news_data is None or news_data.empty:
        print("⚠️  No news data to process")
        return
    
    print("\n🔧 Testing Text Processor...")
    print("=" * 50)
    
    try:
        from src.processing.text_processor import TextProcessor
        processor = TextProcessor()
        print("✅ TextProcessor initialized")
        
        # Process the news dataframe
        processed_df = processor.process_news_dataframe(news_data)
        
        if processed_df is not None and isinstance(processed_df, pd.DataFrame):
            print(f"✅ Processed DataFrame: {len(processed_df)} rows")
            print(f"📋 Processed columns: {list(processed_df.columns)}")
            
            # Check what the processor added/changed
            print("\n📊 Processing results:")
            if 'primary_sector' in processed_df.columns:
                print(f"   Primary sectors: {processed_df['primary_sector'].unique()}")
            if 'business_sentiment' in processed_df.columns:
                print(f"   Sentiment range: {processed_df['business_sentiment'].min():.3f} to {processed_df['business_sentiment'].max():.3f}")
            if 'top_keywords' in processed_df.columns:
                print(f"   Keywords extracted: Yes")
            
            return processed_df
        else:
            print("❌ Text processing failed")
            return None
            
    except Exception as e:
        print(f"❌ Error testing Text Processor: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_company_extraction():
    """Test company extraction from sample text"""
    
    print("\n🏢 Testing Company Extraction...")
    print("=" * 50)
    
    # Sample text with company names
    sample_texts = [
        "HDFC Bank announced new digital banking services in Mumbai. The bank reported strong quarterly results.",
        "Reliance Industries and Tata Group are investing in renewable energy projects across India.",
        "Infosys Technologies signed a major deal with a US client for cloud transformation services.",
        "Reserve Bank of India announced new guidelines for fintech companies operating in the country."
    ]
    
    from src.agents.hybrid_system import HybridRecommendationSystem
    
    # Create a dummy system to access the methods
    try:
        system = HybridRecommendationSystem()
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\n📝 Test {i}: {text[:60]}...")
            
            # Extract company names
            companies = system._extract_company_names(text)
            print(f"   Raw companies: {companies}")
            
            # Test company name validation
            valid_companies = [c for c in companies if system._looks_like_company_name(c)]
            print(f"   Valid companies: {valid_companies}")
            
            # Test sector relevance
            for company in valid_companies:
                finance_relevant = system._is_company_relevant_to_sector(company, "finance", text, "")
                tech_relevant = system._is_company_relevant_to_sector(company, "technology", text, "")
                print(f"   {company}: Finance={finance_relevant}, Tech={tech_relevant}")
        
    except Exception as e:
        print(f"❌ Error testing company extraction: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debug function"""
    
    print("🚀 News API and DataFrame Structure Debug")
    print("=" * 60)
    
    # Test 1: News API
    news_data = test_news_api()
    
    # Test 2: Text Processor
    if news_data is not None:
        processed_data = test_text_processor(news_data)
    
    # Test 3: Company Extraction
    test_company_extraction()
    
    print("\n" + "=" * 60)
    print("🏁 Debug complete!")

if __name__ == "__main__":
    main() 