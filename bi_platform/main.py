"""
Main entry point for BI Platform
This is where you start the system
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    """Main function to run BI Platform"""
    
    print("🚀 Starting BI Platform - Hybrid GRPO Framework")
    print("=" * 60)
    
    try:
        # Import our modules
        from data.news_collector import NewsCollector
        from config.settings import Config
        
        # Validate configuration
        print("⚙️  Checking configuration...")
        try:
            Config.validate()
            print("✅ Configuration valid")
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            print("\n🔧 Please check your config/config.env file")
            return
        
        # Initialize components
        print("📡 Initializing News Collector...")
        news_collector = NewsCollector(api_key=Config.NEWSAPI_KEY)
        
        # Fetch some news
        print("📰 Fetching latest business news...")
        news_df = news_collector.fetch_business_news(
            days_back=1,
            sectors=["technology", "finance", "energy"]
        )
        
        print(f"✅ Collected {len(news_df)} news articles")
        
        if len(news_df) > 0:
            # Show summary
            print(f"\n📊 News Summary:")
            print(f"Sectors: {news_df['primary_sector'].value_counts().to_dict()}")
            
            # Get trending topics
            trending = news_collector.get_trending_topics(news_df, top_n=5)
            print(f"🔥 Trending: {trending}")
            
            # Basic analysis
            recent_articles = len(news_df[news_df['published_datetime'] > 
                                       (news_df['published_datetime'].max() - 
                                        pd.Timedelta(hours=6))])
            print(f"📈 Recent activity: {recent_articles} articles in last 6 hours")
        
        print("\n🎉 BI Platform running successfully!")
        print("Next steps: Implement GRPO agents and recommendation engine")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed all dependencies:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
