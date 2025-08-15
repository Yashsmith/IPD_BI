"""
Integrated demo showing the complete BI Platform pipeline
From news collection to recommendations
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def run_demo():
    """Run complete demo of BI Platform capabilities"""
    
    print("🚀 BI PLATFORM COMPLETE DEMO")
    print("=" * 60)
    print("Demonstrating: News Collection → Text Processing → Recommendations")
    print()
    
    try:
        # Import all our modules
        from data.news_collector import NewsCollector
        from processing.text_processor import TextProcessor
        from processing.recommendation_engine import SimpleRecommendationEngine, UserProfile
        from config.settings import Config
        
        # Validate configuration
        print("⚙️  Step 1: Validating Configuration")
        try:
            Config.validate()
            print("✅ Configuration is valid")
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            print("Please set your NEWSAPI_KEY in config/config.env")
            return
        
        # Initialize components
        print("\n📡 Step 2: Initializing Components")
        news_collector = NewsCollector(api_key=Config.NEWSAPI_KEY)
        text_processor = TextProcessor()
        recommendation_engine = SimpleRecommendationEngine()
        print("✅ All components initialized")
        
        # Collect news data
        print("\n📰 Step 3: Collecting Business News")
        news_df = news_collector.fetch_business_news(
            days_back=1,
            sectors=["technology", "finance", "energy"]
        )
        print(f"✅ Collected {len(news_df)} news articles")
        
        if len(news_df) == 0:
            print("❌ No news articles collected. Check your API key and internet connection.")
            return
        
        # Process text
        print("\n🔍 Step 4: Processing Text and Sentiment Analysis")
        processed_df = text_processor.process_news_dataframe(news_df)
        print("✅ Text processing completed")
        
        # Show sentiment summary
        sentiment_summary = text_processor.get_sentiment_summary(processed_df)
        print(f"\n📊 Sentiment Summary:")
        print(f"Average sentiment: {sentiment_summary['avg_polarity']:.2f}")
        print(f"Business sentiment: {sentiment_summary['avg_business_sentiment']:.2f}")
        print(f"Sentiment distribution: {sentiment_summary['sentiment_distribution']}")
        
        # Detect trends
        trends = text_processor.detect_trends(processed_df)
        print(f"\n🔥 Trending Topics: {dict(list(trends.items())[:5])}")
        
        # Create sample user profiles for different user types
        users = [
            UserProfile(
                role="entrepreneur",
                sectors=["technology", "finance"],
                location="mumbai",
                capital_range="medium",
                risk_appetite="moderate",
                experience_level="intermediate"
            ),
            UserProfile(
                role="investor",
                sectors=["technology", "energy"],
                location="delhi",
                capital_range="high",
                risk_appetite="high",
                experience_level="expert"
            ),
            UserProfile(
                role="business_owner",
                sectors=["finance", "energy"],
                location="bangalore",
                capital_range="medium",
                risk_appetite="low",
                experience_level="intermediate"
            )
        ]
        
        # Generate recommendations for each user type
        print("\n🎯 Step 5: Generating Personalized Recommendations")
        print("=" * 60)
        
        for i, user in enumerate(users, 1):
            print(f"\n👤 User {i}: {user.role.title()} in {user.location.title()}")
            print(f"Interests: {', '.join(user.sectors)} | Risk: {user.risk_appetite}")
            print("-" * 50)
            
            recommendations = recommendation_engine.generate_recommendations(
                user_profile=user,
                news_df=processed_df,
                max_recommendations=3
            )
            
            if recommendations:
                for j, rec in enumerate(recommendations, 1):
                    print(f"\n{j}. {rec.title}")
                    print(f"   Confidence: {rec.confidence_score:.1%} | Risk: {rec.risk_level}")
                    print(f"   Sentiment: {rec.market_sentiment:+.2f} | Sector: {rec.sector}")
                    print(f"   Reasoning: {rec.reasoning[0]}")
                    if rec.supporting_articles:
                        print(f"   Source: {rec.supporting_articles[0][:80]}...")
            else:
                print("   No recommendations generated for this profile")
        
        # Market analysis summary
        print(f"\n📈 Step 6: Market Analysis Summary")
        print("=" * 50)
        
        sector_sentiment = recommendation_engine.analyze_market_sentiment(processed_df)
        print("Sector Sentiment Analysis:")
        for sector, sentiment in sector_sentiment.items():
            sentiment_label = "🟢 Positive" if sentiment > 0.1 else "🔴 Negative" if sentiment < -0.1 else "🟡 Neutral"
            print(f"  {sector.title()}: {sentiment:+.2f} {sentiment_label}")
        
        # Show most impactful articles
        print(f"\n📑 Most Impactful Articles:")
        top_articles = processed_df.nlargest(3, 'business_sentiment')[['title', 'business_sentiment', 'primary_sector']]
        for idx, row in top_articles.iterrows():
            print(f"  • {row['title'][:70]}... [{row['primary_sector']}] (+{row['business_sentiment']:.2f})")
        
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ All pipeline components working:")
        print("  📡 News Collection ← NewsAPI")
        print("  🔍 Text Processing ← Sentiment Analysis + Keywords")
        print("  🎯 Recommendations ← Rule-based Engine")
        print("  👥 Multi-user Support ← Different profiles and preferences")
        
        print(f"\nNext Steps:")
        print("🤖 Implement GRPO agents for group consensus")
        print("🎨 Implement GRPO-P agents for personalization")
        print("🧠 Add learned arbitration controller")
        print("🌐 Create web interface")
        
        # Save demo results
        filename = news_collector.save_news_data(processed_df, "demo_processed_news.csv")
        print(f"\n💾 Demo data saved to: {filename}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed all dependencies:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

def show_quick_start():
    """Show quick start instructions"""
    print("\n🚀 QUICK START GUIDE")
    print("=" * 40)
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Get NewsAPI key:")
    print("   - Visit https://newsapi.org/")
    print("   - Sign up (free)")
    print("   - Copy your API key")
    print("\n3. Configure:")
    print("   - Edit config/config.env")
    print("   - Set NEWSAPI_KEY=your_key")
    print("\n4. Run demo:")
    print("   python demo.py")

if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"Failed to run demo: {e}")
        show_quick_start()
