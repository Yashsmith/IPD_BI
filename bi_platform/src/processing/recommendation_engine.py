"""
Simple rule-based recommendation engine
This serves as a baseline before implementing GRPO agents
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile for personalized recommendations"""
    role: str  # entrepreneur, investor, business_owner, analyst
    sectors: List[str]  # technology, healthcare, finance, etc.
    location: str  # mumbai, delhi, bangalore, etc.
    capital_range: str  # low, medium, high
    risk_appetite: str  # low, moderate, high
    experience_level: str  # beginner, intermediate, expert
    sustainability_focus: bool = False

@dataclass
class Recommendation:
    """Individual recommendation structure"""
    title: str
    description: str
    opportunity_type: str  # investment, business_idea, market_trend
    sector: str
    confidence_score: float  # 0 to 1
    risk_level: str  # low, moderate, high
    reasoning: List[str]  # Why this recommendation was made
    supporting_articles: List[str]  # URLs or titles of supporting news
    market_sentiment: float  # -1 to 1
    
class SimpleRecommendationEngine:
    """
    Rule-based recommendation engine that analyzes news sentiment 
    and generates business opportunities based on user profiles
    """
    
    def __init__(self):
        """Initialize recommendation engine"""
        
        # Define opportunity templates based on sectors and sentiment
        self.opportunity_templates = {
            "technology": {
                "positive": [
                    "Invest in AI and machine learning companies showing growth",
                    "Consider fintech startups with strong user adoption",
                    "Explore semiconductor companies benefiting from demand surge",
                    "Look into cybersecurity firms amid increasing digital threats"
                ],
                "negative": [
                    "Consider defensive tech positions in established companies",
                    "Look for undervalued tech stocks during market corrections",
                    "Explore cost-cutting solutions for businesses",
                    "Consider short-term volatility trading opportunities"
                ]
            },
            "energy": {
                "positive": [
                    "Invest in renewable energy companies with strong growth",
                    "Consider solar and wind power equipment manufacturers",
                    "Explore electric vehicle charging infrastructure",
                    "Look into battery technology companies"
                ],
                "negative": [
                    "Consider traditional energy companies at discounted prices",
                    "Look for energy efficiency service providers",
                    "Explore contrarian plays in oversold energy stocks",
                    "Consider energy storage solutions during price volatility"
                ]
            },
            "finance": {
                "positive": [
                    "Consider digital banking and payment platforms",
                    "Explore insurance technology companies",
                    "Look into wealth management technology",
                    "Consider blockchain and crypto infrastructure"
                ],
                "negative": [
                    "Look for value opportunities in traditional banking",
                    "Consider defensive financial positions",
                    "Explore fintech companies with strong fundamentals",
                    "Look into crisis-resistant financial services"
                ]
            },
            "healthcare": {
                "positive": [
                    "Invest in biotech companies with promising pipelines",
                    "Consider digital health and telemedicine platforms",
                    "Explore medical device innovations",
                    "Look into pharmaceutical companies with new approvals"
                ],
                "negative": [
                    "Consider defensive healthcare positions",
                    "Look for value in established pharmaceutical companies",
                    "Explore cost-effective healthcare solutions",
                    "Consider healthcare REITs during uncertainty"
                ]
            }
        }
        
        # Risk level mapping for different user profiles
        self.risk_mapping = {
            "entrepreneur": {"low": 0.3, "moderate": 0.6, "high": 0.9},
            "investor": {"low": 0.2, "moderate": 0.5, "high": 0.8},
            "business_owner": {"low": 0.4, "moderate": 0.7, "high": 0.8},
            "analyst": {"low": 0.1, "moderate": 0.4, "high": 0.6}
        }
    
    def analyze_market_sentiment(self, news_df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze overall market sentiment by sector
        
        Args:
            news_df: DataFrame with processed news articles
            
        Returns:
            Dictionary with sector sentiment scores
        """
        
        sector_sentiment = {}
        
        # Calculate sentiment by sector
        for sector in news_df['primary_sector'].unique():
            sector_articles = news_df[news_df['primary_sector'] == sector]
            
            if len(sector_articles) > 0:
                avg_sentiment = sector_articles['business_sentiment'].mean()
                avg_polarity = sector_articles['polarity'].mean()
                avg_market_trend = sector_articles['market_trend'].mean()
                
                # Combine different sentiment measures
                combined_sentiment = (avg_sentiment + avg_polarity + avg_market_trend) / 3
                sector_sentiment[sector] = combined_sentiment
            else:
                sector_sentiment[sector] = 0.0
        
        return sector_sentiment
    
    def extract_trending_opportunities(self, news_df: pd.DataFrame) -> List[Dict]:
        """
        Extract potential opportunities from trending news
        
        Args:
            news_df: DataFrame with processed news articles
            
        Returns:
            List of opportunity dictionaries
        """
        
        opportunities = []
        
        # Find articles with high positive sentiment
        positive_articles = news_df[news_df['business_sentiment'] > 0.3]
        
        # Group by sector and find trending topics
        for sector in positive_articles['primary_sector'].unique():
            sector_articles = positive_articles[positive_articles['primary_sector'] == sector]
            
            if len(sector_articles) >= 2:  # At least 2 articles for trend
                # Extract common keywords
                all_keywords = []
                for keywords_list in sector_articles['top_keywords']:
                    all_keywords.extend(keywords_list)
                
                if all_keywords:
                    # Find most common keywords
                    from collections import Counter
                    keyword_counts = Counter(all_keywords)
                    top_keywords = keyword_counts.most_common(3)
                    
                    opportunity = {
                        "sector": sector,
                        "trend": f"Growing interest in {', '.join([kw[0] for kw in top_keywords])}",
                        "sentiment_score": sector_articles['business_sentiment'].mean(),
                        "article_count": len(sector_articles),
                        "supporting_titles": sector_articles['title'].tolist()[:3]
                    }
                    opportunities.append(opportunity)
        
        return opportunities
    
    def generate_recommendations(self, 
                               user_profile: UserProfile, 
                               news_df: pd.DataFrame,
                               max_recommendations: int = 5) -> List[Recommendation]:
        """
        Generate personalized recommendations based on user profile and news analysis
        
        Args:
            user_profile: User's profile and preferences
            news_df: DataFrame with processed news articles
            max_recommendations: Maximum number of recommendations to generate
            
        Returns:
            List of personalized recommendations
        """
        
        recommendations = []
        
        # Analyze market sentiment
        sector_sentiment = self.analyze_market_sentiment(news_df)
        
        # Extract trending opportunities
        trending_opportunities = self.extract_trending_opportunities(news_df)
        
        # Generate recommendations for user's preferred sectors
        for sector in user_profile.sectors:
            if sector in sector_sentiment:
                sentiment_score = sector_sentiment[sector]
                
                # Determine sentiment direction
                sentiment_direction = "positive" if sentiment_score > 0.1 else "negative"
                
                # Get appropriate templates
                if sector in self.opportunity_templates:
                    templates = self.opportunity_templates[sector][sentiment_direction]
                    
                    for template in templates[:2]:  # Max 2 per sector
                        # Calculate confidence based on sentiment and user risk tolerance
                        base_confidence = abs(sentiment_score)
                        risk_factor = self.risk_mapping[user_profile.role][user_profile.risk_appetite]
                        confidence = min(0.95, base_confidence * risk_factor)
                        
                        # Determine risk level
                        if abs(sentiment_score) > 0.5:
                            risk_level = "high" if sentiment_score > 0 else "moderate"
                        elif abs(sentiment_score) > 0.2:
                            risk_level = "moderate"
                        else:
                            risk_level = "low"
                        
                        # Create reasoning
                        reasoning = [
                            f"Market sentiment for {sector} is {sentiment_direction} ({sentiment_score:.2f})",
                            f"Matches your risk appetite ({user_profile.risk_appetite})",
                            f"Aligns with your role as {user_profile.role}"
                        ]
                        
                        # Find supporting articles
                        sector_articles = news_df[news_df['primary_sector'] == sector]
                        supporting_articles = sector_articles['title'].tolist()[:2]
                        
                        recommendation = Recommendation(
                            title=f"{sector.title()} Opportunity: {template}",
                            description=self._generate_detailed_description(template, sector, sentiment_score),
                            opportunity_type=self._determine_opportunity_type(user_profile.role),
                            sector=sector,
                            confidence_score=confidence,
                            risk_level=risk_level,
                            reasoning=reasoning,
                            supporting_articles=supporting_articles,
                            market_sentiment=sentiment_score
                        )
                        
                        recommendations.append(recommendation)
        
        # Add recommendations based on trending opportunities
        for opportunity in trending_opportunities[:2]:
            if opportunity["sector"] in user_profile.sectors:
                recommendation = Recommendation(
                    title=f"Trending: {opportunity['trend']}",
                    description=f"Emerging trend in {opportunity['sector']} based on {opportunity['article_count']} recent articles",
                    opportunity_type="market_trend",
                    sector=opportunity["sector"],
                    confidence_score=min(0.8, opportunity["sentiment_score"]),
                    risk_level="moderate",
                    reasoning=[
                        f"Based on {opportunity['article_count']} recent articles",
                        f"Positive sentiment score: {opportunity['sentiment_score']:.2f}",
                        "Emerging market trend identified"
                    ],
                    supporting_articles=opportunity["supporting_titles"],
                    market_sentiment=opportunity["sentiment_score"]
                )
                recommendations.append(recommendation)
        
        # Sort by confidence score and return top recommendations
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return recommendations[:max_recommendations]
    
    def _generate_detailed_description(self, template: str, sector: str, sentiment: float) -> str:
        """Generate detailed description for recommendation"""
        
        sentiment_desc = "strong positive" if sentiment > 0.5 else "positive" if sentiment > 0 else "negative"
        
        return f"""
        Based on {sentiment_desc} market sentiment in the {sector} sector, this opportunity 
        presents a potential avenue for growth. Current market conditions suggest favorable 
        dynamics for this type of investment or business initiative.
        
        Key factors:
        - Market sentiment: {sentiment:.2f} (scale: -1 to 1)
        - Sector trend: {sentiment_desc}
        - Based on recent news analysis and market indicators
        """.strip()
    
    def _determine_opportunity_type(self, user_role: str) -> str:
        """Determine opportunity type based on user role"""
        
        role_mapping = {
            "entrepreneur": "business_idea",
            "investor": "investment",
            "business_owner": "business_expansion",
            "analyst": "market_analysis"
        }
        
        return role_mapping.get(user_role, "investment")
    
    def generate_recommendation_report(self, 
                                     user_profile: UserProfile,
                                     recommendations: List[Recommendation]) -> str:
        """
        Generate a formatted recommendation report
        
        Args:
            user_profile: User profile
            recommendations: List of recommendations
            
        Returns:
            Formatted report string
        """
        
        report = f"""
        📊 BUSINESS INTELLIGENCE RECOMMENDATIONS
        {'='*50}
        
        👤 User Profile:
        Role: {user_profile.role.title()}
        Sectors: {', '.join(user_profile.sectors)}
        Location: {user_profile.location.title()}
        Risk Appetite: {user_profile.risk_appetite.title()}
        
        🎯 Top Recommendations:
        """
        
        for i, rec in enumerate(recommendations, 1):
            report += f"""
        
        {i}. {rec.title}
        {'─'*40}
        Opportunity Type: {rec.opportunity_type.replace('_', ' ').title()}
        Sector: {rec.sector.title()}
        Confidence: {rec.confidence_score:.1%}
        Risk Level: {rec.risk_level.title()}
        Market Sentiment: {rec.market_sentiment:+.2f}
        
        Description:
        {rec.description}
        
        Reasoning:
        {chr(10).join(f'• {reason}' for reason in rec.reasoning)}
        
        Supporting Evidence:
        {chr(10).join(f'• {article}' for article in rec.supporting_articles[:2])}
        """
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Test the recommendation engine
    engine = SimpleRecommendationEngine()
    
    # Create sample user profile
    user = UserProfile(
        role="entrepreneur",
        sectors=["technology", "finance"],
        location="mumbai",
        capital_range="medium",
        risk_appetite="moderate",
        experience_level="intermediate"
    )
    
    print("🧪 Testing SimpleRecommendationEngine")
    print("=" * 50)
    print(f"User Profile: {user.role} interested in {user.sectors}")
    
    # Would need actual news data to test fully
    print("✅ SimpleRecommendationEngine created successfully!")
    print("Ready to integrate with news data and text processing!")
