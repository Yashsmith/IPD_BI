#!/usr/bin/env python3
"""
Test script for the truly dynamic recommendation system
"""
import sys
import os
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.hybrid_system import HybridRecommendationSystem, SystemUser

def test_dynamic_system():
    """Test the truly dynamic system with no hardcoded fallbacks"""
    
    print("🚀 Testing Truly Dynamic Recommendation System")
    print("=" * 60)
    print("❌ NO HARDCODED FALLBACKS - Everything must come from real news data")
    print("=" * 60)
    
    try:
        # Initialize system
        system = HybridRecommendationSystem()
        print("✅ System initialized")
        
        # Register test user
        test_user = SystemUser(
            user_id="test_dynamic",
            role="entrepreneur",
            sectors=["manufacturing"],
            location="bangalore",
            capital_range="high",
            risk_appetite="high",
            experience_level="intermediate"
        )
        
        user_id = system.register_user(test_user)
        print(f"✅ User registered: {user_id}")
        
        # Test company extraction from news
        print("\n🔍 Testing Dynamic Company Extraction...")
        companies = system._extract_companies_from_news("manufacturing", "bangalore")
        
        if companies:
            print(f"✅ Found {len(companies)} real companies from news:")
            for i, company in enumerate(companies[:3], 1):
                print(f"   {i}. {company.name} - {company.description[:60]}...")
                print(f"      Confidence: {company.confidence_score:.1%}, Risk: {company.risk_level}")
        else:
            print("❌ No companies extracted - dynamic pipeline needs debugging")
        
        # Test evidence generation
        print("\n📰 Testing Dynamic Evidence Generation...")
        evidence = system._generate_supporting_evidence("manufacturing", "invest")
        
        if evidence:
            print(f"✅ Found {len(evidence)} real evidence articles:")
            for i, article in enumerate(evidence[:2], 1):
                print(f"   {i}. {article.title[:80]}...")
                print(f"      Source: {article.source}, Sentiment: {article.sentiment:.2f}")
        else:
            print("❌ No evidence generated - dynamic pipeline needs debugging")
        
        # Test full recommendation generation
        print("\n🎯 Testing Full Dynamic Recommendation...")
        recommendations = system.get_enhanced_recommendations(user_id, max_recommendations=2)
        
        if recommendations:
            print(f"✅ Generated {len(recommendations)} dynamic recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec.sector.title()} - {rec.action.upper()}")
                print(f"   Strategy: {rec.strategy_type.value}")
                print(f"   Companies: {len(rec.target_companies)} real companies")
                print(f"   Evidence: {len(rec.supporting_articles)} real articles")
                print(f"   Reasoning: {rec.comprehensive_reasoning[:100]}...")
        else:
            print("❌ No recommendations generated")
        
        print("\n" + "=" * 60)
        print("🏁 Dynamic System Test Complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dynamic_system() 