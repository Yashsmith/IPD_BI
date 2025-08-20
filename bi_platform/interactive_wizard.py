#!/usr/bin/env python3
"""
Menu-driven interactive wizard for GRPO–GRPO-P Hybrid Recommendation System
Run: python interactive_wizard.py
"""
import sys
import os
from typing import List

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.hybrid_system import HybridRecommendationSystem, SystemUser

try:
    from config.settings import USER_ROLES, RISK_LEVELS, REGIONS
except Exception:
    USER_ROLES = ["entrepreneur", "investor", "business_owner", "analyst"]
    RISK_LEVELS = ["low", "moderate", "high"]
    REGIONS = ["india", "mumbai", "delhi", "bangalore", "pune", "hyderabad", "global"]

ALLOWED_SECTORS = [
    "technology", "finance", "energy", "healthcare", "retail", "manufacturing", "real_estate"
]
CAPITAL_RANGES = ["low", "medium", "high"]
EXPERIENCE_LEVELS = ["beginner", "intermediate", "expert"]


def prompt_text(prompt: str) -> str:
    while True:
        value = input(f"{prompt}: ").strip()
        if value:
            return value
        print("Please enter a value.")


def select_option(prompt: str, options: List[str]) -> str:
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        choice = input("Select an option (1-{}): ".format(len(options))).strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("Invalid selection. Please try again.")


def select_multi(prompt: str, options: List[str], min_select: int = 1) -> List[str]:
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print("Enter one or more numbers separated by commas (e.g., 1,3,5)")
    while True:
        raw = input("Select: ").strip()
        try:
            indices = [int(x) for x in raw.split(',') if x.strip().isdigit()]
            indices = sorted(set(idx for idx in indices if 1 <= idx <= len(options)))
            if len(indices) >= min_select:
                return [options[i - 1] for i in indices]
        except Exception:
            pass
        print("Invalid selection. Please try again.")


def format_enhanced_recommendation(rec, index: int) -> str:
    """Format enhanced recommendation with comprehensive details"""
    
    print(f"\n{'='*80}")
    print(f"🎯 RECOMMENDATION {index + 1}: {rec.action.upper()} in {rec.sector.upper()}")
    print(f"{'='*80}")
    
    # Core recommendation
    print(f"📊 CONFIDENCE: {rec.confidence:.1%} | RISK: {rec.risk_level.upper()}")
    print(f"🏗️  STRATEGY: {rec.strategy_type.value.replace('_', ' ').title()}")
    print(f"💰 INVESTMENT: {rec.investment_amount} | TIMELINE: {rec.time_horizon}")
    print(f"🚀 ENTRY: {rec.entry_strategy}")
    print(f"✨ UNIQUE ANGLE: {rec.unique_angle}")
    
    # Target companies
    print(f"\n🏢 TARGET COMPANIES:")
    for company in rec.target_companies:
        print(f"   • {company.name} ({company.location})")
        print(f"     {company.description}")
        print(f"     Confidence: {company.confidence_score:.1%} | Risk: {company.risk_level}")
    
    # Market analysis
    print(f"\n📈 MARKET ANALYSIS:")
    print(f"   Sentiment: {rec.market_analysis.sector_sentiment:+.2f} ({rec.market_analysis.market_trend})")
    print(f"   Volatility: {rec.market_analysis.volatility_level}")
    print(f"   Key Drivers: {', '.join(rec.market_analysis.key_drivers[:3])}")
    
    # Policy impact
    if rec.policy_impact:
        print(f"\n🏛️  POLICY IMPACT:")
        print(f"   {rec.policy_impact.policy_type}: {rec.policy_impact.description}")
        print(f"   Impact: {rec.policy_impact.impact_level} | Timeline: {rec.policy_impact.timeline}")
    
    # Supporting evidence
    print(f"\n📰 SUPPORTING EVIDENCE:")
    for i, article in enumerate(rec.supporting_articles[:2], 1):
        print(f"   {i}. {article.title}")
        print(f"      Source: {article.source} | Sentiment: {article.sentiment:+.2f}")
        print(f"      Key Points: {', '.join(article.key_points[:2])}")
    
    # Risk and opportunity factors
    print(f"\n⚠️  RISK FACTORS:")
    for risk in rec.risk_factors[:3]:
        print(f"   • {risk}")
    
    print(f"\n💡 OPPORTUNITY FACTORS:")
    for opp in rec.opportunity_factors[:3]:
        print(f"   • {opp}")
    
    # Comprehensive reasoning
    print(f"\n🧠 COMPREHENSIVE REASONING:")
    print(f"   {rec.comprehensive_reasoning}")
    
    # What makes this different
    print(f"\n🔍 WHAT MAKES THIS DIFFERENT:")
    print(f"   {rec.what_makes_this_different}")
    print(f"   Competitive Advantage: {rec.competitive_advantage}")
    print(f"   Market Timing: {rec.market_timing}")
    
    # Blending information
    print(f"\n⚖️  BLENDING STRATEGY:")
    print(f"   GRPO Weight: {rec.grpo_weight:.1%} | GRPO-P Weight: {rec.grpo_p_weight:.1%}")
    print(f"   Strategy: {rec.blending_strategy.replace('_', ' ').title()}")
    
    return f"Enhanced recommendation {index + 1} displayed"


def main():
    print("\n🚀 GRPO–GRPO-P Interactive Wizard")
    print("=" * 40)

    user_id = prompt_text("Enter a unique user id")
    role = select_option("Select your role", USER_ROLES)
    sectors = select_multi("Select your preferred sectors", ALLOWED_SECTORS, min_select=1)
    location = select_option("Select your location/region", REGIONS)
    capital_range = select_option("Select your capital range", CAPITAL_RANGES)
    risk = select_option("Select your risk appetite", RISK_LEVELS)
    experience = select_option("Select your experience level", EXPERIENCE_LEVELS)

    num_recs = select_option("How many recommendations?", [str(n) for n in [1, 2, 3, 5, 10]])
    num_recs = int(num_recs)

    print("\n🔧 Initializing system and generating enhanced recommendations...")
    config_path = os.path.join(PROJECT_ROOT, "config", "config.json")
    system = HybridRecommendationSystem(config_path=config_path)

    user = SystemUser(
        user_id=user_id,
        role=role,
        sectors=sectors,
        location=location,
        capital_range=capital_range,
        risk_appetite=risk,
        experience_level=experience,
    )

    system.register_user(user)

    # Get enhanced recommendations with full evidence and reasoning
    recommendations = system.get_enhanced_recommendations(user_id, max_recommendations=num_recs)

    print(f"\n🎉 Generated {len(recommendations)} diverse, evidence-based recommendations!")
    print("Each recommendation uses a different strategy and includes comprehensive reasoning.")
    
    # Display each enhanced recommendation
    for i, rec in enumerate(recommendations):
        format_enhanced_recommendation(rec, i)
        if i < len(recommendations) - 1:
            input("\nPress Enter to see the next recommendation...")

    # Optional feedback
    send_fb = select_option("Would you like to provide feedback on the first recommendation?", ["no", "yes"]) 
    if send_fb == "yes" and len(recommendations) > 0:
        fb = select_option("Feedback type", ["positive", "negative", "skip"]) 
        if fb in ("positive", "negative"):
            score = 0.8 if fb == "positive" else -0.8
            system.provide_feedback(
                user_id=user_id,
                recommendation_id="rec_1",
                feedback_type="rated_positive" if fb == "positive" else "rated_negative",
                feedback_score=score,
            )
            print("✅ Feedback recorded. Run the wizard again with the same user id to see personalization evolve.")

    # Optional save
    save_opt = select_option("Save these enhanced recommendations to a JSON file?", ["no", "yes"]) 
    if save_opt == "yes":
        import json
        from datetime import datetime
        
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(PROJECT_ROOT, "recommendations", user_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"enhanced_session_{session_ts}.json")
        
        # Convert enhanced recommendations to serializable format
        def serialize_enhanced_rec(rec):
            return {
                "sector": rec.sector,
                "action": rec.action,
                "confidence": rec.confidence,
                "risk_level": rec.risk_level,
                "strategy_type": rec.strategy_type.value,
                "target_companies": [
                    {
                        "name": c.name,
                        "sector": c.sector,
                        "location": c.location,
                        "description": c.description,
                        "confidence_score": c.confidence_score,
                        "risk_level": c.risk_level
                    } for c in rec.target_companies
                ],
                "investment_amount": rec.investment_amount,
                "time_horizon": rec.time_horizon,
                "entry_strategy": rec.entry_strategy,
                "unique_angle": rec.unique_angle,
                "supporting_articles": [
                    {
                        "title": a.title,
                        "source": a.source,
                        "published_date": a.published_date,
                        "sentiment": a.sentiment,
                        "key_points": a.key_points,
                        "relevance_score": a.relevance_score
                    } for a in rec.supporting_articles
                ],
                "market_analysis": {
                    "sector_sentiment": rec.market_analysis.sector_sentiment,
                    "market_trend": rec.market_analysis.market_trend,
                    "volatility_level": rec.market_analysis.volatility_level,
                    "key_drivers": rec.market_analysis.key_drivers,
                    "risk_factors": rec.market_analysis.risk_factors,
                    "opportunity_factors": rec.market_analysis.opportunity_factors
                },
                "policy_impact": {
                    "policy_type": rec.policy_impact.policy_type,
                    "description": rec.policy_impact.description,
                    "impact_level": rec.policy_impact.impact_level,
                    "timeline": rec.policy_impact.timeline,
                    "beneficiaries": rec.policy_impact.beneficiaries
                } if rec.policy_impact else None,
                "risk_factors": rec.risk_factors,
                "opportunity_factors": rec.opportunity_factors,
                "grpo_weight": rec.grpo_weight,
                "grpo_p_weight": rec.grpo_p_weight,
                "blending_strategy": rec.blending_strategy,
                "comprehensive_reasoning": rec.comprehensive_reasoning,
                "market_context": rec.market_context,
                "investment_thesis": rec.investment_thesis,
                "what_makes_this_different": rec.what_makes_this_different,
                "competitive_advantage": rec.competitive_advantage,
                "market_timing": rec.market_timing
            }
        
        payload = {
            "user": {
                "user_id": user_id,
                "role": role,
                "sectors": sectors,
                "location": location,
                "capital_range": capital_range,
                "risk_appetite": risk,
                "experience_level": experience,
            },
            "timestamp": session_ts,
            "recommendations": [serialize_enhanced_rec(r) for r in recommendations],
        }
        
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"💾 Enhanced recommendations saved to: {out_path}")


if __name__ == "__main__":
    main() 