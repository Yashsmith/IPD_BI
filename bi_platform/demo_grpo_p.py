"""
GRPO-P Framework Demo
Complete demonstration of the Hybrid GRPO-GRPO-P Recommendation System
"""

import sys
import os
import logging
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.hybrid_system import HybridRecommendationSystem, SystemUser
from src.agents.grpo_p_agent import UserFeedback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_demo_users():
    """Create diverse demo users to showcase personalization"""
    
    users = [
        SystemUser(
            user_id="tech_entrepreneur_alice",
            role="entrepreneur",
            sectors=["technology", "finance"],
            location="bangalore",
            capital_range="medium",
            risk_appetite="high",
            experience_level="expert",
            learning_mode=True,
            feedback_enabled=True,
            exploration_tolerance=0.6
        ),
        SystemUser(
            user_id="conservative_investor_bob",
            role="investor",
            sectors=["finance", "healthcare", "energy"],
            location="mumbai",
            capital_range="high",
            risk_appetite="low",
            experience_level="expert",
            learning_mode=True,
            feedback_enabled=True,
            exploration_tolerance=0.2
        ),
        SystemUser(
            user_id="new_business_owner_carol",
            role="business_owner",
            sectors=["retail", "technology"],
            location="delhi",
            capital_range="low",
            risk_appetite="moderate",
            experience_level="beginner",
            learning_mode=True,
            feedback_enabled=True,
            exploration_tolerance=0.4
        ),
        SystemUser(
            user_id="market_analyst_david",
            role="analyst",
            sectors=["technology", "finance", "energy", "healthcare"],
            location="pune",
            capital_range="medium",
            risk_appetite="moderate",
            experience_level="expert",
            learning_mode=True,
            feedback_enabled=True,
            exploration_tolerance=0.3
        )
    ]
    
    return users

def simulate_user_learning_journey(system, user_id, num_interactions=5):
    """Simulate a user's learning journey with the system"""
    
    print(f"\n🎓 SIMULATING LEARNING JOURNEY FOR {user_id}")
    print("=" * 60)
    
    interactions = []
    
    for i in range(num_interactions):
        print(f"\n📅 Interaction {i+1}/{num_interactions}")
        print("-" * 30)
        
        # Get recommendations
        recommendations = system.get_recommendations(user_id, max_recommendations=2)
        
        # Display recommendations
        for j, rec in enumerate(recommendations, 1):
            print(f"\n💡 Recommendation {j}:")
            print(f"   Sector: {rec.sector.title()}")
            print(f"   Action: {rec.recommendation_type.upper()}")
            print(f"   Confidence: {rec.confidence:.1%}")
            print(f"   Risk Level: {rec.risk_level.title()}")
            print(f"   GRPO Influence: {rec.grpo_weight:.1%}")
            print(f"   Personalization: {rec.grpo_p_weight:.1%}")
            print(f"   Strategy: {rec.blending_strategy.replace('_', ' ').title()}")
        
        # Simulate user feedback (varies by user type and interaction)
        feedback_score = simulate_user_feedback(user_id, i, recommendations[0])
        feedback_type = "clicked" if feedback_score > 0 else "ignored"
        
        system.provide_feedback(
            user_id=user_id,
            recommendation_id=f"rec_{i+1}",
            feedback_type=feedback_type,
            feedback_score=feedback_score
        )
        
        print(f"\n👤 User Feedback: {feedback_type} (score: {feedback_score:+.1f})")
        
        # Store interaction data
        interactions.append({
            'interaction': i+1,
            'recommendation': recommendations[0],
            'feedback_score': feedback_score,
            'feedback_type': feedback_type
        })
    
    return interactions

def simulate_user_feedback(user_id, interaction_num, recommendation):
    """Simulate realistic user feedback based on user type and recommendation"""
    
    # Base feedback based on user type
    if "entrepreneur" in user_id:
        # Entrepreneurs like high-risk, technology recommendations
        base_score = 0.6 if recommendation.sector == "technology" else 0.2
        if recommendation.risk_level == "high":
            base_score += 0.3
    elif "conservative_investor" in user_id:
        # Conservative investors prefer low-risk, financial recommendations
        base_score = 0.7 if recommendation.sector == "finance" else 0.3
        if recommendation.risk_level == "low":
            base_score += 0.3
        else:
            base_score -= 0.4
    elif "new_business_owner" in user_id:
        # New business owners prefer moderate risk, practical sectors
        base_score = 0.5 if recommendation.sector in ["retail", "technology"] else 0.2
        if recommendation.risk_level == "moderate":
            base_score += 0.2
    else:  # analyst
        # Analysts prefer diverse, well-reasoned recommendations
        base_score = 0.5
        if recommendation.confidence > 0.7:
            base_score += 0.2
    
    # Learning curve - users become more selective over time
    learning_factor = 1.0 - (interaction_num * 0.1)  # Slightly more critical over time
    
    # Add some randomness
    import random
    noise = random.uniform(-0.3, 0.3)
    
    final_score = base_score * learning_factor + noise
    
    # Clamp to [-1, 1] range
    return max(-1.0, min(1.0, final_score))

def analyze_learning_progression(system, user_id, interactions):
    """Analyze how the user's profile evolved during the learning journey"""
    
    print(f"\n📈 LEARNING ANALYSIS FOR {user_id}")
    print("=" * 50)
    
    # Get user's current profile
    user_stats = system.get_system_stats()['user_stats'][user_id]
    agent_stats = user_stats['agent_stats']
    
    print(f"📊 Final Learning State:")
    print(f"   Total Interactions: {agent_stats['total_interactions']}")
    print(f"   Confidence Level: {agent_stats['confidence_level']:.1%}")
    print(f"   Success Rate: {agent_stats['success_rate']:.1%}")
    print(f"   Still in Cold Start: {'Yes' if agent_stats['is_cold_start'] else 'No'}")
    print(f"   Exploration Rate: {agent_stats['exploration_rate']:.1%}")
    
    print(f"\n🎯 Learned Sector Preferences:")
    for sector, preference in agent_stats['sector_preferences'].items():
        preference_level = "High" if preference > 0.7 else "Medium" if preference > 0.4 else "Low"
        print(f"   {sector.title()}: {preference:.2f} ({preference_level})")
    
    # Analyze feedback progression
    feedback_scores = [interaction['feedback_score'] for interaction in interactions]
    personalization_weights = [interaction['recommendation'].grpo_p_weight for interaction in interactions]
    
    print(f"\n📋 Interaction Progression:")
    print(f"   Average Feedback Score: {sum(feedback_scores)/len(feedback_scores):+.2f}")
    print(f"   Feedback Trend: {'Improving' if feedback_scores[-1] > feedback_scores[0] else 'Declining'}")
    print(f"   Personalization Growth: {personalization_weights[0]:.1%} → {personalization_weights[-1]:.1%}")
    
    # Show strategy evolution
    strategies_used = [interaction['recommendation'].blending_strategy for interaction in interactions]
    from collections import Counter
    strategy_counts = Counter(strategies_used)
    
    print(f"\n⚖️ Arbitration Strategies Used:")
    for strategy, count in strategy_counts.most_common():
        print(f"   {strategy.replace('_', ' ').title()}: {count} times")

def compare_user_profiles(system, user_ids):
    """Compare learned profiles across different users"""
    
    print(f"\n🔍 USER PROFILE COMPARISON")
    print("=" * 60)
    
    stats = system.get_system_stats()
    
    comparison_data = []
    
    for user_id in user_ids:
        if user_id in stats['user_stats']:
            user_stats = stats['user_stats'][user_id]
            agent_stats = user_stats['agent_stats']
            
            comparison_data.append({
                'User': user_id.replace('_', ' ').title(),
                'Role': user_stats['role'],
                'Interactions': agent_stats['total_interactions'],
                'Confidence': f"{agent_stats['confidence_level']:.1%}",
                'Success Rate': f"{agent_stats['success_rate']:.1%}",
                'Risk Pref': f"{agent_stats.get('risk_preference', 0.5):.2f}",
                'Top Sector': max(agent_stats['sector_preferences'].items(), 
                                key=lambda x: x[1])[0] if agent_stats['sector_preferences'] else "None"
            })
    
    # Create comparison table
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
    
    # Analyze diversity
    all_preferences = {}
    for user_id in user_ids:
        if user_id in stats['user_stats']:
            agent_stats = stats['user_stats'][user_id]['agent_stats']
            for sector, pref in agent_stats['sector_preferences'].items():
                if sector not in all_preferences:
                    all_preferences[sector] = []
                all_preferences[sector].append(pref)
    
    print(f"\n🌈 Personalization Diversity:")
    for sector, prefs in all_preferences.items():
        if len(prefs) > 1:
            diversity = max(prefs) - min(prefs)
            print(f"   {sector.title()}: Preference range {min(prefs):.2f} - {max(prefs):.2f} (diversity: {diversity:.2f})")

def main():
    """Main demo function"""
    
    print("🚀 GRPO-P FRAMEWORK COMPREHENSIVE DEMO")
    print("=" * 60)
    print("Demonstrating Hybrid GRPO-GRPO-P Business Intelligence System")
    print("with Group Consensus + Personalized Learning + Intelligent Arbitration")
    print()
    
    try:
        # Initialize the hybrid system
        print("🔧 Initializing Hybrid Recommendation System...")
        system = HybridRecommendationSystem()
        print("✅ System initialized successfully!")
        
        # Create and register demo users
        print("\n👥 Creating diverse user profiles...")
        demo_users = create_demo_users()
        user_ids = []
        
        for user in demo_users:
            user_id = system.register_user(user)
            user_ids.append(user_id)
            print(f"✅ Registered: {user.user_id} ({user.role}, {user.risk_appetite} risk)")
        
        # Show initial system state
        print(f"\n📊 Initial System State:")
        initial_report = system.generate_system_report()
        print(initial_report)
        
        # Simulate learning journeys for each user
        all_interactions = {}
        for user_id in user_ids[:2]:  # Demo with first 2 users for brevity
            interactions = simulate_user_learning_journey(system, user_id, num_interactions=3)
            all_interactions[user_id] = interactions
        
        # Analyze learning progression for each user
        for user_id, interactions in all_interactions.items():
            analyze_learning_progression(system, user_id, interactions)
        
        # Compare user profiles
        compare_user_profiles(system, user_ids)
        
        # Show final system performance
        print(f"\n📈 FINAL SYSTEM PERFORMANCE")
        print("=" * 50)
        
        final_stats = system.get_system_stats()
        
        if 'arbitration_stats' in final_stats:
            arb_stats = final_stats['arbitration_stats']
            print(f"📊 Arbitration Controller:")
            print(f"   Total Recommendations: {arb_stats.get('total_recommendations', 0)}")
            print(f"   Exploration Rate: {arb_stats.get('exploration_rate', 0):.1%}")
            
            if 'strategy_performance' in arb_stats:
                print(f"\n⚖️ Strategy Performance:")
                for strategy, perf in arb_stats['strategy_performance'].items():
                    if perf.get('count', 0) > 0:
                        print(f"   {strategy.replace('_', ' ').title()}: {perf['avg_performance']:.2f} "
                              f"({perf['count']} uses)")
        
        # Demonstrate real-time recommendation
        print(f"\n🎯 REAL-TIME RECOMMENDATION DEMO")
        print("=" * 40)
        
        demo_user_id = user_ids[0]
        print(f"Getting fresh recommendations for {demo_user_id}...")
        
        final_recommendations = system.get_recommendations(demo_user_id, max_recommendations=3)
        
        for i, rec in enumerate(final_recommendations, 1):
            print(f"\n💡 Recommendation {i}:")
            print(f"   📍 Sector: {rec.sector.title()}")
            print(f"   🎯 Action: {rec.recommendation_type.upper()}")
            print(f"   📊 Confidence: {rec.confidence:.1%}")
            print(f"   ⚖️ Blend: {rec.grpo_weight:.0%} Group + {rec.grpo_p_weight:.0%} Personal")
            print(f"   🧠 Strategy: {rec.blending_strategy.replace('_', ' ').title()}")
            print(f"   ⚠️ Risk: {rec.risk_level.title()}")
            
            if len(rec.alternative_options) > 0:
                print(f"   🔄 Alternatives: {len(rec.alternative_options)} options available")
        
        # Show final system report
        print(f"\n📋 COMPREHENSIVE SYSTEM REPORT")
        print("=" * 50)
        final_report = system.generate_system_report(demo_user_id)
        print(final_report)
        
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Key Achievements Demonstrated:")
        print("✅ Multi-user personalization learning")
        print("✅ Group consensus vs. individual preference balancing")
        print("✅ Dynamic arbitration strategy selection")
        print("✅ Real-time market analysis integration")
        print("✅ Feedback-driven continuous improvement")
        print("✅ Diverse user profile accommodation")
        print("\n🚀 The GRPO-P framework successfully combines:")
        print("   🤖 GRPO: Population-based group consensus")
        print("   👤 GRPO-P: Individual preference learning")
        print("   ⚖️ Arbitration: Intelligent recommendation blending")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
