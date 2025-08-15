"""
Complete GRPO-GRPO-P Integration System
Orchestrates the entire hybrid recommendation framework
"""

import sys
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.news_collector import NewsCollector
from src.processing.text_processor import TextProcessor
from src.processing.recommendation_engine import UserProfile
from src.agents.grpo_agent import GRPOPopulation, MarketState
from src.agents.grpo_p_agent import GRPOPAgent, UserFeedback, PersonalizedAction
from src.agents.arbitration_controller import ArbitrationController, ArbitrationContext, BlendedRecommendation

logger = logging.getLogger(__name__)

@dataclass
class SystemUser:
    """Enhanced user profile for the complete system"""
    user_id: str
    role: str  # entrepreneur, investor, business_owner, analyst
    sectors: List[str]
    location: str
    capital_range: str
    risk_appetite: str
    experience_level: str
    
    # System interaction preferences
    notification_frequency: str = "daily"  # daily, weekly, real_time
    preferred_explanation_level: str = "medium"  # brief, medium, detailed
    interface_complexity: str = "standard"  # simple, standard, advanced
    
    # Learning preferences
    learning_mode: bool = True  # Whether to actively learn from user
    feedback_enabled: bool = True  # Whether user provides explicit feedback
    exploration_tolerance: float = 0.3  # How much exploration user tolerates (0-1)

class HybridRecommendationSystem:
    """
    Complete GRPO-GRPO-P Hybrid Business Intelligence System
    Combines group consensus with personalized learning
    """
    
    def __init__(self, 
                 config_path: str = "config/config.json",
                 user_profiles_dir: str = "user_profiles",
                 enable_persistence: bool = True):
        """
        Initialize the complete hybrid recommendation system
        
        Args:
            config_path: Path to configuration file
            user_profiles_dir: Directory for storing user profiles
            enable_persistence: Whether to persist user learning
        """
        
        self.config_path = config_path
        self.user_profiles_dir = user_profiles_dir
        self.enable_persistence = enable_persistence
        
        # Initialize core components
        self.news_collector = None
        self.text_processor = None
        self.grpo_population = None
        self.arbitration_controller = None
        
        # User management
        self.active_users: Dict[str, SystemUser] = {}
        self.grpo_p_agents: Dict[str, GRPOPAgent] = {}
        
        # System state
        self.latest_market_state: Optional[MarketState] = None
        self.system_performance_history = []
        
        # Initialize system
        self._initialize_system()
        
        logger.info("🚀 Hybrid GRPO-GRPO-P Recommendation System initialized")
    
    def _initialize_system(self):
        """Initialize all system components"""
        
        try:
            # Load configuration
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Initialize news collection
            self.news_collector = NewsCollector(
                api_key=config.get('news_api_key', 'demo_api_key')
            )
            
            # Initialize text processing
            self.text_processor = TextProcessor()
            
            # Initialize GRPO population
            available_sectors = ["technology", "finance", "energy", "healthcare", 
                               "retail", "manufacturing", "real_estate"]
            self.grpo_population = GRPOPopulation(
                population_size=15,
                sectors=available_sectors
            )
            
            # Initialize arbitration controller
            self.arbitration_controller = ArbitrationController(
                learning_rate=0.05,
                exploration_rate=0.1
            )
            
            # Create user profiles directory
            if self.enable_persistence:
                os.makedirs(self.user_profiles_dir, exist_ok=True)
            
            logger.info("✅ All system components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing system: {e}")
            raise
    
    def register_user(self, user: SystemUser) -> str:
        """
        Register a new user with the system
        
        Args:
            user: User profile to register
            
        Returns:
            User ID for the registered user
        """
        
        # Store user profile
        self.active_users[user.user_id] = user
        
        # Initialize GRPO-P agent for this user
        agent_profile_path = None
        if self.enable_persistence:
            agent_profile_path = os.path.join(self.user_profiles_dir, f"{user.user_id}.json")
        
        self.grpo_p_agents[user.user_id] = GRPOPAgent(
            user_id=user.user_id,
            sectors=user.sectors,
            persistence_path=agent_profile_path
        )
        
        logger.info(f"👤 Registered new user: {user.user_id} ({user.role})")
        logger.info(f"   Sectors: {user.sectors}")
        logger.info(f"   Risk appetite: {user.risk_appetite}")
        
        return user.user_id
    
    def get_recommendations(self, 
                          user_id: str,
                          max_recommendations: int = 3,
                          include_explanations: bool = True) -> List[BlendedRecommendation]:
        """
        Get personalized hybrid recommendations for a user
        
        Args:
            user_id: ID of the user requesting recommendations
            max_recommendations: Maximum number of recommendations to return
            include_explanations: Whether to include detailed explanations
            
        Returns:
            List of blended recommendations
        """
        
        if user_id not in self.active_users:
            raise ValueError(f"User {user_id} not found. Please register first.")
        
        user = self.active_users[user_id]
        grpo_p_agent = self.grpo_p_agents[user_id]
        
        # Get latest market data
        market_state = self._get_current_market_state()
        
        # Get GRPO population consensus
        grpo_recommendation = self.grpo_population.get_consensus_recommendation(market_state)
        
        # Get personalized GRPO-P recommendation
        grpo_p_recommendation = grpo_p_agent.get_action(market_state)
        
        # Create arbitration context
        context = self._create_arbitration_context(user, market_state)
        
        # Arbitrate between GRPO and GRPO-P
        blended_recommendation = self.arbitration_controller.arbitrate(
            grpo_recommendation, grpo_p_recommendation, context
        )
        
        # Generate additional recommendations if requested
        recommendations = [blended_recommendation]
        
        if max_recommendations > 1:
            # Generate alternative recommendations with different strategies
            additional_recs = self._generate_additional_recommendations(
                user, market_state, max_recommendations - 1
            )
            recommendations.extend(additional_recs)
        
        logger.info(f"📊 Generated {len(recommendations)} recommendations for {user_id}")
        
        return recommendations
    
    def _get_current_market_state(self) -> MarketState:
        """Get current market state from news analysis"""
        
        try:
            # Fetch latest news
            news_articles = self.news_collector.fetch_business_news(max_articles=100)
            
            if not news_articles:
                logger.warning("No news articles fetched, using cached market state")
                return self.latest_market_state or self._create_default_market_state()
            
            # Process news articles
            processed_df = self.text_processor.process_news_dataframe(news_articles)
            
            # Calculate market state
            sector_sentiments = {}
            for sector in processed_df['primary_sector'].unique():
                sector_data = processed_df[processed_df['primary_sector'] == sector]
                avg_sentiment = sector_data['business_sentiment'].mean()
                sector_sentiments[sector] = avg_sentiment
            
            overall_sentiment = processed_df['business_sentiment'].mean()
            volatility = processed_df['business_sentiment'].std()
            
            # Extract trending topics
            all_keywords = []
            for keywords_list in processed_df['top_keywords']:
                all_keywords.extend(keywords_list)
            
            from collections import Counter
            trending_topics = dict(Counter(all_keywords).most_common(10))
            
            market_state = MarketState(
                sector_sentiments=sector_sentiments,
                overall_sentiment=overall_sentiment,
                volatility=volatility,
                trending_topics=trending_topics,
                article_count=len(processed_df),
                timestamp=datetime.now().isoformat()
            )
            
            self.latest_market_state = market_state
            
            logger.info(f"📈 Updated market state with {len(processed_df)} articles")
            logger.info(f"   Overall sentiment: {overall_sentiment:.2f}")
            logger.info(f"   Volatility: {volatility:.2f}")
            
            return market_state
            
        except Exception as e:
            logger.error(f"Error getting market state: {e}")
            return self.latest_market_state or self._create_default_market_state()
    
    def _create_default_market_state(self) -> MarketState:
        """Create a default market state for fallback"""
        
        return MarketState(
            sector_sentiments={"technology": 0.1, "finance": 0.0, "energy": -0.1},
            overall_sentiment=0.0,
            volatility=0.3,
            trending_topics={"market": 5, "business": 4, "investment": 3},
            article_count=0,
            timestamp=datetime.now().isoformat()
        )
    
    def _create_arbitration_context(self, 
                                  user: SystemUser, 
                                  market_state: MarketState) -> ArbitrationContext:
        """Create arbitration context for blending recommendations"""
        
        grpo_p_agent = self.grpo_p_agents[user.user_id]
        user_profile = grpo_p_agent.user_profile
        
        # Calculate historical performance (simplified)
        historical_performance = {"grpo": 0.6, "grpo_p": 0.65}  # Default values
        
        # Get current hour
        current_hour = datetime.now().hour
        
        # Calculate session length (simplified - could track actual session)
        session_length = 30  # Default 30 minutes
        
        # Determine market volatility trend
        volatility_trend = "stable"
        if market_state.volatility > 0.6:
            volatility_trend = "increasing"
        elif market_state.volatility < 0.2:
            volatility_trend = "decreasing"
        
        # Calculate user engagement level (simplified)
        engagement_level = min(1.0, user_profile.total_interactions / 50.0)
        
        return ArbitrationContext(
            user_id=user.user_id,
            market_state=market_state,
            user_profile=user_profile,
            historical_performance=historical_performance,
            time_of_day=current_hour,
            session_length=session_length,
            market_volatility_trend=volatility_trend,
            user_engagement_level=engagement_level
        )
    
    def _generate_additional_recommendations(self, 
                                           user: SystemUser,
                                           market_state: MarketState,
                                           count: int) -> List[BlendedRecommendation]:
        """Generate additional recommendations with different strategies"""
        
        additional_recs = []
        grpo_p_agent = self.grpo_p_agents[user.user_id]
        
        for i in range(count):
            # Get alternative GRPO and GRPO-P recommendations
            # (In a real implementation, these would use different strategies/parameters)
            
            # Alternative GRPO recommendation
            alt_grpo_rec = self.grpo_population.get_consensus_recommendation(market_state)
            
            # Alternative GRPO-P recommendation (with more exploration)
            original_exploration = grpo_p_agent.exploration_rate
            grpo_p_agent.exploration_rate = min(0.8, original_exploration + 0.3)
            alt_grpo_p_rec = grpo_p_agent.get_action(market_state)
            grpo_p_agent.exploration_rate = original_exploration
            
            # Create alternative context with different parameters
            context = self._create_arbitration_context(user, market_state)
            
            # Use different arbitration strategy
            original_strategy = self.arbitration_controller.default_strategy
            from src.agents.arbitration_controller import BlendingStrategy
            alt_strategies = [s for s in BlendingStrategy if s != original_strategy]
            if alt_strategies:
                # Temporarily change strategy
                self.arbitration_controller.default_strategy = alt_strategies[i % len(alt_strategies)]
            
            alt_blended = self.arbitration_controller.arbitrate(
                alt_grpo_rec, alt_grpo_p_rec, context
            )
            
            # Restore original strategy
            self.arbitration_controller.default_strategy = original_strategy
            
            additional_recs.append(alt_blended)
        
        return additional_recs
    
    def provide_feedback(self, 
                        user_id: str,
                        recommendation_id: str,
                        feedback_type: str,
                        feedback_score: float,
                        context: Optional[Dict[str, Any]] = None):
        """
        Provide user feedback on a recommendation
        
        Args:
            user_id: ID of the user providing feedback
            recommendation_id: ID of the recommendation being rated
            feedback_type: Type of feedback ("clicked", "saved", "rated", etc.)
            feedback_score: Score from -1 (negative) to 1 (positive)
            context: Additional context about the feedback
        """
        
        if user_id not in self.active_users:
            logger.error(f"User {user_id} not found")
            return
        
        # Create feedback object
        feedback = UserFeedback(
            user_id=user_id,
            recommendation_id=recommendation_id,
            action_taken=feedback_type,
            feedback_score=feedback_score,
            timestamp=datetime.now().isoformat(),
            context=context or {}
        )
        
        # Update GRPO-P agent with feedback
        grpo_p_agent = self.grpo_p_agents[user_id]
        
        # Find the recommendation in agent's history
        if grpo_p_agent.recommendation_history:
            last_action, timestamp = grpo_p_agent.recommendation_history[-1]
            # Simplified - in real implementation, would match by recommendation_id
            grpo_p_agent.update(self.latest_market_state, last_action, feedback)
        
        # Update arbitration controller
        self.arbitration_controller.update_performance(
            recommendation_id, feedback
        )
        
        logger.info(f"📝 Processed feedback from {user_id}: {feedback_type} ({feedback_score})")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        stats = {
            'system_info': {
                'active_users': len(self.active_users),
                'total_grpo_agents': len(self.grpo_population.agents) if self.grpo_population else 0,
                'latest_market_update': self.latest_market_state.timestamp if self.latest_market_state else None
            },
            'user_stats': {},
            'grpo_population_stats': {},
            'arbitration_stats': {}
        }
        
        # User statistics
        for user_id, user in self.active_users.items():
            if user_id in self.grpo_p_agents:
                agent_stats = self.grpo_p_agents[user_id].get_user_profile_summary()
                stats['user_stats'][user_id] = {
                    'role': user.role,
                    'sectors': user.sectors,
                    'risk_appetite': user.risk_appetite,
                    'agent_stats': agent_stats
                }
        
        # GRPO population statistics
        if self.grpo_population:
            stats['grpo_population_stats'] = self.grpo_population.get_population_stats()
        
        # Arbitration statistics
        if self.arbitration_controller:
            stats['arbitration_stats'] = self.arbitration_controller.get_arbitration_stats()
        
        # Market state
        if self.latest_market_state:
            stats['market_state'] = {
                'overall_sentiment': self.latest_market_state.overall_sentiment,
                'volatility': self.latest_market_state.volatility,
                'article_count': self.latest_market_state.article_count,
                'top_sectors': list(self.latest_market_state.sector_sentiments.keys())[:5],
                'trending_topics': list(self.latest_market_state.trending_topics.keys())[:5]
            }
        
        return stats
    
    def generate_system_report(self, user_id: Optional[str] = None) -> str:
        """Generate a comprehensive system report"""
        
        stats = self.get_system_stats()
        
        report = f"""
🚀 HYBRID GRPO-GRPO-P RECOMMENDATION SYSTEM REPORT
{'='*60}

📊 SYSTEM OVERVIEW:
• Active Users: {stats['system_info']['active_users']}
• GRPO Agents: {stats['system_info']['total_grpo_agents']}
• Last Market Update: {stats['system_info']['latest_market_update']}

📈 MARKET STATE:
"""
        
        if 'market_state' in stats:
            market = stats['market_state']
            report += f"""
• Overall Sentiment: {market['overall_sentiment']:+.2f}
• Market Volatility: {market['volatility']:.2f}
• Articles Analyzed: {market['article_count']}
• Top Sectors: {', '.join(market['top_sectors'])}
• Trending Topics: {', '.join(market['trending_topics'])}
"""
        
        # User-specific report
        if user_id and user_id in stats['user_stats']:
            user_stats = stats['user_stats'][user_id]
            agent_stats = user_stats['agent_stats']
            
            report += f"""

👤 USER PROFILE: {user_id}
{'─'*40}
• Role: {user_stats['role']}
• Sectors: {', '.join(user_stats['sectors'])}
• Risk Appetite: {user_stats['risk_appetite']}

🧠 LEARNING PROGRESS:
• Total Interactions: {agent_stats['total_interactions']}
• Confidence Level: {agent_stats['confidence_level']:.1%}
• Success Rate: {agent_stats['success_rate']:.1%}
• Cold Start Mode: {"Yes" if agent_stats['is_cold_start'] else "No"}
• Exploration Rate: {agent_stats['exploration_rate']:.1%}

🎯 LEARNED PREFERENCES:
"""
            
            for sector, preference in agent_stats['sector_preferences'].items():
                report += f"• {sector.title()}: {preference:.2f}\n"
        
        # Population performance
        if 'grpo_population_stats' in stats and stats['grpo_population_stats']:
            pop_stats = stats['grpo_population_stats']
            report += f"""

🤖 GRPO POPULATION PERFORMANCE:
• Population Performance: {pop_stats.get('population_performance', 0):.2f}
• Active Agents: {pop_stats.get('active_agents', 0)}
• Best Agent: {pop_stats.get('best_agent_performance', 0):.2f}
• Performance Std: {pop_stats.get('performance_std', 0):.2f}
"""
        
        # Arbitration performance
        if 'arbitration_stats' in stats and stats['arbitration_stats']:
            arb_stats = stats['arbitration_stats']
            report += f"""

⚖️ ARBITRATION PERFORMANCE:
• Total Recommendations: {arb_stats.get('total_recommendations', 0)}
• Exploration Rate: {arb_stats.get('exploration_rate', 0):.1%}

Strategy Performance:
"""
            
            for strategy, perf in arb_stats.get('strategy_performance', {}).items():
                report += f"• {strategy}: {perf.get('avg_performance', 0):.2f} ({perf.get('count', 0)} uses)\n"
        
        report += f"""

✅ SYSTEM STATUS: OPERATIONAL
📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Complete Hybrid Recommendation System")
    print("=" * 60)
    
    # Initialize system
    try:
        system = HybridRecommendationSystem()
        
        # Register test user
        test_user = SystemUser(
            user_id="entrepreneur_123",
            role="entrepreneur",
            sectors=["technology", "finance"],
            location="mumbai",
            capital_range="medium",
            risk_appetite="moderate",
            experience_level="intermediate",
            learning_mode=True,
            feedback_enabled=True,
            exploration_tolerance=0.4
        )
        
        user_id = system.register_user(test_user)
        print(f"✅ Registered user: {user_id}")
        
        # Get recommendations
        recommendations = system.get_recommendations(user_id, max_recommendations=2)
        
        print(f"\n📊 Generated {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec.sector.title()} - {rec.recommendation_type.upper()}")
            print(f"   Confidence: {rec.confidence:.1%}")
            print(f"   GRPO Weight: {rec.grpo_weight:.1%}")
            print(f"   GRPO-P Weight: {rec.grpo_p_weight:.1%}")
            print(f"   Strategy: {rec.blending_strategy}")
        
        # Simulate feedback
        system.provide_feedback(
            user_id=user_id,
            recommendation_id="rec_1",
            feedback_type="clicked",
            feedback_score=0.8
        )
        print(f"\n✅ Provided positive feedback")
        
        # Generate report
        report = system.generate_system_report(user_id)
        print(f"\n📋 SYSTEM REPORT:")
        print(report)
        
        print("\n🎉 Complete Hybrid System Test Successful!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
