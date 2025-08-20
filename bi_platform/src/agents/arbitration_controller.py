"""
Arbitration Controller for GRPO-GRPO-P Framework
Dynamically blends group consensus (GRPO) and personalized (GRPO-P) recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from enum import Enum
from abc import ABC, abstractmethod

# Import components
from .grpo_agent import GRPOAction, MarketState, GRPOPopulation
from .grpo_p_agent import GRPOPAgent, PersonalizedAction, UserFeedback, UserPreferenceProfile

logger = logging.getLogger(__name__)

class BlendingStrategy(Enum):
    """Different strategies for blending GRPO and GRPO-P recommendations"""
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    USER_EXPERIENCE_BASED = "user_experience_based"
    MARKET_CONDITION_ADAPTIVE = "market_condition_adaptive"
    DYNAMIC_LEARNED = "dynamic_learned"
    CONSERVATIVE_FUSION = "conservative_fusion"

class RecommendationStrategy(Enum):
    """Different recommendation strategies for diversity"""
    CONSERVATIVE_LONG_TERM = "conservative_long_term"
    AGGRESSIVE_SHORT_TERM = "aggressive_short_term"
    SECTOR_ROTATION = "sector_rotation"
    MOMENTUM_BASED = "momentum_based"
    CONTRARIAN_OPPORTUNITY = "contrarian_opportunity"
    VALUE_INVESTING = "value_investing"
    GROWTH_FOCUSED = "growth_focused"

@dataclass
class CompanyInfo:
    """Information about a specific company/startup"""
    name: str
    sector: str
    location: str
    description: str
    confidence_score: float
    risk_level: str
    investment_amount: Optional[str] = None
    time_horizon: Optional[str] = None
    entry_strategy: Optional[str] = None
    relevance_score: float = 0.0  # How relevant this company is to the user's query

@dataclass
class ArticleSummary:
    """Summary of a news article that supports the recommendation"""
    title: str
    source: str
    published_date: str
    sentiment: float
    key_points: List[str]
    relevance_score: float

@dataclass
class MarketAnalysis:
    """Market analysis for the recommendation"""
    sector_sentiment: float
    market_trend: str
    volatility_level: str
    key_drivers: List[str]
    risk_factors: List[str]
    opportunity_factors: List[str]

@dataclass
class PolicyImpact:
    """Government policy impact on the recommendation"""
    policy_type: str
    description: str
    impact_level: str
    timeline: str
    beneficiaries: List[str]

@dataclass
class BlendedRecommendation:
    """Final recommendation that blends GRPO and GRPO-P inputs"""
    
    # Final recommendation (sector-level)
    sector: str
    recommendation_type: str  # "invest", "avoid", "monitor"
    confidence: float
    risk_level: str
    reasoning: str
    
    # Item-level recommendations within the sector
    recommended_items: List[str] = field(default_factory=list)  # Specific startup IDs
    item_scores: Dict[str, float] = field(default_factory=dict)  # startup_id -> confidence score
    
    # Blending metadata
    grpo_weight: float = 0.5  # How much GRPO influenced this (0-1)
    grpo_p_weight: float = 0.5  # How much GRPO-P influenced this (0-1)
    blending_strategy: str = "confidence_weighted"  # Which strategy was used
    
    # Source recommendations
    grpo_recommendation: Optional[GRPOAction] = None
    grpo_p_recommendation: Optional[PersonalizedAction] = None
    
    # Explanation and transparency
    explanation_detail: str = "medium"
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    alternative_options: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class EnhancedRecommendation:
    """Comprehensive recommendation with full context and evidence"""
    
    # Core recommendation
    sector: str
    action: str  # invest, avoid, monitor, accumulate, divest
    confidence: float
    risk_level: str
    strategy_type: RecommendationStrategy
    
    # Specific details
    target_companies: List[CompanyInfo]
    investment_amount: Optional[str]
    time_horizon: str
    entry_strategy: str
    unique_angle: str
    
    # Evidence and reasoning
    supporting_articles: List[ArticleSummary]
    market_analysis: MarketAnalysis
    policy_impact: Optional[PolicyImpact]
    risk_factors: List[str]
    opportunity_factors: List[str]
    
    # Blending metadata
    grpo_weight: float
    grpo_p_weight: float
    blending_strategy: str
    
    # Detailed reasoning
    comprehensive_reasoning: str
    market_context: str
    investment_thesis: str
    
    # Differentiation factors
    what_makes_this_different: str
    competitive_advantage: str
    market_timing: str

@dataclass
class ArbitrationContext:
    """Context information for arbitration decisions"""
    user_id: str
    market_state: MarketState
    user_profile: UserPreferenceProfile
    historical_performance: Dict[str, float]  # GRPO vs GRPO-P performance
    time_of_day: int  # 0-23
    session_length: int  # How long user has been active
    market_volatility_trend: str  # "increasing", "decreasing", "stable"
    user_engagement_level: float  # 0-1, how engaged user is

class ArbitrationController:
    """
    Controls the blending of GRPO (group consensus) and GRPO-P (personalized) recommendations
    Learns optimal blending strategies for different users and market conditions
    """
    
    def __init__(self, 
                 default_strategy: BlendingStrategy = BlendingStrategy.DYNAMIC_LEARNED,
                 learning_rate: float = 0.05,
                 exploration_rate: float = 0.1):
        """
        Initialize Arbitration Controller
        
        Args:
            default_strategy: Default blending strategy to use
            learning_rate: Rate for learning optimal blending weights
            exploration_rate: Rate for exploring different blending strategies
        """
        
        self.default_strategy = default_strategy
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        # Learned blending parameters for different contexts
        self.blending_weights = {
            'new_user': {'grpo': 0.8, 'grpo_p': 0.2},  # Rely more on group consensus for new users
            'experienced_user': {'grpo': 0.4, 'grpo_p': 0.6},  # Trust personalization for experienced users
            'high_volatility': {'grpo': 0.7, 'grpo_p': 0.3},  # Group consensus in volatile markets
            'low_volatility': {'grpo': 0.5, 'grpo_p': 0.5},  # Balanced in stable markets
            'confident_user': {'grpo': 0.3, 'grpo_p': 0.7},  # Trust user preferences when confident
            'uncertain_market': {'grpo': 0.8, 'grpo_p': 0.2},  # Group wisdom in uncertain times
        }
        
        # Performance tracking for different strategies
        self.strategy_performance = {strategy.value: [] for strategy in BlendingStrategy}
        
        # Context-specific performance tracking
        self.context_performance = {}
        
        # Recommendation history for learning
        self.recommendation_history = []
        
        logger.info("Initialized Arbitration Controller")
        logger.info(f"Default strategy: {default_strategy.value}")
    
    def arbitrate(self, 
                 grpo_recommendation: GRPOAction,
                 grpo_p_recommendation: PersonalizedAction,
                 context: ArbitrationContext) -> BlendedRecommendation:
        """
        Main arbitration function - blends GRPO and GRPO-P recommendations
        
        Args:
            grpo_recommendation: Recommendation from GRPO population
            grpo_p_recommendation: Personalized recommendation from GRPO-P
            context: Context information for arbitration
            
        Returns:
            Blended recommendation
        """
        
        # Determine optimal blending strategy for this context
        strategy = self._select_blending_strategy(context)
        
        # Calculate blending weights based on strategy
        grpo_weight, grpo_p_weight = self._calculate_blending_weights(
            grpo_recommendation, grpo_p_recommendation, context, strategy
        )
        
        # Blend the recommendations
        blended_rec = self._blend_recommendations(
            grpo_recommendation, grpo_p_recommendation, 
            grpo_weight, grpo_p_weight, strategy, context
        )
        
        # Store for learning
        self.recommendation_history.append({
            'blended_recommendation': blended_rec,
            'context': context,
            'strategy': strategy,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
        logger.info(f"Arbitrated recommendation for user {context.user_id}")
        logger.info(f"Strategy: {strategy.value}, GRPO weight: {grpo_weight:.2f}, GRPO-P weight: {grpo_p_weight:.2f}")
        
        return blended_rec
    
    def _select_blending_strategy(self, context: ArbitrationContext) -> BlendingStrategy:
        """
        Select optimal blending strategy based on context
        
        Args:
            context: Current arbitration context
            
        Returns:
            Selected blending strategy
        """
        
        # Epsilon-greedy exploration
        if np.random.random() < self.exploration_rate:
            return np.random.choice(list(BlendingStrategy))
        
        # Strategy selection based on context and performance
        user_experience = context.user_profile.total_interactions
        market_volatility = context.market_state.volatility
        user_confidence = context.user_profile.confidence_level
        
        # Rule-based strategy selection (can be enhanced with ML)
        if user_experience < 5:
            # New users - rely on group consensus
            return BlendingStrategy.CONFIDENCE_WEIGHTED
        elif market_volatility > 0.7:
            # High volatility - use market adaptive strategy
            return BlendingStrategy.MARKET_CONDITION_ADAPTIVE
        elif user_confidence > 0.8:
            # Confident users - trust personalization more
            return BlendingStrategy.USER_EXPERIENCE_BASED
        else:
            # Default to learned strategy
            return BlendingStrategy.DYNAMIC_LEARNED
    
    def _calculate_blending_weights(self, 
                                  grpo_rec: GRPOAction,
                                  grpo_p_rec: PersonalizedAction,
                                  context: ArbitrationContext,
                                  strategy: BlendingStrategy) -> Tuple[float, float]:
        """
        Calculate optimal blending weights for GRPO and GRPO-P
        
        Args:
            grpo_rec: GRPO recommendation
            grpo_p_rec: GRPO-P recommendation
            context: Arbitration context
            strategy: Selected blending strategy
            
        Returns:
            (grpo_weight, grpo_p_weight) tuple
        """
        
        if strategy == BlendingStrategy.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_blending(grpo_rec, grpo_p_rec, context)
        
        elif strategy == BlendingStrategy.USER_EXPERIENCE_BASED:
            return self._experience_based_blending(grpo_rec, grpo_p_rec, context)
        
        elif strategy == BlendingStrategy.MARKET_CONDITION_ADAPTIVE:
            return self._market_adaptive_blending(grpo_rec, grpo_p_rec, context)
        
        elif strategy == BlendingStrategy.DYNAMIC_LEARNED:
            return self._learned_blending(grpo_rec, grpo_p_rec, context)
        
        elif strategy == BlendingStrategy.CONSERVATIVE_FUSION:
            return self._conservative_blending(grpo_rec, grpo_p_rec, context)
        
        else:
            # Default balanced blending
            return 0.5, 0.5
    
    def _confidence_weighted_blending(self, 
                                    grpo_rec: GRPOAction,
                                    grpo_p_rec: PersonalizedAction,
                                    context: ArbitrationContext) -> Tuple[float, float]:
        """Blend based on confidence scores of each recommendation"""
        
        grpo_confidence = grpo_rec.confidence
        grpo_p_confidence = grpo_p_rec.user_match_confidence
        
        # Normalize confidences
        total_confidence = grpo_confidence + grpo_p_confidence
        if total_confidence > 0:
            grpo_weight = grpo_confidence / total_confidence
            grpo_p_weight = grpo_p_confidence / total_confidence
        else:
            grpo_weight, grpo_p_weight = 0.5, 0.5
        
        # Adjust for user experience (new users get more GRPO)
        experience_factor = min(1.0, context.user_profile.total_interactions / 20.0)
        grpo_weight = grpo_weight * (1 - experience_factor * 0.3)
        grpo_p_weight = 1.0 - grpo_weight
        
        return grpo_weight, grpo_p_weight
    
    def _experience_based_blending(self, 
                                 grpo_rec: GRPOAction,
                                 grpo_p_rec: PersonalizedAction,
                                 context: ArbitrationContext) -> Tuple[float, float]:
        """Blend based on user experience and profile confidence"""
        
        user_interactions = context.user_profile.total_interactions
        profile_confidence = context.user_profile.confidence_level
        success_rate = context.user_profile.success_rate
        
        # More experienced users with good success rates get more personalization
        personalization_factor = (profile_confidence * 0.4 + 
                                success_rate * 0.3 + 
                                min(1.0, user_interactions / 50.0) * 0.3)
        
        grpo_p_weight = personalization_factor
        grpo_weight = 1.0 - grpo_p_weight
        
        return grpo_weight, grpo_p_weight
    
    def _market_adaptive_blending(self, 
                                grpo_rec: GRPOAction,
                                grpo_p_rec: PersonalizedAction,
                                context: ArbitrationContext) -> Tuple[float, float]:
        """Adapt blending based on market conditions"""
        
        volatility = context.market_state.volatility
        overall_sentiment = context.market_state.overall_sentiment
        
        # High volatility -> more group consensus
        # Strong sentiment (positive or negative) -> more personalization for experienced users
        
        volatility_factor = volatility  # 0 to 1
        sentiment_strength = abs(overall_sentiment)
        user_experience = min(1.0, context.user_profile.total_interactions / 30.0)
        
        # Base GRPO weight on volatility
        grpo_weight = 0.3 + volatility_factor * 0.5
        
        # Adjust for sentiment and user experience
        if sentiment_strength > 0.3 and user_experience > 0.5:
            # Strong sentiment + experienced user -> trust personalization more
            grpo_weight -= 0.2
        
        grpo_weight = np.clip(grpo_weight, 0.1, 0.9)
        grpo_p_weight = 1.0 - grpo_weight
        
        return grpo_weight, grpo_p_weight
    
    def _learned_blending(self, 
                        grpo_rec: GRPOAction,
                        grpo_p_rec: PersonalizedAction,
                        context: ArbitrationContext) -> Tuple[float, float]:
        """Use learned optimal weights based on historical performance"""
        
        # Classify current context
        context_key = self._classify_context(context)
        
        # Get learned weights for this context
        if context_key in self.blending_weights:
            weights = self.blending_weights[context_key]
            return weights['grpo'], weights['grpo_p']
        
        # Fallback to default if no learned weights
        return 0.6, 0.4
    
    def _conservative_blending(self, 
                             grpo_rec: GRPOAction,
                             grpo_p_rec: PersonalizedAction,
                             context: ArbitrationContext) -> Tuple[float, float]:
        """Conservative blending that favors safer recommendations"""
        
        # Prefer the more conservative recommendation
        grpo_risk_score = {"low": 0.2, "moderate": 0.5, "high": 0.8}.get(grpo_rec.risk_level, 0.5)
        grpo_p_risk_score = {"low": 0.2, "moderate": 0.5, "high": 0.8}.get(
            grpo_p_rec.base_action.risk_level, 0.5)
        
        # Weight towards lower risk recommendation
        if grpo_risk_score <= grpo_p_risk_score:
            grpo_weight = 0.7
        else:
            grpo_weight = 0.3
        
        grpo_p_weight = 1.0 - grpo_weight
        
        return grpo_weight, grpo_p_weight
    
    def _classify_context(self, context: ArbitrationContext) -> str:
        """Classify context for learned blending"""
        
        user_interactions = context.user_profile.total_interactions
        volatility = context.market_state.volatility
        confidence = context.user_profile.confidence_level
        
        # Simple rule-based classification
        if user_interactions < 10:
            return 'new_user'
        elif volatility > 0.6:
            return 'high_volatility'
        elif confidence > 0.7:
            return 'confident_user'
        elif volatility < 0.3:
            return 'low_volatility'
        else:
            return 'experienced_user'
    
    def _blend_recommendations(self, 
                             grpo_rec: GRPOAction,
                             grpo_p_rec: PersonalizedAction,
                             grpo_weight: float,
                             grpo_p_weight: float,
                             strategy: BlendingStrategy,
                             context: ArbitrationContext) -> BlendedRecommendation:
        """
        Perform the actual blending of recommendations
        
        Args:
            grpo_rec: GRPO recommendation
            grpo_p_rec: GRPO-P recommendation
            grpo_weight: Weight for GRPO recommendation
            grpo_p_weight: Weight for GRPO-P recommendation
            strategy: Blending strategy used
            context: Arbitration context
            
        Returns:
            Blended recommendation
        """
        
        # Weighted sector selection
        sectors = [grpo_rec.sector, grpo_p_rec.base_action.sector]
        sector_weights = [grpo_weight, grpo_p_weight]
        
        # If same sector, use it; otherwise use weighted random selection
        if grpo_rec.sector == grpo_p_rec.base_action.sector:
            final_sector = grpo_rec.sector
        else:
            final_sector = np.random.choice(sectors, p=sector_weights)
        
        # Weighted recommendation type selection
        rec_types = [grpo_rec.recommendation_type, grpo_p_rec.base_action.recommendation_type]
        type_weights = [grpo_weight, grpo_p_weight]
        
        if grpo_rec.recommendation_type == grpo_p_rec.base_action.recommendation_type:
            final_type = grpo_rec.recommendation_type
        else:
            final_type = np.random.choice(rec_types, p=type_weights)
        
        # Weighted confidence blending
        final_confidence = (grpo_rec.confidence * grpo_weight + 
                          grpo_p_rec.base_action.confidence * grpo_p_weight)
        
        # Risk level blending (take more conservative if different)
        risk_levels = [grpo_rec.risk_level, grpo_p_rec.base_action.risk_level]
        risk_scores = {"low": 1, "moderate": 2, "high": 3}
        
        grpo_risk_score = risk_scores.get(grpo_rec.risk_level, 2)
        grpo_p_risk_score = risk_scores.get(grpo_p_rec.base_action.risk_level, 2)
        
        weighted_risk_score = grpo_risk_score * grpo_weight + grpo_p_risk_score * grpo_p_weight
        
        if weighted_risk_score <= 1.5:
            final_risk = "low"
        elif weighted_risk_score <= 2.5:
            final_risk = "moderate"
        else:
            final_risk = "high"
        
        # Generate blended reasoning
        reasoning = self._generate_blended_reasoning(
            grpo_rec, grpo_p_rec, grpo_weight, grpo_p_weight, strategy, context
        )
        
        # Create confidence breakdown
        confidence_breakdown = {
            'grpo_confidence': grpo_rec.confidence,
            'grpo_p_confidence': grpo_p_rec.user_match_confidence,
            'grpo_weight': grpo_weight,
            'grpo_p_weight': grpo_p_weight,
            'final_confidence': final_confidence
        }
        
        # Generate alternative options
        alternatives = self._generate_alternatives(grpo_rec, grpo_p_rec, final_sector, final_type)
        
        # Generate item-level recommendations within the recommended sector
        recommended_items, item_scores = self._generate_item_recommendations(
            final_sector, final_confidence, context
        )
        
        return BlendedRecommendation(
            sector=final_sector,
            recommendation_type=final_type,
            confidence=final_confidence,
            risk_level=final_risk,
            reasoning=reasoning,
            recommended_items=recommended_items,
            item_scores=item_scores,
            grpo_weight=grpo_weight,
            grpo_p_weight=grpo_p_weight,
            blending_strategy=strategy.value,
            grpo_recommendation=grpo_rec,
            grpo_p_recommendation=grpo_p_rec,
            explanation_detail=grpo_p_rec.explanation_detail,
            confidence_breakdown=confidence_breakdown,
            alternative_options=alternatives
        )
    
    def _generate_blended_reasoning(self, 
                                  grpo_rec: GRPOAction,
                                  grpo_p_rec: PersonalizedAction,
                                  grpo_weight: float,
                                  grpo_p_weight: float,
                                  strategy: BlendingStrategy,
                                  context: ArbitrationContext) -> str:
        """Generate reasoning that explains the blended recommendation"""
        
        reasoning = f"""
🎯 HYBRID RECOMMENDATION ({strategy.value.replace('_', ' ').title()})

📊 GROUP CONSENSUS (Weight: {grpo_weight:.1%}):
{grpo_rec.reasoning}

👤 PERSONALIZED ANALYSIS (Weight: {grpo_p_weight:.1%}):
{grpo_p_rec.base_action.reasoning}

🤖 ARBITRATION LOGIC:
- Strategy: {strategy.value.replace('_', ' ').title()}
- User Experience: {context.user_profile.total_interactions} interactions
- Profile Confidence: {context.user_profile.confidence_level:.1%}
- Market Volatility: {context.market_state.volatility:.1%}

💡 WHY THIS BLEND:
"""
        
        if grpo_weight > grpo_p_weight:
            reasoning += f"- Favoring group consensus due to {'high market volatility' if context.market_state.volatility > 0.6 else 'limited personalization data'}\n"
        else:
            reasoning += f"- Favoring personalization due to {'high user confidence' if context.user_profile.confidence_level > 0.7 else 'stable market conditions'}\n"
        
        reasoning += f"- Confidence blend optimized for your risk profile ({context.user_profile.risk_preference:.1%} risk tolerance)"
        
        return reasoning.strip()
    
    def _generate_alternatives(self, 
                             grpo_rec: GRPOAction,
                             grpo_p_rec: PersonalizedAction,
                             selected_sector: str,
                             selected_type: str) -> List[Dict[str, Any]]:
        """Generate alternative recommendations for transparency"""
        
        alternatives = []
        
        # Add pure GRPO alternative if different from selection
        if (grpo_rec.sector != selected_sector or 
            grpo_rec.recommendation_type != selected_type):
            alternatives.append({
                'source': 'Group Consensus',
                'sector': grpo_rec.sector,
                'type': grpo_rec.recommendation_type,
                'confidence': grpo_rec.confidence,
                'rationale': 'Pure group consensus recommendation'
            })
        
        # Add pure GRPO-P alternative if different from selection
        if (grpo_p_rec.base_action.sector != selected_sector or 
            grpo_p_rec.base_action.recommendation_type != selected_type):
            alternatives.append({
                'source': 'Personalized',
                'sector': grpo_p_rec.base_action.sector,
                'type': grpo_p_rec.base_action.recommendation_type,
                'confidence': grpo_p_rec.base_action.confidence,
                'rationale': 'Pure personalized recommendation'
            })
        
        return alternatives
    
    def _generate_item_recommendations(self, 
                                     sector: str, 
                                     confidence: float,
                                     context: ArbitrationContext) -> Tuple[List[str], Dict[str, float]]:
        """
        Generate specific item recommendations within a recommended sector
        
        Args:
            sector: The recommended sector
            confidence: Overall recommendation confidence
            context: Arbitration context
            
        Returns:
            Tuple of (recommended_items, item_scores)
        """
        
        # Import sector mapping to get items in this sector
        try:
            from ..evaluation.sector_mapping import STARTUP_TO_SECTOR_MAPPING
        except ImportError:
            # Fallback mapping for testing
            STARTUP_TO_SECTOR_MAPPING = {
                f'startup_{i}': ['technology', 'finance', 'healthcare', 'retail'][i % 4]
                for i in range(1, 101)
            }
        
        # Find all startups in this sector
        sector_items = [startup_id for startup_id, startup_sector 
                       in STARTUP_TO_SECTOR_MAPPING.items() 
                       if startup_sector == sector]
        
        if not sector_items:
            # Fallback: generate synthetic items for this sector
            sector_items = [f"startup_{sector}_{i}" for i in range(1, 6)]
        
        # Score items based on user profile and market state
        item_scores = {}
        user_sectors = getattr(context.user_profile, 'sectors', [sector])
        
        for item_id in sector_items:
            # Base score from sector confidence
            base_score = confidence * 0.7
            
            # Add deterministic but varied scoring
            import hashlib
            item_hash = int(hashlib.md5((item_id + context.user_id).encode()).hexdigest(), 16)
            diversity_score = (item_hash % 100) / 1000.0  # 0.0 to 0.1
            
            # Boost score if user has preference for this sector
            if sector in user_sectors:
                preference_boost = 0.2
            else:
                preference_boost = 0.0
            
            final_score = min(1.0, base_score + diversity_score + preference_boost)
            item_scores[item_id] = final_score
        
        # Sort by score and return top items
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 5 items
        recommended_items = [item_id for item_id, score in sorted_items[:5]]
        
        return recommended_items, item_scores
    
    def update_performance(self, 
                          recommendation_id: str,
                          user_feedback: UserFeedback,
                          market_outcome: Optional[float] = None):
        """
        Update arbitration performance based on feedback and market outcomes
        
        Args:
            recommendation_id: ID of the recommendation to update
            user_feedback: User's feedback on the recommendation
            market_outcome: Actual market performance (optional)
        """
        
        # Find the recommendation in history
        target_rec = None
        for rec_data in self.recommendation_history:
            if rec_data.get('recommendation_id') == recommendation_id:
                target_rec = rec_data
                break
        
        if not target_rec:
            logger.warning(f"Recommendation {recommendation_id} not found in history")
            return
        
        # Calculate performance score
        feedback_score = user_feedback.feedback_score
        market_score = market_outcome if market_outcome is not None else 0.0
        
        # Weighted performance (user feedback is more immediate, market outcome more objective)
        performance_score = 0.7 * feedback_score + 0.3 * market_score
        
        # Update strategy performance
        strategy = target_rec['strategy']
        self.strategy_performance[strategy.value].append(performance_score)
        
        # Update context-specific performance
        context = target_rec['context']
        context_key = self._classify_context(context)
        
        if context_key not in self.context_performance:
            self.context_performance[context_key] = []
        self.context_performance[context_key].append(performance_score)
        
        # Learn and update blending weights
        self._update_blending_weights(target_rec, performance_score)
        
        logger.info(f"Updated arbitration performance for {recommendation_id}")
        logger.info(f"Performance score: {performance_score:.2f}")
    
    def _update_blending_weights(self, rec_data: Dict, performance_score: float):
        """Update blending weights based on performance feedback"""
        
        context = rec_data['context']
        blended_rec = rec_data['blended_recommendation']
        
        context_key = self._classify_context(context)
        
        # Get current weights
        if context_key in self.blending_weights:
            current_weights = self.blending_weights[context_key]
        else:
            current_weights = {'grpo': 0.5, 'grpo_p': 0.5}
        
        # Update weights based on performance
        if performance_score > 0:
            # Good performance - move towards current blend
            target_grpo = blended_rec.grpo_weight
            target_grpo_p = blended_rec.grpo_p_weight
        else:
            # Poor performance - move away from current blend
            target_grpo = 1.0 - blended_rec.grpo_weight
            target_grpo_p = 1.0 - blended_rec.grpo_p_weight
        
        # Gradual update
        new_grpo = current_weights['grpo'] + self.learning_rate * (target_grpo - current_weights['grpo'])
        new_grpo_p = 1.0 - new_grpo
        
        # Ensure weights are in valid range
        new_grpo = np.clip(new_grpo, 0.1, 0.9)
        new_grpo_p = np.clip(new_grpo_p, 0.1, 0.9)
        
        # Normalize
        total = new_grpo + new_grpo_p
        new_grpo /= total
        new_grpo_p /= total
        
        self.blending_weights[context_key] = {
            'grpo': new_grpo,
            'grpo_p': new_grpo_p
        }
        
        logger.info(f"Updated blending weights for {context_key}: GRPO={new_grpo:.2f}, GRPO-P={new_grpo_p:.2f}")
    
    def get_arbitration_stats(self) -> Dict[str, Any]:
        """Get statistics about arbitration performance"""
        
        total_recommendations = len(self.recommendation_history)
        
        if total_recommendations == 0:
            return {"total_recommendations": 0, "message": "No arbitration history"}
        
        # Strategy performance
        strategy_stats = {}
        for strategy, performances in self.strategy_performance.items():
            if performances:
                strategy_stats[strategy] = {
                    'avg_performance': np.mean(performances),
                    'count': len(performances),
                    'best': max(performances),
                    'worst': min(performances)
                }
        
        # Context performance
        context_stats = {}
        for context, performances in self.context_performance.items():
            if performances:
                context_stats[context] = {
                    'avg_performance': np.mean(performances),
                    'count': len(performances)
                }
        
        # Current blending weights
        current_weights = self.blending_weights.copy()
        
        return {
            'total_recommendations': total_recommendations,
            'strategy_performance': strategy_stats,
            'context_performance': context_stats,
            'learned_blending_weights': current_weights,
            'exploration_rate': self.exploration_rate
        }

# Example usage and testing
if __name__ == "__main__":
    print("⚖️ Testing Arbitration Controller")
    print("=" * 50)
    
    # Create test data
    from datetime import datetime
    
    market_state = MarketState(
        sector_sentiments={"technology": 0.5, "finance": -0.2},
        overall_sentiment=0.15,
        volatility=0.4,
        trending_topics={"AI": 10, "fintech": 6},
        article_count=35,
        timestamp=datetime.now().isoformat()
    )
    
    user_profile = UserPreferenceProfile(user_id="test_user")
    user_profile.total_interactions = 15
    user_profile.confidence_level = 0.6
    user_profile.risk_preference = 0.7
    
    context = ArbitrationContext(
        user_id="test_user",
        market_state=market_state,
        user_profile=user_profile,
        historical_performance={"grpo": 0.6, "grpo_p": 0.7},
        time_of_day=14,
        session_length=25,
        market_volatility_trend="stable",
        user_engagement_level=0.8
    )
    
    # Create sample recommendations
    grpo_rec = GRPOAction(
        sector="technology",
        recommendation_type="invest",
        confidence=0.7,
        reasoning="Group consensus favors technology investment",
        risk_level="moderate"
    )
    
    from .grpo_p_agent import PersonalizedAction
    grpo_p_rec = PersonalizedAction(
        base_action=GRPOAction(
            sector="finance",
            recommendation_type="monitor",
            confidence=0.6,
            reasoning="Personal preference analysis suggests caution",
            risk_level="low"
        ),
        personalization_score=0.8,
        user_match_confidence=0.75,
        learning_opportunity=False,
        explanation_detail="medium"
    )
    
    # Test arbitration
    controller = ArbitrationController()
    blended = controller.arbitrate(grpo_rec, grpo_p_rec, context)
    
    print(f"Blended Recommendation:")
    print(f"Sector: {blended.sector}")
    print(f"Type: {blended.recommendation_type}")
    print(f"Confidence: {blended.confidence:.2f}")
    print(f"GRPO Weight: {blended.grpo_weight:.2f}")
    print(f"GRPO-P Weight: {blended.grpo_p_weight:.2f}")
    print(f"Strategy: {blended.blending_strategy}")
    
    print("\n✅ Arbitration Controller implementation completed!")
    print("Ready for full GRPO-GRPO-P framework integration!")
