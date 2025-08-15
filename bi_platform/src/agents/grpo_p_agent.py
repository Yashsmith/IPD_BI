"""
GRPO-P (GRPO-Personalization) Agent
Learns individual user preferences through interaction feedback
Complements GRPO population consensus with personalized learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict, deque

# Import base classes from GRPO agent
from .grpo_agent import Agent, MarketState, GRPOAction

logger = logging.getLogger(__name__)

@dataclass
class UserFeedback:
    """Represents user feedback on a recommendation"""
    user_id: str
    recommendation_id: str
    action_taken: str  # "clicked", "saved", "ignored", "rated_positive", "rated_negative"
    feedback_score: float  # -1 to 1 (explicit rating or implicit from action)
    timestamp: str
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context

@dataclass
class UserPreferenceProfile:
    """Dynamic user preference profile learned from interactions"""
    user_id: str
    
    # Learned preferences
    sector_preferences: Dict[str, float] = field(default_factory=dict)  # Sector weights
    risk_preference: float = 0.5  # 0 = very conservative, 1 = very aggressive
    sentiment_sensitivity: float = 0.5  # How much user cares about market sentiment
    timing_preference: str = "moderate"  # "early", "moderate", "late" (when they like to act)
    
    # Behavioral patterns
    interaction_frequency: float = 0.0  # How often user interacts
    feedback_pattern: Dict[str, int] = field(default_factory=dict)  # Types of actions user takes
    success_rate: float = 0.0  # Success rate of user's actions
    
    # Contextual preferences
    time_of_day_active: List[int] = field(default_factory=list)  # Hours when user is active
    preferred_content_length: str = "medium"  # "short", "medium", "long"
    explanation_preference: bool = True  # Whether user values detailed explanations
    
    # Learning metadata
    total_interactions: int = 0
    last_updated: str = ""
    confidence_level: float = 0.0  # How confident we are in these preferences

@dataclass
class PersonalizedAction:
    """Extended action with personalization metadata"""
    base_action: GRPOAction
    personalization_score: float  # How much this was personalized vs. general
    user_match_confidence: float  # Confidence this matches user preferences
    learning_opportunity: bool = False  # Whether this is for exploration/learning
    explanation_detail: str = "medium"  # "brief", "medium", "detailed"

class PreferenceModel:
    """
    Machine learning model for user preference prediction
    Starts simple but can be enhanced with neural networks
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # Simple linear model parameters (can upgrade to neural network)
        self.feature_weights = {
            'sector_match': 1.0,
            'risk_alignment': 0.8,
            'sentiment_strength': 0.6,
            'confidence_level': 0.7,
            'timing_factor': 0.4,
            'recency_bias': 0.3
        }
        
        # Experience buffer for learning
        self.experience_buffer = deque(maxlen=1000)
        self.learning_rate = 0.1
        self.exploration_rate = 0.2  # For epsilon-greedy exploration
        
    def predict_user_interest(self, 
                            action: GRPOAction, 
                            user_profile: UserPreferenceProfile,
                            market_state: MarketState) -> float:
        """
        Predict how interested the user would be in this action
        
        Args:
            action: Proposed action
            user_profile: User's preference profile
            market_state: Current market state
            
        Returns:
            Interest score (0 to 1)
        """
        
        features = self._extract_features(action, user_profile, market_state)
        
        # Simple linear combination (can be replaced with neural network)
        interest_score = 0.0
        for feature_name, feature_value in features.items():
            weight = self.feature_weights.get(feature_name, 0.0)
            interest_score += weight * feature_value
        
        # Normalize to 0-1 range
        interest_score = np.clip(interest_score / sum(self.feature_weights.values()), 0, 1)
        
        return interest_score
    
    def _extract_features(self, 
                         action: GRPOAction, 
                         user_profile: UserPreferenceProfile,
                         market_state: MarketState) -> Dict[str, float]:
        """Extract features for prediction model"""
        
        features = {}
        
        # Sector match feature
        sector_pref = user_profile.sector_preferences.get(action.sector, 0.5)
        features['sector_match'] = sector_pref
        
        # Risk alignment feature
        risk_map = {"low": 0.2, "moderate": 0.5, "high": 0.8}
        action_risk = risk_map.get(action.risk_level, 0.5)
        risk_diff = abs(action_risk - user_profile.risk_preference)
        features['risk_alignment'] = 1.0 - risk_diff
        
        # Sentiment strength feature
        sentiment_strength = abs(market_state.sector_sentiments.get(action.sector, 0.0))
        features['sentiment_strength'] = sentiment_strength * user_profile.sentiment_sensitivity
        
        # Confidence level feature
        features['confidence_level'] = action.confidence
        
        # Timing factor (based on market volatility and user timing preference)
        timing_map = {"early": 0.8, "moderate": 0.5, "late": 0.2}
        timing_pref = timing_map.get(user_profile.timing_preference, 0.5)
        volatility = market_state.volatility
        features['timing_factor'] = timing_pref * (1 - volatility)  # Early movers like low volatility
        
        # Recency bias (prefer recent successful patterns)
        features['recency_bias'] = user_profile.success_rate
        
        return features
    
    def update_model(self, 
                    action: GRPOAction,
                    user_profile: UserPreferenceProfile,
                    market_state: MarketState,
                    feedback: UserFeedback):
        """
        Update model based on user feedback
        
        Args:
            action: Action that was recommended
            user_profile: User profile at time of recommendation
            market_state: Market state at time of recommendation
            feedback: User's feedback on the recommendation
        """
        
        # Store experience
        experience = {
            'features': self._extract_features(action, user_profile, market_state),
            'predicted_interest': self.predict_user_interest(action, user_profile, market_state),
            'actual_feedback': feedback.feedback_score,
            'timestamp': feedback.timestamp
        }
        self.experience_buffer.append(experience)
        
        # Simple gradient descent update
        if len(self.experience_buffer) >= 5:  # Need some examples to learn
            self._gradient_update(experience)
    
    def _gradient_update(self, experience: Dict):
        """Simple gradient descent update of feature weights"""
        
        features = experience['features']
        predicted = experience['predicted_interest']
        actual = (experience['actual_feedback'] + 1) / 2  # Convert -1,1 to 0,1
        
        error = actual - predicted
        
        # Update weights based on error
        for feature_name, feature_value in features.items():
            if feature_name in self.feature_weights:
                gradient = error * feature_value
                self.feature_weights[feature_name] += self.learning_rate * gradient
                
                # Keep weights positive
                self.feature_weights[feature_name] = max(0.1, self.feature_weights[feature_name])

class GRPOPAgent(Agent):
    """
    GRPO-P (GRPO-Personalization) Agent
    Learns individual user preferences through interaction feedback
    """
    
    def __init__(self, 
                 user_id: str,
                 sectors: List[str],
                 persistence_path: Optional[str] = None):
        """
        Initialize GRPO-P Agent for a specific user
        
        Args:
            user_id: Unique identifier for the user
            sectors: Available sectors for recommendations
            persistence_path: Path to save/load user preferences
        """
        
        self.user_id = user_id
        self.sectors = sectors
        self.persistence_path = persistence_path or f"user_profiles/{user_id}.json"
        
        # Load or initialize user preference profile
        self.user_profile = self._load_user_profile()
        
        # Initialize preference learning model
        self.preference_model = PreferenceModel(user_id)
        
        # Feedback and interaction tracking
        self.feedback_history: List[UserFeedback] = []
        self.recommendation_history: List[Tuple[PersonalizedAction, str]] = []  # (action, timestamp)
        
        # Learning parameters
        self.exploration_rate = 0.3  # Start with high exploration
        self.min_exploration_rate = 0.05
        self.exploration_decay = 0.995
        
        # Cold start handling
        self.is_cold_start = self.user_profile.total_interactions < 10
        
        logger.info(f"Initialized GRPO-P Agent for user {user_id}")
        logger.info(f"Cold start mode: {self.is_cold_start}")
        logger.info(f"Total interactions: {self.user_profile.total_interactions}")
    
    def get_action(self, state: MarketState) -> PersonalizedAction:
        """
        Get personalized action based on learned user preferences
        
        Args:
            state: Current market state
            
        Returns:
            Personalized action recommendation
        """
        
        if self.is_cold_start:
            return self._cold_start_recommendation(state)
        else:
            return self._personalized_recommendation(state)
    
    def _cold_start_recommendation(self, state: MarketState) -> PersonalizedAction:
        """
        Handle cold start scenario when we don't know user preferences yet
        Uses exploration-heavy strategy
        """
        
        # Create diverse recommendations to learn user preferences
        best_sector = self._select_exploration_sector(state)
        
        # Generate action for exploration
        sentiment = state.sector_sentiments.get(best_sector, 0.0)
        
        # More aggressive recommendations to get clear feedback
        if sentiment > 0.2:
            rec_type = "invest"
            confidence = 0.7  # Medium confidence for exploration
        elif sentiment < -0.2:
            rec_type = "avoid"
            confidence = 0.6
        else:
            rec_type = "monitor"
            confidence = 0.5
        
        base_action = GRPOAction(
            sector=best_sector,
            recommendation_type=rec_type,
            confidence=confidence,
            reasoning=f"Exploration recommendation for {best_sector} (learning your preferences)",
            risk_level="moderate"
        )
        
        return PersonalizedAction(
            base_action=base_action,
            personalization_score=0.2,  # Low personalization during cold start
            user_match_confidence=0.3,
            learning_opportunity=True,
            explanation_detail="detailed"  # Provide more explanation during cold start
        )
    
    def _personalized_recommendation(self, state: MarketState) -> PersonalizedAction:
        """
        Generate recommendation based on learned user preferences
        """
        
        # Epsilon-greedy: exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            return self._exploration_recommendation(state)
        else:
            return self._exploitation_recommendation(state)
    
    def _exploration_recommendation(self, state: MarketState) -> PersonalizedAction:
        """Generate exploration recommendation to learn more about user"""
        
        # Select sector we know least about
        exploration_sector = self._select_exploration_sector(state)
        
        sentiment = state.sector_sentiments.get(exploration_sector, 0.0)
        
        # Generate moderate recommendation for exploration
        if sentiment > 0.1:
            rec_type = "invest"
            confidence = 0.6
        elif sentiment < -0.1:
            rec_type = "avoid"
            confidence = 0.5
        else:
            rec_type = "monitor"
            confidence = 0.4
        
        base_action = GRPOAction(
            sector=exploration_sector,
            recommendation_type=rec_type,
            confidence=confidence,
            reasoning=f"Exploring opportunities in {exploration_sector} to refine your preferences",
            risk_level="moderate"
        )
        
        return PersonalizedAction(
            base_action=base_action,
            personalization_score=0.4,
            user_match_confidence=0.5,
            learning_opportunity=True,
            explanation_detail=self._get_preferred_explanation_style()
        )
    
    def _exploitation_recommendation(self, state: MarketState) -> PersonalizedAction:
        """Generate recommendation based on known user preferences"""
        
        # Find best sector based on user preferences and market state
        sector_scores = {}
        for sector in self.sectors:
            if sector in state.sector_sentiments:
                # Base market score
                market_score = state.sector_sentiments[sector]
                
                # User preference weight
                user_pref = self.user_profile.sector_preferences.get(sector, 0.5)
                
                # Risk adjustment
                risk_factor = self._calculate_risk_factor(sector, state)
                
                # Combined score
                sector_scores[sector] = market_score * user_pref * risk_factor
        
        if not sector_scores:
            return self._cold_start_recommendation(state)
        
        best_sector = max(sector_scores, key=sector_scores.get)
        best_score = sector_scores[best_sector]
        
        # Generate action based on user's risk preference and market score
        if best_score > 0.3 and self.user_profile.risk_preference > 0.4:
            rec_type = "invest"
            confidence = min(0.9, abs(best_score) + 0.3)
        elif best_score < -0.3 and self.user_profile.risk_preference > 0.6:
            rec_type = "avoid"
            confidence = min(0.8, abs(best_score) + 0.2)
        else:
            rec_type = "monitor"
            confidence = 0.6
        
        # Determine risk level based on user preference
        if self.user_profile.risk_preference > 0.7:
            risk_level = "high"
        elif self.user_profile.risk_preference > 0.4:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        reasoning = f"""Personalized recommendation for {best_sector}:
        - Your preference for {best_sector}: {self.user_profile.sector_preferences.get(best_sector, 0.5):.2f}
        - Your risk preference: {self.user_profile.risk_preference:.2f}
        - Current market sentiment: {state.sector_sentiments.get(best_sector, 0.0):.2f}
        - Based on {self.user_profile.total_interactions} previous interactions"""
        
        base_action = GRPOAction(
            sector=best_sector,
            recommendation_type=rec_type,
            confidence=confidence,
            reasoning=reasoning,
            risk_level=risk_level
        )
        
        # Calculate user match confidence
        predicted_interest = self.preference_model.predict_user_interest(
            base_action, self.user_profile, state
        )
        
        return PersonalizedAction(
            base_action=base_action,
            personalization_score=0.8,  # High personalization
            user_match_confidence=predicted_interest,
            learning_opportunity=False,
            explanation_detail=self._get_preferred_explanation_style()
        )
    
    def _select_exploration_sector(self, state: MarketState) -> str:
        """Select sector for exploration (least known about)"""
        
        # Calculate uncertainty for each sector
        sector_uncertainty = {}
        for sector in self.sectors:
            if sector in state.sector_sentiments:
                # Count interactions with this sector
                sector_interactions = sum(1 for action, _ in self.recommendation_history 
                                        if action.base_action.sector == sector)
                
                # Higher uncertainty = fewer interactions
                uncertainty = 1.0 / (1.0 + sector_interactions)
                sector_uncertainty[sector] = uncertainty
        
        if not sector_uncertainty:
            return np.random.choice(self.sectors)
        
        # Select sector with highest uncertainty
        return max(sector_uncertainty, key=sector_uncertainty.get)
    
    def _calculate_risk_factor(self, sector: str, state: MarketState) -> float:
        """Calculate risk adjustment factor based on user's risk preference"""
        
        market_volatility = state.volatility
        sentiment_strength = abs(state.sector_sentiments.get(sector, 0.0))
        
        # Higher risk users prefer higher volatility and stronger sentiments
        risk_factor = (
            self.user_profile.risk_preference * market_volatility +
            (1 - self.user_profile.risk_preference) * (1 - market_volatility) +
            self.user_profile.sentiment_sensitivity * sentiment_strength
        ) / 2
        
        return np.clip(risk_factor, 0.1, 1.0)
    
    def _get_preferred_explanation_style(self) -> str:
        """Get user's preferred explanation detail level"""
        
        if self.user_profile.explanation_preference:
            if self.user_profile.total_interactions < 5:
                return "detailed"  # New users get detailed explanations
            else:
                return self.user_profile.preferred_content_length
        else:
            return "brief"
    
    def update(self, state: MarketState, action: PersonalizedAction, feedback: UserFeedback):
        """
        Update agent based on user feedback
        
        Args:
            state: Market state when action was recommended
            action: Action that was recommended
            feedback: User's feedback on the recommendation
        """
        
        # Store feedback
        self.feedback_history.append(feedback)
        
        # Update preference model
        self.preference_model.update_model(
            action.base_action, self.user_profile, state, feedback
        )
        
        # Update user profile based on feedback
        self._update_user_profile(action, feedback)
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
        
        # Update cold start status
        if self.user_profile.total_interactions >= 10:
            self.is_cold_start = False
        
        # Save updated profile
        self._save_user_profile()
        
        logger.info(f"Updated GRPO-P agent for user {self.user_id}")
        logger.info(f"Total interactions: {self.user_profile.total_interactions}")
        logger.info(f"Exploration rate: {self.exploration_rate:.3f}")
    
    def _update_user_profile(self, action: PersonalizedAction, feedback: UserFeedback):
        """Update user preference profile based on feedback"""
        
        # Update sector preferences
        sector = action.base_action.sector
        current_pref = self.user_profile.sector_preferences.get(sector, 0.5)
        
        # Learning rate decreases with more interactions
        learning_rate = 0.2 / (1 + self.user_profile.total_interactions * 0.01)
        
        # Update based on feedback
        feedback_normalized = (feedback.feedback_score + 1) / 2  # Convert -1,1 to 0,1
        new_pref = current_pref + learning_rate * (feedback_normalized - current_pref)
        self.user_profile.sector_preferences[sector] = np.clip(new_pref, 0.0, 1.0)
        
        # Update risk preference based on action taken and feedback
        action_risk_map = {"low": 0.2, "moderate": 0.5, "high": 0.8}
        action_risk = action_risk_map.get(action.base_action.risk_level, 0.5)
        
        if feedback.feedback_score > 0:
            # Positive feedback - move towards this risk level
            risk_diff = action_risk - self.user_profile.risk_preference
            self.user_profile.risk_preference += learning_rate * risk_diff * 0.5
        
        # Update sentiment sensitivity
        if abs(action.base_action.confidence - 0.5) > 0.2:  # Strong confidence action
            if feedback.feedback_score > 0:
                # User liked confident recommendation
                self.user_profile.sentiment_sensitivity = min(1.0, 
                    self.user_profile.sentiment_sensitivity + learning_rate * 0.1)
        
        # Update behavioral patterns
        action_type = feedback.action_taken
        self.user_profile.feedback_pattern[action_type] = \
            self.user_profile.feedback_pattern.get(action_type, 0) + 1
        
        # Update success rate
        positive_feedback = sum(1 for f in self.feedback_history if f.feedback_score > 0)
        self.user_profile.success_rate = positive_feedback / len(self.feedback_history)
        
        # Update metadata
        self.user_profile.total_interactions += 1
        self.user_profile.last_updated = datetime.now().isoformat()
        
        # Update confidence in preferences
        if self.user_profile.total_interactions >= 20:
            self.user_profile.confidence_level = min(0.9, 
                self.user_profile.total_interactions / 100.0)
        else:
            self.user_profile.confidence_level = self.user_profile.total_interactions / 20.0
    
    def _load_user_profile(self) -> UserPreferenceProfile:
        """Load user profile from persistence storage"""
        
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                
                profile = UserPreferenceProfile(user_id=self.user_id)
                
                # Load preferences
                profile.sector_preferences = data.get('sector_preferences', {})
                profile.risk_preference = data.get('risk_preference', 0.5)
                profile.sentiment_sensitivity = data.get('sentiment_sensitivity', 0.5)
                profile.timing_preference = data.get('timing_preference', 'moderate')
                
                # Load behavioral patterns
                profile.interaction_frequency = data.get('interaction_frequency', 0.0)
                profile.feedback_pattern = data.get('feedback_pattern', {})
                profile.success_rate = data.get('success_rate', 0.0)
                
                # Load contextual preferences
                profile.time_of_day_active = data.get('time_of_day_active', [])
                profile.preferred_content_length = data.get('preferred_content_length', 'medium')
                profile.explanation_preference = data.get('explanation_preference', True)
                
                # Load metadata
                profile.total_interactions = data.get('total_interactions', 0)
                profile.last_updated = data.get('last_updated', '')
                profile.confidence_level = data.get('confidence_level', 0.0)
                
                logger.info(f"Loaded user profile for {self.user_id}")
                return profile
                
            except Exception as e:
                logger.error(f"Error loading user profile: {e}")
        
        # Return new profile if loading failed
        logger.info(f"Creating new user profile for {self.user_id}")
        return UserPreferenceProfile(user_id=self.user_id)
    
    def _save_user_profile(self):
        """Save user profile to persistence storage"""
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            
            # Convert profile to dictionary
            data = {
                'sector_preferences': self.user_profile.sector_preferences,
                'risk_preference': self.user_profile.risk_preference,
                'sentiment_sensitivity': self.user_profile.sentiment_sensitivity,
                'timing_preference': self.user_profile.timing_preference,
                'interaction_frequency': self.user_profile.interaction_frequency,
                'feedback_pattern': self.user_profile.feedback_pattern,
                'success_rate': self.user_profile.success_rate,
                'time_of_day_active': self.user_profile.time_of_day_active,
                'preferred_content_length': self.user_profile.preferred_content_length,
                'explanation_preference': self.user_profile.explanation_preference,
                'total_interactions': self.user_profile.total_interactions,
                'last_updated': self.user_profile.last_updated,
                'confidence_level': self.user_profile.confidence_level
            }
            
            with open(self.persistence_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved user profile for {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")
    
    def get_user_profile_summary(self) -> Dict[str, Any]:
        """Get summary of learned user preferences"""
        
        return {
            'user_id': self.user_id,
            'total_interactions': self.user_profile.total_interactions,
            'confidence_level': self.user_profile.confidence_level,
            'is_cold_start': self.is_cold_start,
            'exploration_rate': self.exploration_rate,
            'sector_preferences': self.user_profile.sector_preferences,
            'risk_preference': self.user_profile.risk_preference,
            'success_rate': self.user_profile.success_rate,
            'most_common_action': max(self.user_profile.feedback_pattern.items(), 
                                    key=lambda x: x[1]) if self.user_profile.feedback_pattern else None
        }

# Example usage and testing
if __name__ == "__main__":
    print("🧠 Testing GRPO-P Agent Implementation")
    print("=" * 50)
    
    # Create test market state
    market_state = MarketState(
        sector_sentiments={"technology": 0.6, "finance": -0.3, "energy": 0.2},
        overall_sentiment=0.2,
        volatility=0.4,
        trending_topics={"AI": 8, "crypto": 5, "renewable": 6},
        article_count=45,
        timestamp="2024-01-01"
    )
    
    # Test new user (cold start)
    new_user_agent = GRPOPAgent(
        user_id="test_user_new",
        sectors=["technology", "finance", "energy"]
    )
    
    print("Testing Cold Start User:")
    action = new_user_agent.get_action(market_state)
    print(f"Sector: {action.base_action.sector}")
    print(f"Type: {action.base_action.recommendation_type}")
    print(f"Personalization Score: {action.personalization_score:.2f}")
    print(f"Is Learning Opportunity: {action.learning_opportunity}")
    
    # Simulate user feedback
    feedback = UserFeedback(
        user_id="test_user_new",
        recommendation_id="rec_1",
        action_taken="clicked",
        feedback_score=0.8,
        timestamp=datetime.now().isoformat()
    )
    
    new_user_agent.update(market_state, action, feedback)
    
    print(f"\nAfter Feedback:")
    summary = new_user_agent.get_user_profile_summary()
    print(f"Total Interactions: {summary['total_interactions']}")
    print(f"Confidence Level: {summary['confidence_level']:.2f}")
    print(f"Sector Preferences: {summary['sector_preferences']}")
    
    print("\n✅ GRPO-P Agent implementation completed!")
    print("Ready for integration with GRPO population and arbitration!")
