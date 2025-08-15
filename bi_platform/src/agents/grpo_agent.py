"""
Basic GRPO (Group Relative Policy Optimization) Agent
Implements group-consensus based recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class MarketState:
    """Represents current market state"""
    sector_sentiments: Dict[str, float]  # Sentiment by sector
    overall_sentiment: float  # Overall market sentiment
    volatility: float  # Market volatility measure
    trending_topics: Dict[str, int]  # Trending topics and frequency
    article_count: int  # Number of articles analyzed
    timestamp: str  # When this state was computed

@dataclass 
class GRPOAction:
    """Represents a GRPO agent action/recommendation"""
    sector: str
    recommendation_type: str  # "invest", "avoid", "monitor"
    confidence: float  # 0 to 1
    reasoning: str
    risk_level: str  # "low", "moderate", "high"

class Agent(ABC):
    """Abstract base class for all agents"""
    
    @abstractmethod
    def get_action(self, state: MarketState) -> GRPOAction:
        """Get action given current state"""
        pass
    
    @abstractmethod
    def update(self, state: MarketState, action: GRPOAction, reward: float):
        """Update agent based on reward"""
        pass

class GRPOAgent(Agent):
    """
    Group Relative Policy Optimization Agent
    Makes recommendations based on group consensus and market alignment
    """
    
    def __init__(self, 
                 agent_id: str,
                 sectors: List[str],
                 conservative_bias: float = 0.7,
                 learning_rate: float = 0.01):
        """
        Initialize GRPO Agent
        
        Args:
            agent_id: Unique identifier for this agent
            sectors: List of sectors this agent can recommend
            conservative_bias: How conservative the agent is (0-1, higher = more conservative)
            learning_rate: Learning rate for updating policies
        """
        self.agent_id = agent_id
        self.sectors = sectors
        self.conservative_bias = conservative_bias
        self.learning_rate = learning_rate
        
        # Policy parameters (simple approach - could be neural network)
        self.sector_weights = {sector: 1.0 for sector in sectors}
        self.sentiment_threshold = 0.1  # Minimum sentiment to recommend "invest"
        self.risk_threshold = 0.5  # Threshold for high risk classification
        
        # Experience tracking
        self.action_history = []
        self.performance_history = []
        self.group_performance_history = []
        
        logger.info(f"Initialized GRPO Agent {agent_id} for sectors: {sectors}")
    
    def get_action(self, state: MarketState) -> GRPOAction:
        """
        Get recommended action based on current market state
        
        Args:
            state: Current market state
            
        Returns:
            Recommended action
        """
        
        # Calculate sector scores based on sentiment and group consensus
        sector_scores = {}
        
        for sector in self.sectors:
            if sector in state.sector_sentiments:
                sentiment = state.sector_sentiments[sector]
                
                # Apply conservative bias - reduce extreme positions
                adjusted_sentiment = sentiment * (1 - self.conservative_bias) + self.conservative_bias * 0.0
                
                # Weight by sector preference
                sector_weight = self.sector_weights.get(sector, 1.0)
                
                # Calculate final score
                sector_scores[sector] = adjusted_sentiment * sector_weight
            else:
                sector_scores[sector] = 0.0
        
        # Select best sector
        if not sector_scores:
            # Fallback if no sectors available
            selected_sector = self.sectors[0] if self.sectors else "general"
            score = 0.0
        else:
            selected_sector = max(sector_scores, key=sector_scores.get)
            score = sector_scores[selected_sector]
        
        # Determine recommendation type based on score
        if score > self.sentiment_threshold:
            rec_type = "invest"
            confidence = min(0.9, abs(score) * (1 - self.conservative_bias) + 0.3)
        elif score < -self.sentiment_threshold:
            rec_type = "avoid"
            confidence = min(0.8, abs(score) * (1 - self.conservative_bias) + 0.2)
        else:
            rec_type = "monitor"
            confidence = 0.5
        
        # Determine risk level
        if abs(score) > self.risk_threshold and state.volatility > 0.5:
            risk_level = "high"
        elif abs(score) > 0.3 or state.volatility > 0.3:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        # Generate reasoning
        reasoning = self._generate_reasoning(selected_sector, score, state, rec_type)
        
        action = GRPOAction(
            sector=selected_sector,
            recommendation_type=rec_type,
            confidence=confidence,
            reasoning=reasoning,
            risk_level=risk_level
        )
        
        # Store action in history
        self.action_history.append((state, action))
        
        return action
    
    def _generate_reasoning(self, sector: str, score: float, state: MarketState, rec_type: str) -> str:
        """Generate human-readable reasoning for the recommendation"""
        
        sentiment_desc = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        volatility_desc = "high" if state.volatility > 0.5 else "moderate" if state.volatility > 0.3 else "low"
        
        reasoning = f"""
        GRPO Analysis for {sector}:
        - Market sentiment: {sentiment_desc} ({score:.2f})
        - Overall market volatility: {volatility_desc} ({state.volatility:.2f})
        - Group consensus approach: conservative bias applied
        - Based on {state.article_count} recent articles
        - Recommendation: {rec_type.upper()} with {volatility_desc} risk
        """.strip()
        
        return reasoning
    
    def update(self, state: MarketState, action: GRPOAction, reward: float):
        """
        Update agent based on observed reward
        
        Args:
            state: State when action was taken
            action: Action that was taken
            reward: Observed reward (could be user feedback, market performance, etc.)
        """
        
        # Store performance
        self.performance_history.append(reward)
        
        # Simple policy update based on reward
        if reward > 0:
            # Positive reward - increase weight for this sector
            if action.sector in self.sector_weights:
                self.sector_weights[action.sector] += self.learning_rate * reward
        else:
            # Negative reward - decrease weight for this sector
            if action.sector in self.sector_weights:
                self.sector_weights[action.sector] -= self.learning_rate * abs(reward)
        
        # Ensure weights stay positive
        for sector in self.sector_weights:
            self.sector_weights[sector] = max(0.1, self.sector_weights[sector])
        
        # Adapt thresholds based on performance
        if len(self.performance_history) > 10:
            recent_performance = np.mean(self.performance_history[-10:])
            if recent_performance < 0:
                # Poor performance - become more conservative
                self.conservative_bias = min(0.9, self.conservative_bias + 0.05)
                self.sentiment_threshold = min(0.3, self.sentiment_threshold + 0.02)
            elif recent_performance > 0.5:
                # Good performance - become slightly less conservative
                self.conservative_bias = max(0.3, self.conservative_bias - 0.02)
                self.sentiment_threshold = max(0.05, self.sentiment_threshold - 0.01)
    
    def update_group_performance(self, group_performance: List[float]):
        """
        Update agent based on group performance (key part of GRPO)
        
        Args:
            group_performance: Performance of all agents in the population
        """
        
        self.group_performance_history.append(group_performance)
        
        if len(self.performance_history) > 0 and len(group_performance) > 0:
            my_performance = self.performance_history[-1]
            group_avg = np.mean(group_performance)
            
            # GRPO update: adjust based on relative performance
            relative_performance = my_performance - group_avg
            
            if relative_performance < 0:
                # Performing worse than group - become more conservative
                self.conservative_bias = min(0.95, self.conservative_bias + 0.1)
            else:
                # Performing better than group - maintain or slightly reduce conservatism
                self.conservative_bias = max(0.3, self.conservative_bias - 0.05)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for this agent"""
        
        if not self.performance_history:
            return {"avg_performance": 0.0, "total_actions": 0}
        
        return {
            "avg_performance": np.mean(self.performance_history),
            "total_actions": len(self.performance_history),
            "current_conservative_bias": self.conservative_bias,
            "best_performance": max(self.performance_history),
            "worst_performance": min(self.performance_history),
            "sector_weights": self.sector_weights.copy()
        }

class GRPOPopulation:
    """
    Manages a population of GRPO agents for group optimization
    """
    
    def __init__(self, 
                 population_size: int = 10,
                 sectors: List[str] = None):
        """
        Initialize GRPO population
        
        Args:
            population_size: Number of agents in population
            sectors: Available sectors for agents
        """
        
        if sectors is None:
            sectors = ["technology", "finance", "energy", "healthcare"]
        
        self.sectors = sectors
        self.population_size = population_size
        
        # Create diverse population of agents
        self.agents = []
        for i in range(population_size):
            # Create agents with different characteristics
            agent_sectors = np.random.choice(sectors, size=min(3, len(sectors)), replace=False).tolist()
            conservative_bias = 0.3 + (i / population_size) * 0.6  # Range from 0.3 to 0.9
            
            agent = GRPOAgent(
                agent_id=f"grpo_agent_{i}",
                sectors=agent_sectors,
                conservative_bias=conservative_bias,
                learning_rate=0.01
            )
            self.agents.append(agent)
        
        logger.info(f"Created GRPO population with {population_size} agents")
    
    def get_group_recommendations(self, state: MarketState) -> List[GRPOAction]:
        """
        Get recommendations from all agents in the population
        
        Args:
            state: Current market state
            
        Returns:
            List of actions from all agents
        """
        
        recommendations = []
        for agent in self.agents:
            try:
                action = agent.get_action(state)
                recommendations.append(action)
            except Exception as e:
                logger.error(f"Error getting action from agent {agent.agent_id}: {e}")
        
        return recommendations
    
    def get_consensus_recommendation(self, state: MarketState) -> GRPOAction:
        """
        Get consensus recommendation from the population
        
        Args:
            state: Current market state
            
        Returns:
            Consensus recommendation
        """
        
        recommendations = self.get_group_recommendations(state)
        
        if not recommendations:
            # Fallback recommendation
            return GRPOAction(
                sector="general",
                recommendation_type="monitor",
                confidence=0.5,
                reasoning="No agent recommendations available",
                risk_level="moderate"
            )
        
        # Simple consensus: most common sector and recommendation type
        sectors = [rec.sector for rec in recommendations]
        rec_types = [rec.recommendation_type for rec in recommendations]
        
        from collections import Counter
        most_common_sector = Counter(sectors).most_common(1)[0][0]
        most_common_type = Counter(rec_types).most_common(1)[0][0]
        
        # Average confidence and aggregate reasoning
        avg_confidence = np.mean([rec.confidence for rec in recommendations])
        
        consensus_reasoning = f"""
        Group Consensus (based on {len(recommendations)} agents):
        - Most recommended sector: {most_common_sector}
        - Consensus action: {most_common_type}
        - Average confidence: {avg_confidence:.2f}
        - Population agreement: {Counter(rec_types)[most_common_type]/len(recommendations):.1%}
        """
        
        return GRPOAction(
            sector=most_common_sector,
            recommendation_type=most_common_type,
            confidence=avg_confidence,
            reasoning=consensus_reasoning,
            risk_level="moderate"  # Consensus tends to be moderate risk
        )
    
    def update_population(self, state: MarketState, rewards: List[float]):
        """
        Update all agents based on their rewards
        
        Args:
            state: State when actions were taken
            rewards: List of rewards for each agent
        """
        
        if len(rewards) != len(self.agents):
            logger.error(f"Reward count ({len(rewards)}) doesn't match agent count ({len(self.agents)})")
            return
        
        # Update each agent individually
        for agent, reward in zip(self.agents, rewards):
            if agent.action_history:
                last_state, last_action = agent.action_history[-1]
                agent.update(last_state, last_action, reward)
        
        # Update all agents with group performance (key GRPO step)
        for agent in self.agents:
            agent.update_group_performance(rewards)
    
    def get_population_stats(self) -> Dict[str, Any]:
        """Get statistics for the entire population"""
        
        all_performances = []
        for agent in self.agents:
            stats = agent.get_performance_stats()
            if stats["total_actions"] > 0:
                all_performances.append(stats["avg_performance"])
        
        if not all_performances:
            return {"population_performance": 0.0, "active_agents": 0}
        
        return {
            "population_performance": np.mean(all_performances),
            "active_agents": len(all_performances),
            "best_agent_performance": max(all_performances),
            "worst_agent_performance": min(all_performances),
            "performance_std": np.std(all_performances)
        }

# Example usage and testing
if __name__ == "__main__":
    print("🤖 Testing GRPO Agent Implementation")
    print("=" * 50)
    
    # Create sample market state
    market_state = MarketState(
        sector_sentiments={"technology": 0.4, "finance": -0.2, "energy": 0.1},
        overall_sentiment=0.1,
        volatility=0.3,
        trending_topics={"AI": 5, "blockchain": 3, "renewable": 4},
        article_count=25,
        timestamp="2024-01-01"
    )
    
    # Test single agent
    agent = GRPOAgent(
        agent_id="test_agent",
        sectors=["technology", "finance", "energy"],
        conservative_bias=0.6
    )
    
    action = agent.get_action(market_state)
    print(f"Single Agent Recommendation:")
    print(f"Sector: {action.sector}")
    print(f"Type: {action.recommendation_type}")
    print(f"Confidence: {action.confidence:.2f}")
    print(f"Risk: {action.risk_level}")
    
    # Test population
    print(f"\nTesting GRPO Population...")
    population = GRPOPopulation(population_size=5)
    consensus = population.get_consensus_recommendation(market_state)
    
    print(f"Population Consensus:")
    print(f"Sector: {consensus.sector}")
    print(f"Type: {consensus.recommendation_type}")
    print(f"Confidence: {consensus.confidence:.2f}")
    
    print("\n✅ GRPO Agent implementation completed!")
    print("Ready for integration with news data and recommendations!")
