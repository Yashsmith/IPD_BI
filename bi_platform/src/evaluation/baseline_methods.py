"""
Baseline Recommendation Methods for Comparative Research
Implements standard baselines to compare against GRPO-GRPO-P framework
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, Counter
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.agents.grpo_agent import MarketState, GRPOAction

logger = logging.getLogger(__name__)

class BaselineRecommender(ABC):
    """
    Abstract base class for all baseline recommendation methods
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        
    @abstractmethod
    def fit(self, user_item_matrix: np.ndarray, item_features: Optional[np.ndarray] = None):
        """Train the recommendation model"""
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """Generate recommendations for a user"""
        pass
    
    def get_name(self) -> str:
        return self.name

class PopularityBaseline(BaselineRecommender):
    """
    Popularity-based recommendation baseline
    Recommends most popular items across all users
    """
    
    def __init__(self):
        super().__init__("Popularity")
        self.item_popularity = {}
        self.sorted_items = []
    
    def fit(self, user_item_matrix: np.ndarray, item_features: Optional[np.ndarray] = None):
        """
        Train popularity baseline
        
        Args:
            user_item_matrix: User-item interaction matrix (users x items)
            item_features: Not used for popularity baseline
        """
        
        # Calculate item popularity (sum of interactions)
        item_scores = np.sum(user_item_matrix, axis=0)
        
        # Store popularity scores
        self.item_popularity = {i: score for i, score in enumerate(item_scores)}
        
        # Sort items by popularity
        self.sorted_items = sorted(self.item_popularity.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        self.is_trained = True
        logger.info("PopularityBaseline trained successfully")
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Recommend top popular items
        
        Args:
            user_id: User ID (not used in popularity baseline)
            n_recommendations: Number of recommendations to generate
            
        Returns:
            List of (item_id, score) tuples
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Return top N popular items
        return self.sorted_items[:n_recommendations]

class RandomBaseline(BaselineRecommender):
    """
    Random recommendation baseline
    Provides random recommendations as a sanity check
    """
    
    def __init__(self, random_seed: int = 42):
        super().__init__("Random")
        self.random_seed = random_seed
        self.n_items = 0
        np.random.seed(random_seed)
    
    def fit(self, user_item_matrix: np.ndarray, item_features: Optional[np.ndarray] = None):
        """
        Fit random baseline (just store number of items)
        
        Args:
            user_item_matrix: User-item interaction matrix
            item_features: Not used
        """
        self.n_items = user_item_matrix.shape[1]
        self.is_trained = True
        logger.info("RandomBaseline fitted successfully")
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Generate random recommendations
        
        Args:
            user_id: User ID (not used)
            n_recommendations: Number of recommendations
            
        Returns:
            List of random (item_id, score) tuples
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Generate random item indices
        random_items = np.random.choice(self.n_items, 
                                      size=min(n_recommendations, self.n_items), 
                                      replace=False)
        
        # Generate random scores
        random_scores = np.random.random(len(random_items))
        
        return list(zip(random_items, random_scores))

class UserBasedCollaborativeFiltering(BaselineRecommender):
    """
    User-based Collaborative Filtering baseline
    Recommends items liked by similar users
    """
    
    def __init__(self, similarity_threshold: float = 0.1, n_neighbors: int = 50):
        super().__init__("UserCF")
        self.similarity_threshold = similarity_threshold
        self.n_neighbors = n_neighbors
        self.user_item_matrix = None
        self.user_similarities = None
    
    def fit(self, user_item_matrix: np.ndarray, item_features: Optional[np.ndarray] = None):
        """
        Train user-based collaborative filtering
        
        Args:
            user_item_matrix: User-item interaction matrix
            item_features: Not used
        """
        
        self.user_item_matrix = user_item_matrix
        n_users = user_item_matrix.shape[0]
        
        # Calculate user-user similarities (cosine similarity)
        self.user_similarities = np.zeros((n_users, n_users))
        
        for i in range(n_users):
            for j in range(i + 1, n_users):
                user_i = user_item_matrix[i]
                user_j = user_item_matrix[j]
                
                # Cosine similarity
                norm_i = np.linalg.norm(user_i)
                norm_j = np.linalg.norm(user_j)
                
                if norm_i > 0 and norm_j > 0:
                    similarity = np.dot(user_i, user_j) / (norm_i * norm_j)
                    self.user_similarities[i, j] = similarity
                    self.user_similarities[j, i] = similarity
        
        self.is_trained = True
        logger.info("UserCF trained successfully")
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Generate recommendations using user-based CF
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        
        if not self.is_trained or user_id >= len(self.user_similarities):
            return []
        
        # Find similar users
        user_sims = self.user_similarities[user_id]
        similar_users = []
        
        for other_user, similarity in enumerate(user_sims):
            if similarity > self.similarity_threshold and other_user != user_id:
                similar_users.append((other_user, similarity))
        
        # Sort by similarity and take top neighbors
        similar_users.sort(key=lambda x: x[1], reverse=True)
        similar_users = similar_users[:self.n_neighbors]
        
        if not similar_users:
            return []
        
        # Calculate item scores based on similar users
        target_user_items = self.user_item_matrix[user_id]
        item_scores = defaultdict(float)
        similarity_sums = defaultdict(float)
        
        for other_user, similarity in similar_users:
            other_user_items = self.user_item_matrix[other_user]
            
            for item_id, rating in enumerate(other_user_items):
                if rating > 0 and target_user_items[item_id] == 0:  # Item not seen by target user
                    item_scores[item_id] += similarity * rating
                    similarity_sums[item_id] += similarity
        
        # Normalize scores and create recommendations
        recommendations = []
        for item_id, score in item_scores.items():
            if similarity_sums[item_id] > 0:
                normalized_score = score / similarity_sums[item_id]
                recommendations.append((item_id, normalized_score))
        
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

class ItemBasedCollaborativeFiltering(BaselineRecommender):
    """
    Item-based Collaborative Filtering baseline
    Recommends items similar to those the user has interacted with
    """
    
    def __init__(self, similarity_threshold: float = 0.1, n_neighbors: int = 50):
        super().__init__("ItemCF")
        self.similarity_threshold = similarity_threshold
        self.n_neighbors = n_neighbors
        self.user_item_matrix = None
        self.item_similarities = None
    
    def fit(self, user_item_matrix: np.ndarray, item_features: Optional[np.ndarray] = None):
        """
        Train item-based collaborative filtering
        
        Args:
            user_item_matrix: User-item interaction matrix
            item_features: Not used
        """
        
        self.user_item_matrix = user_item_matrix
        n_items = user_item_matrix.shape[1]
        
        # Calculate item-item similarities (cosine similarity)
        self.item_similarities = np.zeros((n_items, n_items))
        
        for i in range(n_items):
            for j in range(i + 1, n_items):
                item_i = user_item_matrix[:, i]
                item_j = user_item_matrix[:, j]
                
                # Cosine similarity
                norm_i = np.linalg.norm(item_i)
                norm_j = np.linalg.norm(item_j)
                
                if norm_i > 0 and norm_j > 0:
                    similarity = np.dot(item_i, item_j) / (norm_i * norm_j)
                    self.item_similarities[i, j] = similarity
                    self.item_similarities[j, i] = similarity
        
        self.is_trained = True
        logger.info("ItemCF trained successfully")
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Generate recommendations using item-based CF
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        
        if not self.is_trained or user_id >= self.user_item_matrix.shape[0]:
            return []
        
        user_items = self.user_item_matrix[user_id]
        item_scores = defaultdict(float)
        similarity_sums = defaultdict(float)
        
        # For each item the user has interacted with
        for item_id, rating in enumerate(user_items):
            if rating > 0:  # User has interacted with this item
                
                # Find similar items
                item_sims = self.item_similarities[item_id]
                similar_items = []
                
                for other_item, similarity in enumerate(item_sims):
                    if (similarity > self.similarity_threshold and 
                        other_item != item_id and 
                        user_items[other_item] == 0):  # User hasn't seen this item
                        similar_items.append((other_item, similarity))
                
                # Sort by similarity and take top neighbors
                similar_items.sort(key=lambda x: x[1], reverse=True)
                similar_items = similar_items[:self.n_neighbors]
                
                # Accumulate scores
                for other_item, similarity in similar_items:
                    item_scores[other_item] += similarity * rating
                    similarity_sums[other_item] += similarity
        
        # Normalize scores and create recommendations
        recommendations = []
        for item_id, score in item_scores.items():
            if similarity_sums[item_id] > 0:
                normalized_score = score / similarity_sums[item_id]
                recommendations.append((item_id, normalized_score))
        
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

class MatrixFactorizationBaseline(BaselineRecommender):
    """
    Matrix Factorization baseline using Alternating Least Squares (ALS)
    Simple implementation without external dependencies
    """
    
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01, 
                 regularization: float = 0.01, n_iterations: int = 100):
        super().__init__("MatrixFactorization")
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_iterations = n_iterations
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = 0.0
    
    def fit(self, user_item_matrix: np.ndarray, item_features: Optional[np.ndarray] = None):
        """
        Train matrix factorization model using gradient descent
        
        Args:
            user_item_matrix: User-item interaction matrix
            item_features: Not used
        """
        
        n_users, n_items = user_item_matrix.shape
        
        # Initialize factors randomly
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        
        # Calculate global mean of non-zero ratings
        nonzero_ratings = user_item_matrix[user_item_matrix > 0]
        self.global_mean = np.mean(nonzero_ratings) if len(nonzero_ratings) > 0 else 0.0
        
        # Get indices of non-zero entries
        user_indices, item_indices = np.nonzero(user_item_matrix)
        
        # Training loop
        for iteration in range(self.n_iterations):
            
            # Shuffle the order of training examples
            indices = np.random.permutation(len(user_indices))
            
            total_error = 0.0
            
            for idx in indices:
                user_id = user_indices[idx]
                item_id = item_indices[idx]
                rating = user_item_matrix[user_id, item_id]
                
                # Predict rating
                prediction = (self.global_mean + 
                            self.user_bias[user_id] + 
                            self.item_bias[item_id] + 
                            np.dot(self.user_factors[user_id], self.item_factors[item_id]))
                
                # Calculate error
                error = rating - prediction
                total_error += error ** 2
                
                # Update biases
                user_bias_old = self.user_bias[user_id]
                self.user_bias[user_id] += self.learning_rate * (error - self.regularization * self.user_bias[user_id])
                self.item_bias[item_id] += self.learning_rate * (error - self.regularization * self.item_bias[item_id])
                
                # Update factors
                user_factors_old = self.user_factors[user_id].copy()
                self.user_factors[user_id] += self.learning_rate * (error * self.item_factors[item_id] - 
                                                                  self.regularization * self.user_factors[user_id])
                self.item_factors[item_id] += self.learning_rate * (error * user_factors_old - 
                                                                  self.regularization * self.item_factors[item_id])
            
            # Print progress occasionally
            if iteration % 20 == 0:
                rmse = np.sqrt(total_error / len(indices))
                logger.info(f"MatrixFactorization iteration {iteration}, RMSE: {rmse:.4f}")
        
        self.is_trained = True
        logger.info("MatrixFactorization trained successfully")
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Generate recommendations using matrix factorization
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        
        if not self.is_trained or user_id >= len(self.user_factors):
            return []
        
        # Predict scores for all items
        user_factor = self.user_factors[user_id]
        user_bias_val = self.user_bias[user_id]
        
        item_scores = []
        for item_id in range(len(self.item_factors)):
            score = (self.global_mean + 
                    user_bias_val + 
                    self.item_bias[item_id] + 
                    np.dot(user_factor, self.item_factors[item_id]))
            item_scores.append((item_id, score))
        
        # Sort by score and return top N
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:n_recommendations]

class ContentBasedBaseline(BaselineRecommender):
    """
    Content-based recommendation baseline
    Uses item features to recommend similar items
    """
    
    def __init__(self, similarity_threshold: float = 0.1):
        super().__init__("ContentBased")
        self.similarity_threshold = similarity_threshold
        self.item_features = None
        self.item_similarities = None
        self.user_profiles = None
    
    def fit(self, user_item_matrix: np.ndarray, item_features: Optional[np.ndarray] = None):
        """
        Train content-based model
        
        Args:
            user_item_matrix: User-item interaction matrix
            item_features: Item feature matrix (items x features)
        """
        
        if item_features is None:
            # Create dummy features based on item popularity
            item_popularity = np.sum(user_item_matrix, axis=0)
            self.item_features = item_popularity.reshape(-1, 1)
        else:
            self.item_features = item_features
        
        n_items = self.item_features.shape[0]
        
        # Calculate item-item similarities based on features
        self.item_similarities = np.zeros((n_items, n_items))
        
        for i in range(n_items):
            for j in range(i + 1, n_items):
                # Cosine similarity between feature vectors
                feat_i = self.item_features[i]
                feat_j = self.item_features[j]
                
                norm_i = np.linalg.norm(feat_i)
                norm_j = np.linalg.norm(feat_j)
                
                if norm_i > 0 and norm_j > 0:
                    similarity = np.dot(feat_i, feat_j) / (norm_i * norm_j)
                    self.item_similarities[i, j] = similarity
                    self.item_similarities[j, i] = similarity
        
        # Build user profiles based on their interaction history
        n_users = user_item_matrix.shape[0]
        n_features = self.item_features.shape[1]
        self.user_profiles = np.zeros((n_users, n_features))
        
        for user_id in range(n_users):
            user_interactions = user_item_matrix[user_id]
            total_interactions = np.sum(user_interactions)
            
            if total_interactions > 0:
                # Weighted average of item features
                for item_id, rating in enumerate(user_interactions):
                    if rating > 0:
                        weight = rating / total_interactions
                        self.user_profiles[user_id] += weight * self.item_features[item_id]
        
        self.is_trained = True
        logger.info("ContentBased trained successfully")
    
    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Generate content-based recommendations
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        
        if not self.is_trained or user_id >= len(self.user_profiles):
            return []
        
        user_profile = self.user_profiles[user_id]
        item_scores = []
        
        # Calculate similarity between user profile and each item
        for item_id in range(len(self.item_features)):
            item_features = self.item_features[item_id]
            
            # Cosine similarity
            norm_user = np.linalg.norm(user_profile)
            norm_item = np.linalg.norm(item_features)
            
            if norm_user > 0 and norm_item > 0:
                similarity = np.dot(user_profile, item_features) / (norm_user * norm_item)
                if similarity > self.similarity_threshold:
                    item_scores.append((item_id, similarity))
        
        # Sort by score and return top N
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:n_recommendations]

# Factory function to create all baselines
def create_all_baselines() -> Dict[str, BaselineRecommender]:
    """
    Create all baseline recommendation methods for comparison
    
    Returns:
        Dictionary mapping method names to baseline recommender objects
    """
    
    baselines = {
        'PopularityBaseline': PopularityBaseline(),
        'RandomBaseline': RandomBaseline(),
        'UserCollaborativeFiltering': UserBasedCollaborativeFiltering(),
        'ItemCollaborativeFiltering': ItemBasedCollaborativeFiltering(),
        'MatrixFactorization': MatrixFactorizationBaseline(),
        'ContentBasedFiltering': ContentBasedBaseline()
    }
    
    return baselines

# Example usage and testing
if __name__ == "__main__":
    print("🎯 Testing Baseline Recommendation Methods")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_users, n_items = 100, 50
    
    # Generate synthetic user-item matrix (sparse)
    user_item_matrix = np.random.choice([0, 1, 2, 3, 4, 5], 
                                       size=(n_users, n_items), 
                                       p=[0.85, 0.03, 0.03, 0.03, 0.03, 0.03])
    
    # Generate synthetic item features
    item_features = np.random.random((n_items, 10))
    
    print(f"📊 Dataset: {n_users} users, {n_items} items")
    print(f"Sparsity: {100 * np.mean(user_item_matrix == 0):.1f}%")
    
    # Test all baselines
    baselines = create_all_baselines()
    
    for baseline_name, baseline in baselines.items():
        print(f"\n🧪 Testing {baseline_name}...")
        
        # Train baseline
        start_time = datetime.now()
        baseline.fit(user_item_matrix, item_features)
        train_time = (datetime.now() - start_time).total_seconds()
        
        # Generate recommendations for first few users
        test_user = 0
        recommendations = baseline.recommend(test_user, n_recommendations=5)
        
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Sample recommendations for user {test_user}:")
        for item_id, score in recommendations:
            print(f"     Item {item_id}: {score:.3f}")
    
    print("\n✅ All baseline methods implemented and tested!")
    print("Ready for comparative evaluation against GRPO-GRPO-P framework!")
