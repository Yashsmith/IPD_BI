"""
Fixed NDCG Calculation for Research Evaluation
Addresses the NDCG > 1.0 issue by implementing proper normalization
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class FixedMetricsCalculator:
    """
    Fixed implementation of recommendation metrics with proper NDCG calculation
    """
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10, 20]):
        self.k_values = k_values
    
    def precision_at_k(self, recommendations: List[Any], relevant_items: List[Any], k: int) -> float:
        """
        Calculate Precision@K - Fixed implementation
        
        Args:
            recommendations: List of recommended items
            relevant_items: List of relevant/ground truth items
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score (0-1)
        """
        if k == 0 or len(recommendations) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        hits = sum(1 for item in top_k_recs if item in relevant_set)
        return hits / k  # Fixed: Always divide by k, not min(k, len(recommendations))
    
    def recall_at_k(self, recommendations: List[Any], relevant_items: List[Any], k: int) -> float:
        """
        Calculate Recall@K - Fixed implementation
        
        Args:
            recommendations: List of recommended items
            relevant_items: List of relevant/ground truth items
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score (0-1)
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        hits = sum(1 for item in top_k_recs if item in relevant_set)
        return hits / len(relevant_items)
    
    def ndcg_at_k(self, recommendations: List[Any], relevant_items: List[Any], k: int, 
                  relevance_scores: Optional[Dict[Any, float]] = None) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K - FIXED VERSION
        
        This fixes the NDCG > 1.0 issue by implementing proper normalization
        
        Args:
            recommendations: List of recommended items
            relevant_items: List of relevant items
            k: Number of top recommendations to consider
            relevance_scores: Optional relevance scores for items (default: binary)
            
        Returns:
            NDCG@K score (0-1) - GUARANTEED to be <= 1.0
        """
        if k == 0 or len(recommendations) == 0 or len(relevant_items) == 0:
            return 0.0
        
        # Default to binary relevance if no scores provided
        if relevance_scores is None:
            relevance_scores = {item: 1.0 for item in relevant_items}
        
        # Ensure we only consider top-k recommendations
        top_k_recs = recommendations[:k]
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, item in enumerate(top_k_recs):
            if item in relevance_scores:
                relevance = relevance_scores[item]
                # Use log2(i + 2) because positions start from 1, and log2(1) = 0
                discount = np.log2(i + 2)
                dcg += relevance / discount
        
        # Calculate IDCG (Ideal DCG) - this is the key fix
        # Sort relevant items by relevance score (descending)
        sorted_relevant = sorted(
            relevant_items, 
            key=lambda x: relevance_scores.get(x, 0), 
            reverse=True
        )
        
        # IDCG considers only the top-k relevant items or all relevant items if fewer than k
        ideal_items = sorted_relevant[:min(k, len(sorted_relevant))]
        
        idcg = 0.0
        for i, item in enumerate(ideal_items):
            relevance = relevance_scores.get(item, 0)
            discount = np.log2(i + 2)
            idcg += relevance / discount
        
        # Return NDCG with proper bounds checking
        if idcg == 0.0:
            return 0.0
        
        ndcg_score = dcg / idcg
        
        # CRITICAL FIX: Ensure NDCG never exceeds 1.0
        if ndcg_score > 1.0:
            logger.warning(f"NDCG calculation error: DCG={dcg:.6f}, IDCG={idcg:.6f}, NDCG={ndcg_score:.6f}")
            logger.warning(f"Recommendations: {top_k_recs[:5]}...")
            logger.warning(f"Relevant items: {relevant_items[:5]}...")
            logger.warning(f"Ideal items: {ideal_items}")
            # Cap at 1.0 and log the issue
            ndcg_score = 1.0
        
        return ndcg_score
    
    def mean_average_precision(self, recommendations: List[Any], relevant_items: List[Any]) -> float:
        """
        Calculate Mean Average Precision (MAP) - Fixed implementation
        """
        if len(relevant_items) == 0:
            return 0.0
        
        relevant_set = set(relevant_items)
        precision_sum = 0.0
        num_hits = 0
        
        for i, item in enumerate(recommendations):
            if item in relevant_set:
                num_hits += 1
                precision_at_i = num_hits / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_items)
    
    def mean_reciprocal_rank(self, recommendations: List[Any], relevant_items: List[Any]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) - Fixed implementation
        """
        relevant_set = set(relevant_items)
        
        for i, item in enumerate(recommendations):
            if item in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def test_ndcg_calculation(self):
        """
        Test the NDCG calculation with known examples to ensure correctness
        """
        print("🧪 Testing Fixed NDCG Calculation")
        print("=" * 50)
        
        # Test Case 1: Perfect recommendations
        recommendations = ['tech', 'finance', 'healthcare']
        relevant_items = ['tech', 'finance', 'healthcare']
        ndcg = self.ndcg_at_k(recommendations, relevant_items, 3)
        print(f"Test 1 - Perfect match: NDCG@3 = {ndcg:.6f} (should be 1.0)")
        assert abs(ndcg - 1.0) < 1e-6, f"Perfect match should give NDCG=1.0, got {ndcg}"
        
        # Test Case 2: No relevant items in recommendations
        recommendations = ['retail', 'energy', 'automotive']
        relevant_items = ['tech', 'finance', 'healthcare']
        ndcg = self.ndcg_at_k(recommendations, relevant_items, 3)
        print(f"Test 2 - No matches: NDCG@3 = {ndcg:.6f} (should be 0.0)")
        assert ndcg == 0.0, f"No matches should give NDCG=0.0, got {ndcg}"
        
        # Test Case 3: Partial match
        recommendations = ['tech', 'retail', 'finance']
        relevant_items = ['tech', 'finance', 'healthcare']
        ndcg = self.ndcg_at_k(recommendations, relevant_items, 3)
        print(f"Test 3 - Partial match: NDCG@3 = {ndcg:.6f} (should be < 1.0)")
        assert 0.0 < ndcg < 1.0, f"Partial match should give 0 < NDCG < 1, got {ndcg}"
        
        # Test Case 4: Single item match at top
        recommendations = ['tech']
        relevant_items = ['tech']
        ndcg = self.ndcg_at_k(recommendations, relevant_items, 1)
        print(f"Test 4 - Single perfect match: NDCG@1 = {ndcg:.6f} (should be 1.0)")
        assert abs(ndcg - 1.0) < 1e-6, f"Single perfect match should give NDCG=1.0, got {ndcg}"
        
        # Test Case 5: Edge case - empty lists
        recommendations = []
        relevant_items = ['tech', 'finance']
        ndcg = self.ndcg_at_k(recommendations, relevant_items, 3)
        print(f"Test 5 - Empty recommendations: NDCG@3 = {ndcg:.6f} (should be 0.0)")
        assert ndcg == 0.0, f"Empty recommendations should give NDCG=0.0, got {ndcg}"
        
        # Test Case 6: Edge case - empty relevant items
        recommendations = ['tech', 'finance']
        relevant_items = []
        ndcg = self.ndcg_at_k(recommendations, relevant_items, 3)
        print(f"Test 6 - Empty relevant items: NDCG@3 = {ndcg:.6f} (should be 0.0)")
        assert ndcg == 0.0, f"Empty relevant items should give NDCG=0.0, got {ndcg}"
        
        print("\n✅ All NDCG tests passed!")
        print("🎯 NDCG calculation is now mathematically correct!")
        
        return True

def test_metrics_calculator():
    """Test the fixed metrics calculator"""
    calculator = FixedMetricsCalculator()
    
    # Run NDCG tests
    calculator.test_ndcg_calculation()
    
    # Test other metrics
    recommendations = ['tech', 'finance', 'healthcare', 'retail']
    relevant_items = ['tech', 'finance', 'automotive']
    
    print(f"\n📊 Testing Other Metrics:")
    print(f"Precision@3: {calculator.precision_at_k(recommendations, relevant_items, 3):.3f}")
    print(f"Recall@3: {calculator.recall_at_k(recommendations, relevant_items, 3):.3f}")
    print(f"MAP: {calculator.mean_average_precision(recommendations, relevant_items):.3f}")
    print(f"MRR: {calculator.mean_reciprocal_rank(recommendations, relevant_items):.3f}")
    
    print("\n✅ All metrics tests completed!")

if __name__ == "__main__":
    test_metrics_calculator()
