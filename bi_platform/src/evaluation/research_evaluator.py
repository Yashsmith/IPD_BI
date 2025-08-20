"""
Research Evaluation Framework for GRPO-GRPO-P System
Comprehensive metrics, baselines, and statistical testing for academic research
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import json
import os
from datetime import datetime
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Standard recommendation system evaluation metrics"""
    precision_at_k: Dict[int, float]  # Precision@K for different K values
    recall_at_k: Dict[int, float]     # Recall@K for different K values
    ndcg_at_k: Dict[int, float]       # NDCG@K for different K values
    map_score: float                  # Mean Average Precision
    mrr_score: float                  # Mean Reciprocal Rank
    diversity: float                  # Intra-list diversity
    novelty: float                    # Novelty score
    coverage: float                   # Catalog coverage
    user_satisfaction: Optional[float] = None  # From user studies
    engagement_metrics: Optional[Dict[str, float]] = None  # CTR, dwell time, etc.

@dataclass
class ExperimentResult:
    """Results from a single experimental run"""
    method_name: str
    user_id: str
    run_id: int
    metrics: EvaluationMetrics
    execution_time: float
    memory_usage: float
    recommendations: List[Any]
    ground_truth: List[Any]
    timestamp: str

@dataclass
class ComparisonResult:
    """Statistical comparison between methods - Enhanced for Phase 2"""
    method_a: str
    method_b: str
    metric_name: str
    mean_diff: float
    std_diff: float
    p_value: float
    effect_size: float  # Cohen's d
    confidence_interval: Tuple[float, float]
    significant: bool
    verdict: str  # "A significantly better", "B significantly better", "No significant difference"

@dataclass
class EnhancedComparisonResult:
    """Comprehensive statistical comparison result for publication-ready research"""
    method_a_name: str
    method_b_name: str
    metric_name: str
    
    # Descriptive statistics
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    median_a: float
    median_b: float
    mean_difference: float
    
    # Statistical tests
    t_statistic: float
    t_p_value: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    mann_whitney_statistic: float
    mann_whitney_p_value: float
    
    # Effect sizes
    cohens_d: float
    effect_size_interpretation: str
    
    # Confidence intervals
    mean_diff_ci_lower: float
    mean_diff_ci_upper: float
    confidence_level: float
    
    # Normality and variance tests
    normality_a_p: float
    normality_b_p: float
    variance_test_p: float
    
    # Final recommendation
    is_significant: bool
    recommended_test: str
    interpretation: str

class RecommendationEvaluator:
    """
    Comprehensive evaluation framework for recommendation systems
    Implements standard metrics used in RecSys research
    """
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10, 20]):
        """
        Initialize evaluator
        
        Args:
            k_values: List of K values for top-K metrics
        """
        self.k_values = k_values
        self.results_cache = {}
        
    def precision_at_k(self, recommendations: List[Any], relevant_items: List[Any], k: int) -> float:
        """
        Calculate Precision@K
        
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
        return hits / min(k, len(top_k_recs))
    
    def recall_at_k(self, recommendations: List[Any], relevant_items: List[Any], k: int) -> float:
        """
        Calculate Recall@K
        
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
        and handling duplicate recommendations correctly
        
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
        
        # Remove duplicates from recommendations while preserving order
        seen = set()
        unique_recommendations = []
        for item in recommendations:
            if item not in seen:
                seen.add(item)
                unique_recommendations.append(item)
        
        # Ensure we only consider top-k unique recommendations
        top_k_recs = unique_recommendations[:k]
        
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
            logger.warning(f"NDCG calculation error after deduplication: DCG={dcg:.6f}, IDCG={idcg:.6f}, NDCG={ndcg_score:.6f}")
            logger.warning(f"Original recommendations: {recommendations[:10]}...")
            logger.warning(f"Unique recommendations: {top_k_recs}")
            logger.warning(f"Relevant items: {relevant_items}")
            logger.warning(f"Ideal items: {ideal_items}")
            # Cap at 1.0 and log the issue
            ndcg_score = 1.0
        
        return ndcg_score
    
    def mean_average_precision(self, recommendations: List[Any], relevant_items: List[Any]) -> float:
        """
        Calculate Mean Average Precision (MAP)
        
        Args:
            recommendations: List of recommended items
            relevant_items: List of relevant items
            
        Returns:
            MAP score (0-1)
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
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            recommendations: List of recommended items
            relevant_items: List of relevant items
            
        Returns:
            MRR score (0-1)
        """
        relevant_set = set(relevant_items)
        
        for i, item in enumerate(recommendations):
            if item in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def diversity_score(self, recommendations: List[Any], 
                       similarity_matrix: Optional[np.ndarray] = None) -> float:
        """
        Calculate intra-list diversity of recommendations
        
        Args:
            recommendations: List of recommended items
            similarity_matrix: Item similarity matrix (if available)
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(recommendations) <= 1:
            return 1.0
        
        if similarity_matrix is not None:
            # Use provided similarity matrix
            total_similarity = 0.0
            pairs = 0
            
            for i in range(len(recommendations)):
                for j in range(i + 1, len(recommendations)):
                    try:
                        item_i_idx = recommendations[i]
                        item_j_idx = recommendations[j]
                        similarity = similarity_matrix[item_i_idx, item_j_idx]
                        total_similarity += similarity
                        pairs += 1
                    except (IndexError, TypeError):
                        continue
            
            avg_similarity = total_similarity / pairs if pairs > 0 else 0.0
            return 1.0 - avg_similarity
        else:
            # Simple diversity based on uniqueness
            unique_items = len(set(recommendations))
            return unique_items / len(recommendations)
    
    def novelty_score(self, recommendations: List[Any], 
                     item_popularity: Dict[Any, float]) -> float:
        """
        Calculate novelty of recommendations (inverse popularity)
        
        Args:
            recommendations: List of recommended items
            item_popularity: Popularity scores for items
            
        Returns:
            Novelty score (0-1, higher is more novel)
        """
        if len(recommendations) == 0:
            return 0.0
        
        novelty_sum = 0.0
        for item in recommendations:
            popularity = item_popularity.get(item, 0.0)
            novelty = 1.0 - popularity  # Higher novelty for less popular items
            novelty_sum += novelty
        
        return novelty_sum / len(recommendations)
    
    def catalog_coverage(self, all_recommendations: List[List[Any]], 
                        catalog_size: int) -> float:
        """
        Calculate catalog coverage across all recommendations
        
        Args:
            all_recommendations: List of recommendation lists for all users
            catalog_size: Total number of items in catalog
            
        Returns:
            Coverage score (0-1)
        """
        recommended_items = set()
        for rec_list in all_recommendations:
            recommended_items.update(rec_list)
        
        return len(recommended_items) / catalog_size if catalog_size > 0 else 0.0
    
    def evaluate_single_user(self, recommendations: List[Any], 
                            relevant_items: List[Any],
                            relevance_scores: Optional[Dict[Any, float]] = None,
                            similarity_matrix: Optional[np.ndarray] = None,
                            item_popularity: Optional[Dict[Any, float]] = None) -> EvaluationMetrics:
        """
        Evaluate recommendations for a single user
        
        Args:
            recommendations: List of recommended items
            relevant_items: List of relevant/ground truth items
            relevance_scores: Optional relevance scores
            similarity_matrix: Optional item similarity matrix
            item_popularity: Optional item popularity scores
            
        Returns:
            EvaluationMetrics object with all computed metrics
        """
        
        # Calculate precision@k and recall@k for all k values
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        
        for k in self.k_values:
            precision_at_k[k] = self.precision_at_k(recommendations, relevant_items, k)
            recall_at_k[k] = self.recall_at_k(recommendations, relevant_items, k)
            ndcg_at_k[k] = self.ndcg_at_k(recommendations, relevant_items, k, relevance_scores)
        
        # Calculate other metrics
        map_score = self.mean_average_precision(recommendations, relevant_items)
        mrr_score = self.mean_reciprocal_rank(recommendations, relevant_items)
        
        diversity = self.diversity_score(recommendations, similarity_matrix)
        
        novelty = 0.0
        if item_popularity is not None:
            novelty = self.novelty_score(recommendations, item_popularity)
        
        return EvaluationMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            map_score=map_score,
            mrr_score=mrr_score,
            diversity=diversity,
            novelty=novelty,
            coverage=0.0  # Will be calculated at system level
        )
    
    def evaluate_system(self, user_recommendations: Dict[str, List[Any]],
                       user_ground_truth: Dict[str, List[Any]],
                       catalog_size: int,
                       relevance_scores: Optional[Dict[str, Dict[Any, float]]] = None,
                       similarity_matrix: Optional[np.ndarray] = None,
                       item_popularity: Optional[Dict[Any, float]] = None) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate entire recommendation system across all users
        
        Args:
            user_recommendations: Dict mapping user_id to recommended items
            user_ground_truth: Dict mapping user_id to relevant items
            catalog_size: Total number of items in catalog
            relevance_scores: Optional user-specific relevance scores
            similarity_matrix: Optional item similarity matrix
            item_popularity: Optional item popularity scores
            
        Returns:
            Dict mapping user_id to EvaluationMetrics
        """
        
        user_metrics = {}
        all_recommendations = []
        
        for user_id in user_recommendations:
            if user_id not in user_ground_truth:
                continue
            
            recommendations = user_recommendations[user_id]
            relevant_items = user_ground_truth[user_id]
            user_relevance = relevance_scores.get(user_id) if relevance_scores else None
            
            metrics = self.evaluate_single_user(
                recommendations, relevant_items, user_relevance,
                similarity_matrix, item_popularity
            )
            
            user_metrics[user_id] = metrics
            all_recommendations.append(recommendations)
        
        # Calculate system-level coverage
        coverage = self.catalog_coverage(all_recommendations, catalog_size)
        
        # Update coverage for all users
        for user_id in user_metrics:
            user_metrics[user_id].coverage = coverage
        
        return user_metrics
    
    def aggregate_metrics(self, user_metrics: Dict[str, EvaluationMetrics]) -> EvaluationMetrics:
        """
        Aggregate metrics across all users to get system-level performance
        
        Args:
            user_metrics: Dict mapping user_id to EvaluationMetrics
            
        Returns:
            Aggregated EvaluationMetrics
        """
        
        if not user_metrics:
            return EvaluationMetrics(
                precision_at_k={}, recall_at_k={}, ndcg_at_k={},
                map_score=0.0, mrr_score=0.0, diversity=0.0,
                novelty=0.0, coverage=0.0
            )
        
        # Aggregate precision@k, recall@k, ndcg@k
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        
        for k in self.k_values:
            precision_values = [metrics.precision_at_k.get(k, 0.0) for metrics in user_metrics.values()]
            recall_values = [metrics.recall_at_k.get(k, 0.0) for metrics in user_metrics.values()]
            ndcg_values = [metrics.ndcg_at_k.get(k, 0.0) for metrics in user_metrics.values()]
            
            precision_at_k[k] = np.mean(precision_values)
            recall_at_k[k] = np.mean(recall_values)
            ndcg_at_k[k] = np.mean(ndcg_values)
        
        # Aggregate other metrics
        map_scores = [metrics.map_score for metrics in user_metrics.values()]
        mrr_scores = [metrics.mrr_score for metrics in user_metrics.values()]
        diversity_scores = [metrics.diversity for metrics in user_metrics.values()]
        novelty_scores = [metrics.novelty for metrics in user_metrics.values()]
        
        # Coverage is system-level, so take from any user
        coverage = list(user_metrics.values())[0].coverage
        
        return EvaluationMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            map_score=np.mean(map_scores),
            mrr_score=np.mean(mrr_scores),
            diversity=np.mean(diversity_scores),
            novelty=np.mean(novelty_scores),
            coverage=coverage
        )

class StatisticalAnalyzer:
    """
    Enhanced statistical analysis tools for publication-ready recommendation research
    Implements comprehensive statistical testing with proper assumptions checking
    """
    
    def __init__(self, alpha: float = 0.05, confidence_level: float = 0.95):
        """
        Initialize enhanced statistical analyzer
        
        Args:
            alpha: Significance level for hypothesis testing
            confidence_level: Confidence level for intervals
        """
        self.alpha = alpha
        self.confidence_level = confidence_level
    
    def cohens_d(self, group1: List[float], group2: List[float]) -> Tuple[float, str]:
        """Calculate Cohen's d with interpretation"""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        
        # Interpretation based on Cohen (1988)
        if abs(d) < 0.2:
            interpretation = "negligible"
        elif abs(d) < 0.5:
            interpretation = "small"
        elif abs(d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
            
        return d, interpretation
    
    def test_normality(self, data: List[float]) -> Tuple[float, bool]:
        """Test normality using Shapiro-Wilk test"""
        if len(data) < 3:
            return 1.0, True  # Assume normal for very small samples
        
        # Use Shapiro-Wilk for smaller samples, D'Agostino for larger
        if len(data) <= 50:
            from scipy.stats import shapiro
            statistic, p_value = shapiro(data)
        else:
            from scipy.stats import normaltest
            statistic, p_value = normaltest(data)
            
        is_normal = p_value > self.alpha
        return p_value, is_normal
    
    def test_equal_variances(self, group1: List[float], group2: List[float]) -> Tuple[float, bool]:
        """Test equality of variances using Levene's test"""
        from scipy.stats import levene
        statistic, p_value = levene(group1, group2)
        equal_variances = p_value > self.alpha
        return p_value, equal_variances
    
    def confidence_interval_diff(self, group1: List[float], group2: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means"""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Standard error of difference
        se_diff = np.sqrt(var1/n1 + var2/n2)
        
        # Degrees of freedom (Welch's formula)
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        # t-critical value
        t_crit = stats.t.ppf((1 + self.confidence_level) / 2, df)
        
        # Difference and margin of error
        diff = mean1 - mean2
        margin = t_crit * se_diff
        
        return (diff - margin, diff + margin)
    
    def comprehensive_comparison(self, 
                               method_a_results: List[float],
                               method_b_results: List[float],
                               method_a_name: str,
                               method_b_name: str,
                               metric_name: str) -> EnhancedComparisonResult:
        """
        Comprehensive statistical comparison with multiple tests and interpretations
        """
        
        # Descriptive statistics
        mean_a, mean_b = np.mean(method_a_results), np.mean(method_b_results)
        std_a, std_b = np.std(method_a_results), np.std(method_b_results)
        median_a, median_b = np.median(method_a_results), np.median(method_b_results)
        mean_diff = mean_a - mean_b
        
        # Effect size
        cohens_d_val, effect_interpretation = self.cohens_d(method_a_results, method_b_results)
        
        # Normality tests
        norm_p_a, is_normal_a = self.test_normality(method_a_results)
        norm_p_b, is_normal_b = self.test_normality(method_b_results)
        
        # Variance test
        var_p, equal_vars = self.test_equal_variances(method_a_results, method_b_results)
        
        # Parametric tests
        t_stat, t_p = stats.ttest_ind(method_a_results, method_b_results, equal_var=equal_vars)
        
        # Non-parametric tests
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(method_a_results, method_b_results)
        except:
            # If paired test fails, use independent samples
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan
            
        mw_stat, mw_p = stats.mannwhitneyu(method_a_results, method_b_results, alternative='two-sided')
        
        # Confidence interval for difference
        ci_lower, ci_upper = self.confidence_interval_diff(method_a_results, method_b_results)
        
        # Determine recommended test and significance
        both_normal = is_normal_a and is_normal_b
        
        if both_normal and equal_vars:
            recommended_test = "Independent t-test"
            p_value_to_use = t_p
        elif both_normal and not equal_vars:
            recommended_test = "Welch's t-test"
            p_value_to_use = t_p
        else:
            recommended_test = "Mann-Whitney U test"
            p_value_to_use = mw_p
            
        is_significant = p_value_to_use < self.alpha
        
        # Generate interpretation
        if is_significant:
            if mean_diff > 0:
                winner = method_a_name
                direction = "significantly better than"
            else:
                winner = method_b_name
                direction = "significantly better than"
                
            interpretation = f"{winner} is {direction} the other method (p={p_value_to_use:.3f}, effect size={effect_interpretation})"
        else:
            interpretation = f"No significant difference found (p={p_value_to_use:.3f})"
            
        return EnhancedComparisonResult(
            method_a_name=method_a_name,
            method_b_name=method_b_name,
            metric_name=metric_name,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            median_a=median_a,
            median_b=median_b,
            mean_difference=mean_diff,
            t_statistic=t_stat,
            t_p_value=t_p,
            wilcoxon_statistic=wilcoxon_stat,
            wilcoxon_p_value=wilcoxon_p,
            mann_whitney_statistic=mw_stat,
            mann_whitney_p_value=mw_p,
            cohens_d=cohens_d_val,
            effect_size_interpretation=effect_interpretation,
            mean_diff_ci_lower=ci_lower,
            mean_diff_ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            normality_a_p=norm_p_a,
            normality_b_p=norm_p_b,
            variance_test_p=var_p,
            is_significant=is_significant,
            recommended_test=recommended_test,
            interpretation=interpretation
        )
    
    def paired_t_test(self, group1: List[float], group2: List[float]) -> Tuple[float, float]:
        """
        Perform paired t-test
        
        Args:
            group1: First group of paired values
            group2: Second group of paired values
            
        Returns:
            (t_statistic, p_value)
        """
        return stats.ttest_rel(group1, group2)
    
    def independent_t_test(self, group1: List[float], group2: List[float]) -> Tuple[float, float]:
        """
        Perform independent samples t-test
        
        Args:
            group1: First group of values
            group2: Second group of values
            
        Returns:
            (t_statistic, p_value)
        """
        return stats.ttest_ind(group1, group2)
    
    def wilcoxon_test(self, group1: List[float], group2: List[float]) -> Tuple[float, float]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
        
        Args:
            group1: First group of paired values
            group2: Second group of paired values
            
        Returns:
            (statistic, p_value)
        """
        return stats.wilcoxon(group1, group2)
    
    def mann_whitney_test(self, group1: List[float], group2: List[float]) -> Tuple[float, float]:
        """
        Perform Mann-Whitney U test (non-parametric alternative to independent t-test)
        
        Args:
            group1: First group of values
            group2: Second group of values
            
        Returns:
            (statistic, p_value)
        """
        return stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    def confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for mean
        
        Args:
            data: List of values
            confidence: Confidence level (default 95%)
            
        Returns:
            (lower_bound, upper_bound)
        """
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)
        
        # Use t-distribution for small samples
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_val * std_err
        
        return (mean - margin_error, mean + margin_error)
    
    def compare_methods(self, method_a_results: List[float], 
                       method_b_results: List[float],
                       method_a_name: str, method_b_name: str,
                       metric_name: str,
                       paired: bool = True) -> ComparisonResult:
        """
        Statistical comparison between two methods (backward compatibility)
        
        Args:
            method_a_results: Results from method A
            method_b_results: Results from method B
            method_a_name: Name of method A
            method_b_name: Name of method B
            metric_name: Name of the metric being compared
            paired: Whether to use paired or independent test
            
        Returns:
            ComparisonResult with statistical analysis
        """
        
        # Basic statistics
        mean_a, mean_b = np.mean(method_a_results), np.mean(method_b_results)
        std_a, std_b = np.std(method_a_results), np.std(method_b_results)
        mean_diff = mean_a - mean_b
        
        # Effect size
        effect_size, _ = self.cohens_d(method_a_results, method_b_results)
        
        # Statistical test
        if paired:
            t_stat, p_value = self.paired_t_test(method_a_results, method_b_results)
        else:
            t_stat, p_value = self.independent_t_test(method_a_results, method_b_results)
        
        # Confidence interval for difference
        differences = [a - b for a, b in zip(method_a_results, method_b_results)]
        ci_lower, ci_upper = self.confidence_interval(differences)
        
        # Significance
        significant = p_value < self.alpha
        
        # Verdict
        if not significant:
            verdict = "No significant difference"
        elif mean_diff > 0:
            verdict = f"{method_a_name} significantly better"
        else:
            verdict = f"{method_b_name} significantly better"
        
        return ComparisonResult(
            method_a=method_a_name,
            method_b=method_b_name,
            metric_name=metric_name,
            mean_diff=mean_diff,
            std_diff=np.std(differences),
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            significant=significant,
            verdict=verdict
        )
    
    def generate_comparison_table(self, comparisons: List[EnhancedComparisonResult]) -> str:
        """Generate a publication-ready comparison table"""
        
        table_lines = []
        
        # Header
        header = f"{'Metric':<12} {'Method A':<15} {'Method B':<15} {'Mean A±SD':<12} {'Mean B±SD':<12} {'Diff':<8} {'CI':<15} {'d':<6} {'Test':<15} {'p':<8} {'Sig':<4}"
        table_lines.append(header)
        table_lines.append("=" * len(header))
        
        for comp in comparisons:
            mean_a_str = f"{comp.mean_a:.3f}±{comp.std_a:.3f}"
            mean_b_str = f"{comp.mean_b:.3f}±{comp.std_b:.3f}"
            ci_str = f"[{comp.mean_diff_ci_lower:.3f},{comp.mean_diff_ci_upper:.3f}]"
            p_val_str = f"{comp.t_p_value:.3f}" if 't-test' in comp.recommended_test else f"{comp.mann_whitney_p_value:.3f}"
            
            row = f"{comp.metric_name:<12} {comp.method_a_name:<15} {comp.method_b_name:<15} {mean_a_str:<12} {mean_b_str:<12} {comp.mean_difference:<8.3f} {ci_str:<15} {comp.cohens_d:<6.3f} {comp.recommended_test:<15} {p_val_str:<8} {'Yes' if comp.is_significant else 'No':<4}"
            table_lines.append(row)
            
        return "\n".join(table_lines)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Research Evaluation Framework")
    print("=" * 50)
    
    # Create sample data
    evaluator = RecommendationEvaluator()
    
    # Sample recommendations and ground truth
    recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
    relevant_items = ['item1', 'item3', 'item6', 'item7']
    
    # Test individual metrics
    print("📊 Testing Individual Metrics:")
    print(f"Precision@3: {evaluator.precision_at_k(recommendations, relevant_items, 3):.3f}")
    print(f"Recall@3: {evaluator.recall_at_k(recommendations, relevant_items, 3):.3f}")
    print(f"NDCG@3: {evaluator.ndcg_at_k(recommendations, relevant_items, 3):.3f}")
    print(f"MAP: {evaluator.mean_average_precision(recommendations, relevant_items):.3f}")
    print(f"MRR: {evaluator.mean_reciprocal_rank(recommendations, relevant_items):.3f}")
    
    # Test full evaluation
    metrics = evaluator.evaluate_single_user(recommendations, relevant_items)
    print(f"\n📈 Full Evaluation:")
    print(f"Precision@5: {metrics.precision_at_k[5]:.3f}")
    print(f"Recall@5: {metrics.recall_at_k[5]:.3f}")
    print(f"NDCG@5: {metrics.ndcg_at_k[5]:.3f}")
    print(f"Diversity: {metrics.diversity:.3f}")
    
    # Test statistical analysis
    analyzer = StatisticalAnalyzer()
    
    # Sample comparison data
    method_a_scores = [0.65, 0.72, 0.68, 0.71, 0.69, 0.73, 0.67, 0.70]
    method_b_scores = [0.58, 0.61, 0.59, 0.63, 0.60, 0.62, 0.57, 0.64]
    
    comparison = analyzer.compare_methods(
        method_a_scores, method_b_scores,
        "GRPO-GRPO-P", "Baseline", "NDCG@5"
    )
    
    print(f"\n📊 Statistical Comparison:")
    print(f"Verdict: {comparison.verdict}")
    print(f"Mean Difference: {comparison.mean_diff:.3f}")
    print(f"P-value: {comparison.p_value:.3f}")
    print(f"Effect Size (Cohen's d): {comparison.effect_size:.3f}")
    print(f"95% CI: ({comparison.confidence_interval[0]:.3f}, {comparison.confidence_interval[1]:.3f})")
    
    print("\n✅ Research Evaluation Framework Ready!")
    print("Ready for rigorous academic evaluation!")
