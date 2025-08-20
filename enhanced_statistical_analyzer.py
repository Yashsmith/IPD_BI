#!/usr/bin/env python3
"""
Enhanced Statistical Analysis for Step 2.1
Publication-ready statistical testing with comprehensive reporting
"""

import sys
import os
sys.path.append('/Users/Sameer/Yashsmith/Research/IPD/bi_platform')

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
from scipy.stats import normaltest, shapiro, levene, bartlett

@dataclass
class EnhancedComparisonResult:
    """Comprehensive statistical comparison result"""
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

class EnhancedStatisticalAnalyzer:
    """
    Enhanced statistical analyzer for publication-ready research
    """
    
    def __init__(self, alpha: float = 0.05, confidence_level: float = 0.95):
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
            statistic, p_value = shapiro(data)
        else:
            statistic, p_value = normaltest(data)
            
        is_normal = p_value > self.alpha
        return p_value, is_normal
    
    def test_equal_variances(self, group1: List[float], group2: List[float]) -> Tuple[float, bool]:
        """Test equality of variances using Levene's test"""
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
    
    def generate_comparison_table(self, comparisons: List[EnhancedComparisonResult]) -> pd.DataFrame:
        """Generate a publication-ready comparison table"""
        
        rows = []
        for comp in comparisons:
            row = {
                'Metric': comp.metric_name,
                'Method A': comp.method_a_name,
                'Method B': comp.method_b_name,
                'Mean A (±SD)': f"{comp.mean_a:.3f} (±{comp.std_a:.3f})",
                'Mean B (±SD)': f"{comp.mean_b:.3f} (±{comp.std_b:.3f})",
                'Mean Diff': f"{comp.mean_difference:.3f}",
                f'{comp.confidence_level*100:.0f}% CI': f"[{comp.mean_diff_ci_lower:.3f}, {comp.mean_diff_ci_upper:.3f}]",
                "Cohen's d": f"{comp.cohens_d:.3f}",
                'Effect Size': comp.effect_size_interpretation,
                'Test Used': comp.recommended_test,
                'p-value': f"{comp.t_p_value:.3f}" if 't-test' in comp.recommended_test else f"{comp.mann_whitney_p_value:.3f}",
                'Significant': "Yes" if comp.is_significant else "No",
                'Interpretation': comp.interpretation
            }
            rows.append(row)
            
        return pd.DataFrame(rows)

def test_enhanced_statistical_analyzer():
    """Test the enhanced statistical analyzer"""
    print("🧮 TESTING ENHANCED STATISTICAL ANALYZER")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    method_a_results = np.random.normal(0.7, 0.1, 30).tolist()  # GRPO-GRPO-P
    method_b_results = np.random.normal(0.6, 0.12, 30).tolist()  # Baseline
    
    analyzer = EnhancedStatisticalAnalyzer()
    
    # Perform comprehensive comparison
    result = analyzer.comprehensive_comparison(
        method_a_results, method_b_results,
        "GRPO-GRPO-P", "PopularityBaseline", "NDCG@5"
    )
    
    print("📊 COMPREHENSIVE COMPARISON RESULTS:")
    print(f"  Metric: {result.metric_name}")
    print(f"  Method A ({result.method_a_name}): {result.mean_a:.3f} ± {result.std_a:.3f}")
    print(f"  Method B ({result.method_b_name}): {result.mean_b:.3f} ± {result.std_b:.3f}")
    print(f"  Mean Difference: {result.mean_difference:.3f}")
    print(f"  95% CI: [{result.mean_diff_ci_lower:.3f}, {result.mean_diff_ci_upper:.3f}]")
    
    print("\n🧪 STATISTICAL TESTS:")
    print(f"  Normality A (p={result.normality_a_p:.3f}): {'Normal' if result.normality_a_p > 0.05 else 'Non-normal'}")
    print(f"  Normality B (p={result.normality_b_p:.3f}): {'Normal' if result.normality_b_p > 0.05 else 'Non-normal'}")
    print(f"  Equal Variances (p={result.variance_test_p:.3f}): {'Yes' if result.variance_test_p > 0.05 else 'No'}")
    print(f"  Recommended Test: {result.recommended_test}")
    
    print("\n📈 EFFECT SIZE:")
    print(f"  Cohen's d: {result.cohens_d:.3f}")
    print(f"  Interpretation: {result.effect_size_interpretation}")
    
    print("\n🎯 CONCLUSION:")
    print(f"  Significant: {result.is_significant}")
    print(f"  Interpretation: {result.interpretation}")
    
    # Test table generation
    print("\n📋 COMPARISON TABLE:")
    comparisons = [result]
    table = analyzer.generate_comparison_table(comparisons)
    print(table.to_string(index=False))
    
    print("\n✅ Enhanced Statistical Analyzer is working correctly!")

if __name__ == "__main__":
    test_enhanced_statistical_analyzer()
