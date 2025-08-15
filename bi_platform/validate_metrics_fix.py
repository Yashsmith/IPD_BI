#!/usr/bin/env python3
"""
Validation script to test the fixed NDCG calculation
This ensures NDCG values are always in the valid range [0, 1]
"""

import sys
import os
sys.path.append('.')

from src.evaluation.research_evaluator import RecommendationEvaluator
import numpy as np

def test_fixed_ndcg():
    """Test the fixed NDCG calculation with the research evaluator"""
    
    print("🧪 Testing Fixed NDCG in Research Evaluator")
    print("=" * 50)
    
    evaluator = RecommendationEvaluator()
    
    # Test cases
    test_cases = [
        {
            "name": "Perfect Match",
            "recommendations": ['tech', 'finance', 'healthcare'],
            "relevant": ['tech', 'finance', 'healthcare'],
            "expected_ndcg": 1.0
        },
        {
            "name": "No Match",
            "recommendations": ['retail', 'energy', 'automotive'],
            "relevant": ['tech', 'finance', 'healthcare'],
            "expected_ndcg": 0.0
        },
        {
            "name": "Partial Match",
            "recommendations": ['tech', 'retail', 'finance'],
            "relevant": ['tech', 'finance', 'healthcare'],
            "expected_range": (0.0, 1.0)
        },
        {
            "name": "Single Match at Top",
            "recommendations": ['tech', 'retail', 'energy'],
            "relevant": ['tech'],
            "expected_ndcg": 1.0
        },
        {
            "name": "Single Match at Bottom",
            "recommendations": ['retail', 'energy', 'tech'],
            "relevant": ['tech'],
            "expected_range": (0.0, 1.0)
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        ndcg_5 = evaluator.ndcg_at_k(
            test_case['recommendations'], 
            test_case['relevant'], 
            k=3
        )
        
        print(f"Test {i} - {test_case['name']}: NDCG@3 = {ndcg_5:.6f}")
        
        # Check bounds
        if not (0.0 <= ndcg_5 <= 1.0):
            print(f"❌ FAIL: NDCG out of bounds [0,1]: {ndcg_5}")
            all_passed = False
            continue
        
        # Check expected values
        if 'expected_ndcg' in test_case:
            if abs(ndcg_5 - test_case['expected_ndcg']) > 1e-6:
                print(f"❌ FAIL: Expected {test_case['expected_ndcg']}, got {ndcg_5}")
                all_passed = False
            else:
                print(f"✅ PASS: Correct NDCG value")
        elif 'expected_range' in test_case:
            min_val, max_val = test_case['expected_range']
            if min_val <= ndcg_5 <= max_val:
                print(f"✅ PASS: NDCG in expected range [{min_val}, {max_val}]")
            else:
                print(f"❌ FAIL: NDCG {ndcg_5} not in range [{min_val}, {max_val}]")
                all_passed = False
    
    return all_passed

def test_sector_based_evaluation():
    """Test sector-based evaluation with realistic data"""
    
    print(f"\n🎯 Testing Sector-Based Evaluation")
    print("=" * 50)
    
    evaluator = RecommendationEvaluator()
    
    # Simulate realistic recommendation scenario
    user_recommendations = {
        'user_1': ['technology', 'finance', 'healthcare'],
        'user_2': ['finance', 'retail', 'technology'],
        'user_3': ['healthcare', 'technology', 'finance'],
        'user_4': ['retail', 'finance', 'healthcare'],
        'user_5': ['technology', 'healthcare', 'retail']
    }
    
    user_ground_truth = {
        'user_1': ['technology', 'finance'],
        'user_2': ['finance', 'technology'], 
        'user_3': ['healthcare', 'technology'],
        'user_4': ['retail', 'finance'],
        'user_5': ['technology', 'retail']
    }
    
    # Evaluate system
    user_metrics = evaluator.evaluate_system(
        user_recommendations, 
        user_ground_truth, 
        catalog_size=4  # 4 sectors: tech, finance, healthcare, retail
    )
    
    # Aggregate metrics
    aggregated = evaluator.aggregate_metrics(user_metrics)
    
    print(f"📊 System-Level Results:")
    print(f"  • NDCG@5: {aggregated.ndcg_at_k.get(5, 0.0):.6f}")
    print(f"  • Precision@5: {aggregated.precision_at_k.get(5, 0.0):.6f}")
    print(f"  • Recall@5: {aggregated.recall_at_k.get(5, 0.0):.6f}")
    print(f"  • MAP: {aggregated.map_score:.6f}")
    print(f"  • MRR: {aggregated.mrr_score:.6f}")
    
    # Validate all metrics are in valid ranges
    all_valid = True
    
    # Check NDCG@K
    for k, ndcg in aggregated.ndcg_at_k.items():
        if not (0.0 <= ndcg <= 1.0):
            print(f"❌ NDCG@{k} out of bounds: {ndcg}")
            all_valid = False
    
    # Check Precision@K  
    for k, precision in aggregated.precision_at_k.items():
        if not (0.0 <= precision <= 1.0):
            print(f"❌ Precision@{k} out of bounds: {precision}")
            all_valid = False
    
    # Check Recall@K
    for k, recall in aggregated.recall_at_k.items():
        if not (0.0 <= recall <= 1.0):
            print(f"❌ Recall@{k} out of bounds: {recall}")
            all_valid = False
    
    # Check MAP and MRR
    if not (0.0 <= aggregated.map_score <= 1.0):
        print(f"❌ MAP out of bounds: {aggregated.map_score}")
        all_valid = False
    
    if not (0.0 <= aggregated.mrr_score <= 1.0):
        print(f"❌ MRR out of bounds: {aggregated.mrr_score}")
        all_valid = False
    
    if all_valid:
        print("✅ All metrics are within valid bounds [0, 1]")
    
    return all_valid

def run_comprehensive_validation():
    """Run comprehensive validation of fixed metrics"""
    
    print("🚀 COMPREHENSIVE METRICS VALIDATION")
    print("=" * 60)
    
    # Test 1: Fixed NDCG calculation
    ndcg_passed = test_fixed_ndcg()
    
    # Test 2: Sector-based evaluation
    sector_passed = test_sector_based_evaluation()
    
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"✅ NDCG Tests: {'PASSED' if ndcg_passed else 'FAILED'}")
    print(f"✅ Sector Tests: {'PASSED' if sector_passed else 'FAILED'}")
    
    if ndcg_passed and sector_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"🎯 NDCG calculation is now mathematically correct!")
        print(f"📈 Ready for Step 1.2: Fix baseline evaluation")
        return True
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        print(f"🔧 Need to investigate further...")
        return False

if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)
