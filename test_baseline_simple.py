#!/usr/bin/env python3
"""
Test the baseline evaluation fix with proper DatasetSplit structure
"""

import sys
import os
sys.path.append('/Users/Sameer/Yashsmith/Research/IPD/bi_platform')

from src.evaluation.experimental_framework import ExperimentalFramework, DatasetSplit
from src.evaluation.baseline_methods import PopularityBaseline, RandomBaseline

def test_baseline_fix_simple():
    print("=== TESTING BASELINE EVALUATION FIX (SIMPLE) ===")
    
    # Create framework
    framework = ExperimentalFramework()
    
    # Create minimal test dataset
    test_data = {
        'user_1': {
            'interactions': [
                {'item_id': 'startup_1', 'rating': 4.0},  # technology
                {'item_id': 'startup_2', 'rating': 5.0},  # finance 
            ],
            'sectors': ['technology', 'finance']
        },
        'user_2': {
            'interactions': [
                {'item_id': 'startup_3', 'rating': 3.0},  # healthcare
                {'item_id': 'startup_4', 'rating': 4.0},  # technology
            ],
            'sectors': ['healthcare', 'technology']
        }
    }
    
    print("Test data relevant sectors:")
    for user_id, data in test_data.items():
        relevant_sectors = framework._extract_relevant_sectors(data)
        print(f"  {user_id}: {relevant_sectors}")
    
    # Create proper DatasetSplit
    dataset_split = DatasetSplit(
        train_users=['user_1'],
        val_users=[],
        test_users=['user_1', 'user_2'],
        train_data={'user_1': test_data['user_1']},
        val_data={},
        test_data=test_data,
        split_timestamp="2024-01-01T00:00:00"
    )
    
    print(f"\nTraining on: {dataset_split.train_users}")
    print(f"Testing on: {dataset_split.test_users}")
    
    # Test PopularityBaseline
    print("\n=== TESTING POPULARITY BASELINE ===")
    pop_baseline = PopularityBaseline()
    
    # Create a dummy config (the method needs this parameter)
    from src.evaluation.experimental_framework import ExperimentConfig
    config = ExperimentConfig(
        experiment_name="Test PopularityBaseline",
        method_name="PopularityBaseline",
        hyperparameters={},
        dataset_config={},
        evaluation_config={},
        random_seed=42
    )
    
    try:
        results = framework.run_baseline_experiment(
            pop_baseline, 
            dataset_split,
            config
        )
        
        print("✅ PopularityBaseline evaluation completed!")
        print("Results:")
        if hasattr(results, 'test_metrics'):
            # Let's inspect the test_metrics object
            print(f"test_metrics type: {type(results.test_metrics)}")
            print(f"test_metrics: {results.test_metrics}")
            
            # Try to access attributes
            try:
                print(f"  NDCG@5: {getattr(results.test_metrics, 'ndcg_at_5', 'N/A')}")
                print(f"  NDCG@10: {getattr(results.test_metrics, 'ndcg_at_10', 'N/A')}")
                print(f"  Precision@5: {getattr(results.test_metrics, 'precision_at_5', 'N/A')}")
                print(f"  Precision@10: {getattr(results.test_metrics, 'precision_at_10', 'N/A')}")
                print(f"  Recall@5: {getattr(results.test_metrics, 'recall_at_5', 'N/A')}")
                print(f"  MAP: {getattr(results.test_metrics, 'map_score', 'N/A')}")
                print(f"  MRR: {getattr(results.test_metrics, 'mrr', 'N/A')}")
            except Exception as e:
                print(f"Error accessing metrics: {e}")
        else:
            print("❌ No test_metrics found in results")
            
    except Exception as e:
        print(f"❌ PopularityBaseline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_baseline_fix_simple()
