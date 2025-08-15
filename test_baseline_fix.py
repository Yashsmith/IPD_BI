#!/usr/bin/env python3
"""
Test the baseline evaluation fix with real evaluation
"""

import sys
import os
sys.path.append('/Users/Sameer/Yashsmith/Research/IPD/bi_platform')

from src.evaluation.experimental_framework import ExperimentalFramework
from src.evaluation.baseline_methods import PopularityBaseline, RandomBaseline

def test_baseline_fix():
    print("=== TESTING BASELINE EVALUATION FIX ===")
    
    # Create framework
    framework = ExperimentalFramework()
    
    # Create test dataset with known relevant sectors
    test_data = {
        'user_1': {
            'interactions': [
                {'item_id': 'startup_1', 'rating': 4.0},  # technology
                {'item_id': 'startup_2', 'rating': 5.0},  # finance 
                {'item_id': 'startup_3', 'rating': 3.0}   # healthcare
            ],
            'sectors': ['technology', 'finance']
        },
        'user_2': {
            'interactions': [
                {'item_id': 'startup_4', 'rating': 4.0},  # technology
                {'item_id': 'startup_5', 'rating': 2.0},  # finance
                {'item_id': 'startup_6', 'rating': 5.0}   # healthcare
            ],
            'sectors': ['healthcare', 'retail']
        },
        'user_3': {
            'interactions': [
                {'item_id': 'startup_7', 'rating': 5.0},  # retail
                {'item_id': 'startup_8', 'rating': 3.0},  # technology
                {'item_id': 'startup_9', 'rating': 4.0}   # finance
            ],
            'sectors': ['retail', 'technology']
        }
    }
    
    print("Test data relevant sectors:")
    for user_id, data in test_data.items():
        relevant_sectors = framework._extract_relevant_sectors(data)
        print(f"  {user_id}: {relevant_sectors}")
    
    # Create simple dataset split
    class SimpleDatasetSplit:
        def __init__(self, users, test_data):
            self.train_users = users[:2]  # Use first 2 for training
            self.test_users = users  # Test on all users
            self.val_users = []
            self.train_data = {k: test_data[k] for k in self.train_users}
            self.test_data = test_data
            self.val_data = {}
            self.split_timestamp = "2024-01-01T00:00:00"
    
    dataset_split = SimpleDatasetSplit(list(test_data.keys()), test_data)
    
    print(f"\nTraining on: {dataset_split.train_users}")
    print(f"Testing on: {dataset_split.test_users}")
    
    # Test PopularityBaseline
    print("\n=== TESTING POPULARITY BASELINE ===")
    pop_baseline = PopularityBaseline()
    
    try:
        results = framework.run_baseline_experiment(
            pop_baseline, 
            test_data, 
            dataset_split
        )
        
        print("✅ PopularityBaseline evaluation completed!")
        print("Results:")
        if hasattr(results, 'test_metrics'):
            metrics_dict = results.test_metrics.__dict__
            for metric, value in metrics_dict.items():
                print(f"  {metric}: {value:.4f}")
                
            # Check if we got non-zero results
            non_zero_metrics = [k for k, v in metrics_dict.items() if v > 0]
            if non_zero_metrics:
                print(f"✅ Non-zero metrics: {non_zero_metrics}")
            else:
                print("❌ All metrics are still zero")
        else:
            print("❌ No test_metrics found in results")
            
    except Exception as e:
        print(f"❌ PopularityBaseline failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test RandomBaseline  
    print("\n=== TESTING RANDOM BASELINE ===")
    random_baseline = RandomBaseline()
    
    try:
        results = framework.run_baseline_experiment(
            random_baseline, 
            test_data, 
            dataset_split
        )
        
        print("✅ RandomBaseline evaluation completed!")
        print("Results:")
        if hasattr(results, 'test_metrics'):
            metrics_dict = results.test_metrics.__dict__
            for metric, value in metrics_dict.items():
                print(f"  {metric}: {value:.4f}")
                
            # Check if we got non-zero results
            non_zero_metrics = [k for k, v in metrics_dict.items() if v > 0]
            if non_zero_metrics:
                print(f"✅ Non-zero metrics: {non_zero_metrics}")
            else:
                print("❌ All metrics are still zero")
        else:
            print("❌ No test_metrics found in results")
            
    except Exception as e:
        print(f"❌ RandomBaseline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_baseline_fix()
