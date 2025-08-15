#!/usr/bin/env python3
"""
Quick test to verify GRPO-GRPO-P now produces valid NDCG scores
"""

import sys
import os
sys.path.append('/Users/Sameer/Yashsmith/Research/IPD/bi_platform')

from src.evaluation.experimental_framework import ExperimentalFramework, DatasetSplit, ExperimentConfig

def test_grpo_grpo_p_fix():
    print("=== TESTING GRPO-GRPO-P NDCG FIX ===")
    
    # Create framework
    framework = ExperimentalFramework()
    
    # Create test dataset
    test_data = {
        'user_1': {
            'interactions': [
                {'item_id': 'startup_1', 'rating': 4.0, 'timestamp': '2024-01-01'},
                {'item_id': 'startup_2', 'rating': 5.0, 'timestamp': '2024-01-01'},
            ],
            'sectors': ['technology', 'finance'],
            'role': 'investor'
        },
        'user_2': {
            'interactions': [
                {'item_id': 'startup_3', 'rating': 3.0, 'timestamp': '2024-01-01'},
                {'item_id': 'startup_4', 'rating': 4.5, 'timestamp': '2024-01-01'},
            ],
            'sectors': ['healthcare', 'technology'],
            'role': 'entrepreneur'
        }
    }
    
    # Create dataset split
    dataset_split = DatasetSplit(
        train_users=['user_1'],
        val_users=[],
        test_users=['user_1', 'user_2'],
        train_data={'user_1': test_data['user_1']},
        val_data={},
        test_data=test_data,
        split_timestamp="2024-01-01T00:00:00"
    )
    
    # Create config
    config = ExperimentConfig(
        experiment_name="GRPO_GRPO_P_NDCG_Test",
        method_name="GRPO-GRPO-P",
        hyperparameters={
            'learning_rate': 0.01,
            'exploration_rate': 0.1,
            'population_weight': 0.5,
            'personal_weight': 0.7
        },
        dataset_config={},
        evaluation_config={},
        random_seed=42
    )
    
    print(f"Training on: {dataset_split.train_users}")
    print(f"Testing on: {dataset_split.test_users}")
    
    try:
        results = framework.run_grpo_grpo_p_experiment(dataset_split, config)
        
        print("✅ GRPO-GRPO-P evaluation completed!")
        print(f"Status: {results.status}")
        
        if hasattr(results, 'test_metrics'):
            metrics = results.test_metrics
            print(f"Test metrics type: {type(metrics)}")
            
            # Check NDCG values
            ndcg_scores = getattr(metrics, 'ndcg_at_k', {})
            print(f"NDCG scores: {ndcg_scores}")
            
            # Check for impossible values
            invalid_ndcg = []
            for k, score in ndcg_scores.items():
                if score > 1.0:
                    invalid_ndcg.append((k, score))
                    
            if invalid_ndcg:
                print(f"❌ INVALID NDCG SCORES FOUND:")
                for k, score in invalid_ndcg:
                    print(f"  NDCG@{k}: {score:.4f} (> 1.0 - IMPOSSIBLE)")
                print("🔧 NDCG fix has NOT been applied to GRPO-GRPO-P evaluation")
            else:
                print(f"✅ ALL NDCG SCORES ARE VALID (≤ 1.0)")
                print("🎉 NDCG fix has been successfully applied!")
                
            # Show other key metrics
            print(f"\nKey Metrics:")
            print(f"  MAP: {getattr(metrics, 'map_score', 'N/A')}")
            print(f"  MRR: {getattr(metrics, 'mrr_score', 'N/A')}")
            
            precision_scores = getattr(metrics, 'precision_at_k', {})
            recall_scores = getattr(metrics, 'recall_at_k', {})
            if 5 in precision_scores:
                print(f"  Precision@5: {precision_scores[5]:.4f}")
            if 5 in recall_scores:
                print(f"  Recall@5: {recall_scores[5]:.4f}")
                
        else:
            print("❌ No test_metrics found in results")
            
    except Exception as e:
        print(f"❌ GRPO-GRPO-P test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_grpo_grpo_p_fix()
