#!/usr/bin/env python3
"""
Debug the baseline evaluation to see what's happening step by step
"""

import sys
import os
sys.path.append('/Users/Sameer/Yashsmith/Research/IPD/bi_platform')

from src.evaluation.experimental_framework import ExperimentalFramework, DatasetSplit
from src.evaluation.baseline_methods import PopularityBaseline
from src.evaluation.sector_mapping import get_sector_for_startup

def debug_baseline_detailed():
    print("=== DETAILED BASELINE DEBUG ===")
    
    # Create framework
    framework = ExperimentalFramework()
    
    # Create test dataset with exact mapping verification
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
                {'item_id': 'startup_1', 'rating': 4.5},  # technology
            ],
            'sectors': ['healthcare', 'technology']
        }
    }
    
    print("1. Verify sector mappings:")
    for startup_id in ['startup_1', 'startup_2', 'startup_3']:
        sector = get_sector_for_startup(startup_id)
        print(f"   {startup_id} -> {sector}")
    
    print("\n2. Verify relevant sectors extraction:")
    for user_id, data in test_data.items():
        relevant_sectors = framework._extract_relevant_sectors(data)
        print(f"   {user_id}: {relevant_sectors}")
    
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
    
    print(f"\n3. Dataset split:")
    print(f"   Train users: {dataset_split.train_users}")
    print(f"   Test users: {dataset_split.test_users}")
    
    # Test matrix conversion
    print("\n4. Matrix conversion test:")
    train_matrix, user_mapping, item_mapping = framework._convert_to_matrix_format(dataset_split.train_data)
    print(f"   Train matrix shape: {train_matrix.shape}")
    print(f"   User mapping: {user_mapping}")
    print(f"   Item mapping: {item_mapping}")
    print(f"   Train matrix:\n{train_matrix}")
    
    # Test baseline training
    print("\n5. Baseline training:")
    pop_baseline = PopularityBaseline()
    pop_baseline.fit(train_matrix)
    print(f"   Item popularity: {pop_baseline.item_popularity}")
    
    # Test recommendations for each test user
    print("\n6. Generate recommendations:")
    for user_id in dataset_split.test_users:
        print(f"\n   User: {user_id}")
        
        if user_id in user_mapping:
            matrix_user_id = user_mapping[user_id]
            print(f"   Matrix user ID: {matrix_user_id}")
            
            recommendations = pop_baseline.recommend(matrix_user_id, n_recommendations=3)
            print(f"   Raw recommendations: {recommendations}")
            
            # Convert to item IDs
            item_ids = [item_mapping[item_idx] for item_idx, score in recommendations 
                      if item_idx in item_mapping]
            print(f"   Item IDs: {item_ids}")
            
            # Convert to sectors
            item_sectors = []
            for item_id in item_ids:
                sector = get_sector_for_startup(item_id)
                item_sectors.append(sector)
                print(f"     {item_id} -> {sector}")
            
            print(f"   Recommended sectors: {item_sectors}")
            
            # Get ground truth
            user_data = dataset_split.test_data[user_id]
            relevant_sectors = framework._extract_relevant_sectors(user_data)
            print(f"   Relevant sectors: {relevant_sectors}")
            
            # Check overlap
            overlap = set(item_sectors) & set(relevant_sectors)
            print(f"   Overlap: {overlap}")
            
        else:
            print(f"   User {user_id} not in user_mapping: {user_mapping}")

if __name__ == "__main__":
    debug_baseline_detailed()
