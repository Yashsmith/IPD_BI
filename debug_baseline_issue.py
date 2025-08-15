#!/usr/bin/env python3
"""
Debug script to diagnose why baseline methods return 0.0000 performance
"""

import sys
import os
sys.path.append('/Users/Sameer/Yashsmith/Research/IPD/bi_platform')

from src.evaluation.baseline_methods import PopularityBaseline, RandomBaseline
from src.evaluation.experimental_framework import ExperimentalFramework
import numpy as np

def debug_baseline_issue():
    print("=== BASELINE DEBUG SESSION ===")
    
    # Create sample user data
    sample_data = {
        'user_1': {
            'interactions': [
                {'item_id': 'startup_1', 'rating': 4.0},
                {'item_id': 'startup_2', 'rating': 5.0},
                {'item_id': 'startup_3', 'rating': 3.0}
            ],
            'sectors': ['technology', 'finance']
        },
        'user_2': {
            'interactions': [
                {'item_id': 'startup_4', 'rating': 4.0},
                {'item_id': 'startup_5', 'rating': 2.0},
                {'item_id': 'startup_6', 'rating': 5.0}
            ],
            'sectors': ['healthcare', 'retail']
        }
    }
    
    # Initialize framework
    framework = ExperimentalFramework()
    
    # Test matrix conversion
    print("\n1. Testing matrix conversion...")
    matrix, user_mapping, item_mapping = framework._convert_to_matrix_format(sample_data)
    print(f"Matrix shape: {matrix.shape}")
    print(f"User mapping: {user_mapping}")
    print(f"Item mapping: {item_mapping}")
    print(f"Matrix:\n{matrix}")
    
    # Test baseline training
    print("\n2. Testing PopularityBaseline...")
    pop_baseline = PopularityBaseline()
    pop_baseline.fit(matrix)
    print(f"Item popularity: {pop_baseline.item_popularity}")
    
    # Test recommendations
    print("\n3. Testing recommendations for user 0...")
    user_idx = 0
    recommendations = pop_baseline.recommend(user_idx, n_recommendations=3)
    print(f"Raw recommendations: {recommendations}")
    
    # Test sector mapping
    print("\n4. Testing sector mapping...")
    for item_idx, score in recommendations:
        if item_idx in item_mapping:
            item_id = item_mapping[item_idx]
            print(f"Item {item_idx} -> {item_id} (score: {score})")
            
            # Test the startup to sector mapping logic
            if isinstance(item_id, str) and item_id.startswith('startup_'):
                try:
                    startup_num = int(item_id.split('_')[1])
                    if startup_num <= 25:
                        sector = 'technology'
                    elif startup_num <= 50:
                        sector = 'finance'
                    elif startup_num <= 75:
                        sector = 'healthcare'
                    else:
                        sector = 'retail'
                    print(f"  -> Sector: {sector}")
                except (ValueError, IndexError):
                    print(f"  -> Sector: technology (default)")
    
    # Test relevant sectors extraction
    print("\n5. Testing relevant sectors extraction...")
    relevant_sectors_user1 = framework._extract_relevant_sectors(sample_data['user_1'])
    relevant_sectors_user2 = framework._extract_relevant_sectors(sample_data['user_2'])
    print(f"User 1 relevant sectors: {relevant_sectors_user1}")
    print(f"User 2 relevant sectors: {relevant_sectors_user2}")
    
    # Check sector mapping consistency
    print("\n6. Checking sector mapping consistency...")
    startup_to_sector_in_extract = {
        'startup_1': 'technology', 'startup_2': 'finance', 'startup_3': 'healthcare',
        'startup_4': 'technology', 'startup_5': 'finance', 'startup_6': 'healthcare'
    }
    
    startup_to_sector_in_baseline = {}
    for startup_id in ['startup_1', 'startup_2', 'startup_3', 'startup_4', 'startup_5', 'startup_6']:
        startup_num = int(startup_id.split('_')[1])
        if startup_num <= 25:
            sector = 'technology'
        elif startup_num <= 50:
            sector = 'finance'
        elif startup_num <= 75:
            sector = 'healthcare'
        else:
            sector = 'retail'
        startup_to_sector_in_baseline[startup_id] = sector
    
    print("Extract method mapping:")
    for k, v in startup_to_sector_in_extract.items():
        print(f"  {k} -> {v}")
    
    print("Baseline method mapping:")
    for k, v in startup_to_sector_in_baseline.items():
        print(f"  {k} -> {v}")
    
    # Check for mismatches
    mismatches = []
    for startup_id in startup_to_sector_in_extract:
        if startup_to_sector_in_extract[startup_id] != startup_to_sector_in_baseline[startup_id]:
            mismatches.append((startup_id, 
                             startup_to_sector_in_extract[startup_id], 
                             startup_to_sector_in_baseline[startup_id]))
    
    if mismatches:
        print(f"\n⚠️  SECTOR MAPPING MISMATCHES FOUND:")
        for startup_id, extract_sector, baseline_sector in mismatches:
            print(f"  {startup_id}: extract={extract_sector}, baseline={baseline_sector}")
    else:
        print("\n✅ Sector mappings are consistent")

if __name__ == "__main__":
    debug_baseline_issue()
