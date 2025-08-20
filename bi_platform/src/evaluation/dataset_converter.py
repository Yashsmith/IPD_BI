"""
Dataset Converter for Standard Datasets - Step 3.1 Phase 3
Converts standard datasets to experimental framework format
"""

import sys
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from dataset_manager import StandardDatasetManager, DatasetInfo

logger = logging.getLogger(__name__)

class DatasetConverter:
    """
    Converts standard datasets to experimental framework format
    """
    
    def __init__(self):
        self.dataset_manager = StandardDatasetManager()
        
    def convert_to_experimental_format(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a standard dataset to experimental framework format
        
        Returns format used by ExperimentalFramework:
        {
            'user_data': {user_id: {'interactions': [{'item_id': str, 'rating': float, 'timestamp': str}]}},
            'item_data': {item_id: {metadata}},
            'dataset_info': DatasetInfo
        }
        """
        
        if not dataset:
            return {}
        
        # Extract components
        user_data = dataset.get('user_data', {})
        item_data = dataset.get('item_data', {})
        dataset_info = dataset.get('dataset_info')
        
        # Validate format
        if not user_data or not dataset_info:
            logger.error("Invalid dataset format")
            return {}
        
        # Convert interactions to ensure proper format
        converted_user_data = {}
        
        for user_id, user_profile in user_data.items():
            if 'interactions' not in user_profile:
                continue
                
            # Ensure interactions have required fields
            converted_interactions = []
            for interaction in user_profile['interactions']:
                if all(key in interaction for key in ['item_id', 'rating']):
                    converted_interaction = {
                        'item_id': str(interaction['item_id']),
                        'rating': float(interaction['rating']),
                        'timestamp': interaction.get('timestamp', ''),
                    }
                    converted_interactions.append(converted_interaction)
            
            # Copy user profile (excluding interactions)
            converted_profile = {k: v for k, v in user_profile.items() if k != 'interactions'}
            converted_profile['interactions'] = converted_interactions
            
            converted_user_data[str(user_id)] = converted_profile
        
        # Ensure item_data has string keys
        converted_item_data = {str(k): v for k, v in item_data.items()}
        
        return {
            'user_data': converted_user_data,
            'item_data': converted_item_data,
            'dataset_info': dataset_info
        }
    
    def create_train_test_split(self, 
                               dataset: Dict[str, Any], 
                               test_ratio: float = 0.2,
                               min_interactions: int = 5) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Create train/test split maintaining user presence in both sets
        """
        
        user_data = dataset['user_data']
        item_data = dataset['item_data']
        dataset_info = dataset['dataset_info']
        
        train_user_data = {}
        test_user_data = {}
        
        for user_id, user_profile in user_data.items():
            interactions = user_profile['interactions']
            
            # Skip users with too few interactions
            if len(interactions) < min_interactions:
                continue
            
            # Sort by timestamp if available
            if interactions and 'timestamp' in interactions[0] and interactions[0]['timestamp']:
                try:
                    interactions = sorted(interactions, key=lambda x: x['timestamp'])
                except:
                    pass  # Keep original order if sorting fails
            
            # Split interactions
            n_test = max(1, int(len(interactions) * test_ratio))
            train_interactions = interactions[:-n_test]
            test_interactions = interactions[-n_test:]
            
            # Create profiles
            train_profile = user_profile.copy()
            train_profile['interactions'] = train_interactions
            train_user_data[user_id] = train_profile
            
            test_profile = user_profile.copy()
            test_profile['interactions'] = test_interactions
            test_user_data[user_id] = test_profile
        
        # Update dataset info
        train_interactions_count = sum(len(user['interactions']) for user in train_user_data.values())
        test_interactions_count = sum(len(user['interactions']) for user in test_user_data.values())
        
        train_info = DatasetInfo(
            name=f"{dataset_info.name} (Train)",
            description=f"{dataset_info.description} - Training Set",
            n_users=len(train_user_data),
            n_items=dataset_info.n_items,
            n_interactions=train_interactions_count,
            sparsity=1 - (train_interactions_count / (len(train_user_data) * dataset_info.n_items)),
            rating_scale=dataset_info.rating_scale,
            domain=dataset_info.domain,
            source=dataset_info.source
        )
        
        test_info = DatasetInfo(
            name=f"{dataset_info.name} (Test)",
            description=f"{dataset_info.description} - Test Set",
            n_users=len(test_user_data),
            n_items=dataset_info.n_items,
            n_interactions=test_interactions_count,
            sparsity=1 - (test_interactions_count / (len(test_user_data) * dataset_info.n_items)),
            rating_scale=dataset_info.rating_scale,
            domain=dataset_info.domain,
            source=dataset_info.source
        )
        
        train_dataset = {
            'user_data': train_user_data,
            'item_data': item_data,
            'dataset_info': train_info
        }
        
        test_dataset = {
            'user_data': test_user_data,
            'item_data': item_data,
            'dataset_info': test_info
        }
        
        return train_dataset, test_dataset
    
    def subsample_dataset(self, 
                         dataset: Dict[str, Any],
                         max_users: Optional[int] = None,
                         max_items: Optional[int] = None,
                         min_interactions_per_user: int = 5) -> Dict[str, Any]:
        """
        Create a subsample of the dataset for faster experimentation
        """
        
        user_data = dataset['user_data']
        item_data = dataset['item_data']
        dataset_info = dataset['dataset_info']
        
        # Filter users by interaction count
        filtered_users = {}
        for user_id, user_profile in user_data.items():
            if len(user_profile['interactions']) >= min_interactions_per_user:
                filtered_users[user_id] = user_profile
        
        # Subsample users if needed
        if max_users and len(filtered_users) > max_users:
            import random
            selected_users = random.sample(list(filtered_users.keys()), max_users)
            filtered_users = {uid: filtered_users[uid] for uid in selected_users}
        
        # Get all items that appear in filtered users' interactions
        used_items = set()
        for user_profile in filtered_users.values():
            for interaction in user_profile['interactions']:
                used_items.add(interaction['item_id'])
        
        # Subsample items if needed
        if max_items and len(used_items) > max_items:
            import random
            selected_items = set(random.sample(list(used_items), max_items))
            
            # Filter interactions to only include selected items
            final_users = {}
            for user_id, user_profile in filtered_users.items():
                filtered_interactions = [
                    interaction for interaction in user_profile['interactions']
                    if interaction['item_id'] in selected_items
                ]
                
                # Keep users who still have enough interactions
                if len(filtered_interactions) >= min_interactions_per_user:
                    new_profile = user_profile.copy()
                    new_profile['interactions'] = filtered_interactions
                    final_users[user_id] = new_profile
            
            used_items = selected_items
            filtered_users = final_users
        
        # Filter item data
        filtered_items = {item_id: item_data[item_id] for item_id in used_items if item_id in item_data}
        
        # Update dataset info
        total_interactions = sum(len(user['interactions']) for user in filtered_users.values())
        new_sparsity = 1 - (total_interactions / (len(filtered_users) * len(filtered_items)))
        
        new_info = DatasetInfo(
            name=f"{dataset_info.name} (Subsample)",
            description=f"{dataset_info.description} - Subsampled",
            n_users=len(filtered_users),
            n_items=len(filtered_items),
            n_interactions=total_interactions,
            sparsity=new_sparsity,
            rating_scale=dataset_info.rating_scale,
            domain=dataset_info.domain,
            source=dataset_info.source
        )
        
        return {
            'user_data': filtered_users,
            'item_data': filtered_items,
            'dataset_info': new_info
        }
    
    def get_dataset_statistics(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive statistics about a dataset
        """
        
        user_data = dataset['user_data']
        dataset_info = dataset['dataset_info']
        
        # Interaction statistics
        interaction_counts = [len(user['interactions']) for user in user_data.values()]
        
        # Rating statistics
        all_ratings = []
        for user in user_data.values():
            for interaction in user['interactions']:
                all_ratings.append(interaction['rating'])
        
        # Item popularity
        item_counts = defaultdict(int)
        for user in user_data.values():
            for interaction in user['interactions']:
                item_counts[interaction['item_id']] += 1
        
        statistics = {
            'basic_info': {
                'name': dataset_info.name,
                'n_users': dataset_info.n_users,
                'n_items': dataset_info.n_items,
                'n_interactions': dataset_info.n_interactions,
                'sparsity': dataset_info.sparsity,
                'domain': dataset_info.domain
            },
            'interaction_stats': {
                'min_interactions_per_user': min(interaction_counts) if interaction_counts else 0,
                'max_interactions_per_user': max(interaction_counts) if interaction_counts else 0,
                'avg_interactions_per_user': sum(interaction_counts) / len(interaction_counts) if interaction_counts else 0,
                'median_interactions_per_user': sorted(interaction_counts)[len(interaction_counts)//2] if interaction_counts else 0
            },
            'rating_stats': {
                'min_rating': min(all_ratings) if all_ratings else 0,
                'max_rating': max(all_ratings) if all_ratings else 0,
                'avg_rating': sum(all_ratings) / len(all_ratings) if all_ratings else 0,
                'rating_scale': dataset_info.rating_scale
            },
            'item_stats': {
                'min_popularity': min(item_counts.values()) if item_counts else 0,
                'max_popularity': max(item_counts.values()) if item_counts else 0,
                'avg_popularity': sum(item_counts.values()) / len(item_counts) if item_counts else 0
            }
        }
        
        return statistics

def test_dataset_converter():
    """Test the dataset converter"""
    print("🔄 TESTING DATASET CONVERTER")
    print("=" * 60)
    
    converter = DatasetConverter()
    
    # Test with synthetic investment dataset
    print("🏦 Loading synthetic investment dataset...")
    raw_dataset = converter.dataset_manager.load_dataset('synthetic-investment', 
                                                        n_users=50, n_items=30)
    
    if not raw_dataset:
        print("  ❌ Failed to load dataset")
        return
    
    print("📊 Converting to experimental format...")
    converted_dataset = converter.convert_to_experimental_format(raw_dataset)
    
    if converted_dataset:
        print("  ✅ Conversion successful")
        
        # Get statistics
        stats = converter.get_dataset_statistics(converted_dataset)
        print(f"  📈 Users: {stats['basic_info']['n_users']}")
        print(f"  📦 Items: {stats['basic_info']['n_items']}")
        print(f"  🔗 Interactions: {stats['basic_info']['n_interactions']}")
        print(f"  ⭐ Avg rating: {stats['rating_stats']['avg_rating']:.2f}")
        
        # Test train/test split
        print("\n✂️ Creating train/test split...")
        train_dataset, test_dataset = converter.create_train_test_split(converted_dataset, test_ratio=0.2)
        
        print(f"  📚 Train: {train_dataset['dataset_info'].n_users} users, {train_dataset['dataset_info'].n_interactions} interactions")
        print(f"  🧪 Test: {test_dataset['dataset_info'].n_users} users, {test_dataset['dataset_info'].n_interactions} interactions")
        
        # Test subsampling
        print("\n🎯 Creating subsample...")
        subsample = converter.subsample_dataset(converted_dataset, max_users=25, max_items=20)
        
        if subsample:
            print(f"  📊 Subsample: {subsample['dataset_info'].n_users} users, {subsample['dataset_info'].n_items} items")
            print(f"  🔗 Interactions: {subsample['dataset_info'].n_interactions}")
    
    print("\n✅ Dataset Converter testing completed!")

if __name__ == "__main__":
    test_dataset_converter()
