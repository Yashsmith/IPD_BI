"""
Standard Dataset Integration for Step 3.1 - Phase 3
Supports MovieLens, Amazon Reviews, and synthetic investment datasets
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import urllib.request
import zipfile
import gzip
import json
from collections import defaultdict, Counter
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    """Information about a dataset"""
    name: str
    description: str
    n_users: int
    n_items: int
    n_interactions: int
    sparsity: float
    rating_scale: Tuple[float, float]
    domain: str
    source: str

@dataclass
class DatasetSample:
    """Sample from a dataset"""
    user_id: str
    item_id: str
    rating: float
    timestamp: str
    additional_features: Optional[Dict[str, Any]] = None

class StandardDatasetManager:
    """
    Manager for standard datasets used in recommendation system research
    """
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.dataset_configs = {
            'movielens-1m': {
                'url': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
                'description': 'MovieLens 1M Dataset - 1 million ratings from 6000 users on 4000 movies',
                'domain': 'movies',
                'rating_scale': (1, 5)
            },
            'movielens-100k': {
                'url': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
                'description': 'MovieLens 100K Dataset - 100,000 ratings from 943 users on 1682 movies',
                'domain': 'movies', 
                'rating_scale': (1, 5)
            },
            'synthetic-investment': {
                'description': 'Synthetic Investment Recommendation Dataset',
                'domain': 'investment',
                'rating_scale': (1, 5)
            }
        }
        
        self.loaded_datasets = {}
        
    def download_dataset(self, dataset_name: str) -> bool:
        """Download a dataset if not already present, or use local files if available"""
        if dataset_name not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        config = self.dataset_configs[dataset_name]
        if 'url' not in config:
            logger.info(f"Dataset {dataset_name} is synthetic, no download needed")
            return True
        dataset_dir = self.data_dir / dataset_name
        # Special case: if movielens-100k and ml-100k dir exists, skip download
        if dataset_name == 'movielens-100k':
            local_ml100k = self.data_dir / dataset_name / 'ml-100k'
            alt_ml100k = self.data_dir / 'ml-100k'
            if local_ml100k.exists() or alt_ml100k.exists():
                logger.info(f"[DEBUG] Local MovieLens-100K found at {local_ml100k} or {alt_ml100k}, skipping download.")
                return True
        if dataset_dir.exists():
            logger.info(f"Dataset {dataset_name} already exists")
            return True
        logger.info(f"Downloading {dataset_name}...")
        try:
            url = config['url']
            filename = url.split('/')[-1]
            file_path = self.data_dir / filename
            urllib.request.urlretrieve(url, file_path)
            logger.info(f"Downloaded {filename}")
            if filename.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                logger.info(f"Extracted {filename}")
                file_path.unlink()
            return True
        except Exception as e:
            logger.error(f"[DEBUG] Failed to download {dataset_name}: {e}")
            # Always try to proceed with local files for movielens-100k
            if dataset_name == 'movielens-100k':
                local_ml100k = self.data_dir / dataset_name / 'ml-100k'
                alt_ml100k = self.data_dir / 'ml-100k'
                if local_ml100k.exists() or alt_ml100k.exists():
                    logger.info(f"[DEBUG] Proceeding with local MovieLens-100K files despite download error.")
                    return True
            return False
    
    def load_movielens_1m(self) -> Dict[str, Any]:
        """Load MovieLens 1M dataset"""
        
        dataset_dir = self.data_dir / 'movielens-1m' / 'ml-1m'
        
        if not dataset_dir.exists():
            logger.error(f"MovieLens 1M not found at {dataset_dir}")
            return {}
        
        # Load ratings
        ratings_file = dataset_dir / 'ratings.dat'
        users_file = dataset_dir / 'users.dat'
        movies_file = dataset_dir / 'movies.dat'
        
        # Read ratings (UserID::MovieID::Rating::Timestamp)
        ratings_data = []
        with open(ratings_file, 'r', encoding='latin-1') as f:
            for line in f:
                user_id, movie_id, rating, timestamp = line.strip().split('::')
                ratings_data.append({
                    'user_id': f'user_{user_id}',
                    'item_id': f'movie_{movie_id}',
                    'rating': int(rating),
                    'timestamp': datetime.fromtimestamp(int(timestamp)).isoformat()
                })
        
        # Read users (UserID::Gender::Age::Occupation::Zip-code)
        users_data = {}
        with open(users_file, 'r', encoding='latin-1') as f:
            for line in f:
                user_id, gender, age, occupation, zipcode = line.strip().split('::')
                users_data[f'user_{user_id}'] = {
                    'gender': gender,
                    'age': int(age),
                    'occupation': int(occupation),
                    'zipcode': zipcode
                }
        
        # Read movies (MovieID::Title::Genres)
        movies_data = {}
        with open(movies_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 3:
                    movie_id, title, genres = parts[0], parts[1], parts[2]
                    movies_data[f'movie_{movie_id}'] = {
                        'title': title,
                        'genres': genres.split('|')
                    }
        
        # Convert to user-centric format
        user_data = defaultdict(lambda: {'interactions': []})
        
        for rating in ratings_data:
            user_id = rating['user_id']
            user_data[user_id]['interactions'].append({
                'item_id': rating['item_id'],
                'rating': rating['rating'],
                'timestamp': rating['timestamp']
            })
            
            # Add user features
            if user_id in users_data:
                user_data[user_id].update(users_data[user_id])
        
        # Add dataset metadata
        n_users = len(user_data)
        n_items = len(movies_data)
        n_interactions = len(ratings_data)
        sparsity = 1 - (n_interactions / (n_users * n_items))
        
        dataset_info = DatasetInfo(
            name='MovieLens-1M',
            description='MovieLens 1M Dataset - Movies Recommendation',
            n_users=n_users,
            n_items=n_items,
            n_interactions=n_interactions,
            sparsity=sparsity,
            rating_scale=(1, 5),
            domain='movies',
            source='GroupLens Research'
        )
        
        return {
            'user_data': dict(user_data),
            'item_data': movies_data,
            'dataset_info': dataset_info,
            'raw_ratings': ratings_data
        }
    
    def load_movielens_100k(self) -> Dict[str, Any]:
        """Load MovieLens 100K dataset, trying all possible local paths and logging them."""
        paths_to_try = [
            self.data_dir / 'movielens-100k' / 'ml-100k',
            self.data_dir / 'ml-100k',
        ]
        dataset_dir = None
        for path in paths_to_try:
            logger.info(f"[DEBUG] Checking for MovieLens 100K at {path}")
            if path.exists():
                logger.info(f"[DEBUG] Found MovieLens 100K at {path}")
                dataset_dir = path
                break
        if dataset_dir is None:
            logger.error(f"[DEBUG] MovieLens 100K not found at any of: {[str(p) for p in paths_to_try]}")
            return {}
        # Proceed to load files from dataset_dir
        logger.info(f"[DEBUG] Proceeding to load MovieLens 100K from {dataset_dir}")
        
        # Load ratings (user id | item id | rating | timestamp)
        ratings_file = dataset_dir / 'u.data'
        users_file = dataset_dir / 'u.user'
        items_file = dataset_dir / 'u.item'
        
        # Read ratings
        ratings_data = []
        with open(ratings_file, 'r') as f:
            for line in f:
                user_id, item_id, rating, timestamp = line.strip().split('\t')
                ratings_data.append({
                    'user_id': f'user_{user_id}',
                    'item_id': f'movie_{item_id}',
                    'rating': int(rating),
                    'timestamp': datetime.fromtimestamp(int(timestamp)).isoformat()
                })
        
        # Read users (user id | age | gender | occupation | zip code)
        users_data = {}
        with open(users_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 5:
                    user_id, age, gender, occupation, zipcode = parts[:5]
                    users_data[f'user_{user_id}'] = {
                        'age': int(age),
                        'gender': gender,
                        'occupation': occupation,
                        'zipcode': zipcode
                    }
        
        # Read items (movie info)
        movies_data = {}
        with open(items_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 5:
                    movie_id, title = parts[0], parts[1]
                    movies_data[f'movie_{movie_id}'] = {
                        'title': title,
                        'genres': []  # Simplified for now
                    }
        
        # Convert to user-centric format
        user_data = defaultdict(lambda: {'interactions': []})
        
        for rating in ratings_data:
            user_id = rating['user_id']
            user_data[user_id]['interactions'].append({
                'item_id': rating['item_id'],
                'rating': rating['rating'],
                'timestamp': rating['timestamp']
            })
            
            # Add user features
            if user_id in users_data:
                user_data[user_id].update(users_data[user_id])
        
        # Dataset metadata
        n_users = len(user_data)
        n_items = len(movies_data)
        n_interactions = len(ratings_data)
        sparsity = 1 - (n_interactions / (n_users * n_items))
        
        dataset_info = DatasetInfo(
            name='MovieLens-100K',
            description='MovieLens 100K Dataset - Movies Recommendation',
            n_users=n_users,
            n_items=n_items,
            n_interactions=n_interactions,
            sparsity=sparsity,
            rating_scale=(1, 5),
            domain='movies',
            source='GroupLens Research'
        )
        
        return {
            'user_data': dict(user_data),
            'item_data': movies_data,
            'dataset_info': dataset_info,
            'raw_ratings': ratings_data
        }
    
    def generate_synthetic_investment_dataset(self, 
                                            n_users: int = 1000,
                                            n_items: int = 500,
                                            avg_interactions_per_user: int = 20,
                                            sparsity_target: float = 0.95) -> Dict[str, Any]:
        """Generate a synthetic investment recommendation dataset"""
        
        logger.info(f"Generating synthetic investment dataset: {n_users} users, {n_items} items")
        
        # Investment sectors
        sectors = [
            'Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer Goods',
            'Real Estate', 'Manufacturing', 'Telecommunications', 'Retail',
            'Transportation', 'Agriculture', 'Entertainment', 'Education'
        ]
        
        # Investment types
        investment_types = ['Stock', 'Bond', 'ETF', 'Mutual Fund', 'REIT', 'Commodity']
        
        # Risk levels
        risk_levels = ['Low', 'Medium', 'High']
        
        # Generate items (investments)
        items_data = {}
        for i in range(n_items):
            item_id = f'investment_{i:04d}'
            items_data[item_id] = {
                'name': f'Investment {i:04d}',
                'sector': random.choice(sectors),
                'type': random.choice(investment_types),
                'risk_level': random.choice(risk_levels),
                'market_cap': random.choice(['Small', 'Medium', 'Large']),
                'expected_return': round(random.uniform(0.02, 0.15), 3),
                'volatility': round(random.uniform(0.05, 0.40), 3)
            }
        
        # Generate users with investment preferences
        users_data = {}
        for i in range(n_users):
            user_id = f'investor_{i:04d}'
            
            # User demographics and preferences
            age = random.randint(25, 65)
            risk_tolerance = random.choice(risk_levels)
            preferred_sectors = random.sample(sectors, random.randint(2, 5))
            
            users_data[user_id] = {
                'age': age,
                'risk_tolerance': risk_tolerance,
                'preferred_sectors': preferred_sectors,
                'investment_experience': random.choice(['Beginner', 'Intermediate', 'Expert']),
                'portfolio_size': random.choice(['Small', 'Medium', 'Large'])
            }
        
        # Generate interactions based on user preferences
        user_data = {}
        all_ratings = []
        
        for user_id, user_profile in users_data.items():
            interactions = []
            
            # Number of interactions per user (following power law)
            n_interactions = max(5, int(np.random.exponential(avg_interactions_per_user)))
            n_interactions = min(n_interactions, n_items // 2)  # Cap at half the items
            
            # Select items based on user preferences
            preferred_items = []
            other_items = []
            
            for item_id, item_data in items_data.items():
                # Higher probability for preferred sectors and matching risk tolerance
                preference_score = 0
                
                if item_data['sector'] in user_profile['preferred_sectors']:
                    preference_score += 0.6
                
                if item_data['risk_level'] == user_profile['risk_tolerance']:
                    preference_score += 0.3
                else:
                    # Penalty for mismatched risk
                    risk_penalty = {'Low': 0.1, 'Medium': 0.0, 'High': -0.2}
                    preference_score += risk_penalty.get(item_data['risk_level'], 0)
                
                if preference_score > 0.3:
                    preferred_items.append((item_id, preference_score))
                else:
                    other_items.append((item_id, preference_score))
            
            # Sample items (more from preferred)
            selected_items = []
            
            # 70% from preferred items
            if preferred_items:
                preferred_items.sort(key=lambda x: x[1], reverse=True)
                n_preferred = min(int(n_interactions * 0.7), len(preferred_items))
                selected_items.extend([item[0] for item in preferred_items[:n_preferred]])
            
            # 30% from other items
            remaining = n_interactions - len(selected_items)
            if remaining > 0 and other_items:
                other_sample = random.sample(other_items, min(remaining, len(other_items)))
                selected_items.extend([item[0] for item in other_sample])
            
            # Generate ratings for selected items
            for item_id in selected_items:
                item_data = items_data[item_id]
                
                # Rating based on user-item compatibility
                base_rating = 3.0  # Neutral rating
                
                # Adjust based on preferences
                if item_data['sector'] in user_profile['preferred_sectors']:
                    base_rating += random.uniform(0.3, 0.8)
                
                if item_data['risk_level'] == user_profile['risk_tolerance']:
                    base_rating += random.uniform(0.2, 0.6)
                
                # Add some randomness
                base_rating += random.uniform(-0.5, 0.5)
                
                # Clamp to rating scale
                rating = max(1, min(5, round(base_rating)))
                
                # Generate timestamp (within last 2 years)
                days_ago = random.randint(0, 730)
                timestamp = (datetime.now() - timedelta(days=days_ago)).isoformat()
                
                interaction = {
                    'item_id': item_id,
                    'rating': rating,
                    'timestamp': timestamp
                }
                
                interactions.append(interaction)
                all_ratings.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': rating,
                    'timestamp': timestamp
                })
            
            # Add user profile to user data
            user_data[user_id] = {
                'interactions': interactions,
                **user_profile
            }
        
        # Calculate dataset statistics
        n_interactions = len(all_ratings)
        actual_sparsity = 1 - (n_interactions / (n_users * n_items))
        
        dataset_info = DatasetInfo(
            name='Synthetic Investment Dataset',
            description=f'Synthetic investment recommendation dataset with {n_users} users and {n_items} items',
            n_users=n_users,
            n_items=n_items,
            n_interactions=n_interactions,
            sparsity=actual_sparsity,
            rating_scale=(1, 5),
            domain='investment',
            source='Synthetic Generation'
        )
        
        logger.info(f"Generated dataset: {n_interactions} interactions, sparsity: {actual_sparsity:.3f}")
        
        return {
            'user_data': user_data,
            'item_data': items_data,
            'dataset_info': dataset_info,
            'raw_ratings': all_ratings
        }
    
    def load_dataset(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """Load a dataset by name"""
        
        if dataset_name in self.loaded_datasets:
            logger.info(f"Using cached dataset: {dataset_name}")
            return self.loaded_datasets[dataset_name]
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        if dataset_name == 'movielens-1m':
            if self.download_dataset(dataset_name):
                dataset = self.load_movielens_1m()
            else:
                dataset = {}
                
        elif dataset_name == 'movielens-100k':
            if self.download_dataset(dataset_name):
                dataset = self.load_movielens_100k()
            else:
                dataset = {}
                
        elif dataset_name == 'synthetic-investment':
            dataset = self.generate_synthetic_investment_dataset(**kwargs)
            
        else:
            logger.error(f"Unknown dataset: {dataset_name}")
            dataset = {}
        
        if dataset:
            self.loaded_datasets[dataset_name] = dataset
            logger.info(f"Successfully loaded {dataset_name}")
            
        return dataset
    
    def get_dataset_summary(self, dataset_name: str) -> Optional[DatasetInfo]:
        """Get summary information about a dataset"""
        
        if dataset_name not in self.loaded_datasets:
            dataset = self.load_dataset(dataset_name)
            if not dataset:
                return None
        
        return self.loaded_datasets[dataset_name].get('dataset_info')
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets"""
        return list(self.dataset_configs.keys())

def test_dataset_manager():
    """Test the standard dataset manager"""
    print("📊 TESTING STANDARD DATASET MANAGER")
    print("=" * 60)
    
    manager = StandardDatasetManager()
    
    print("📋 Available datasets:")
    for dataset in manager.list_available_datasets():
        print(f"  • {dataset}")
    
    # Test synthetic investment dataset
    print("\n🏦 Testing Synthetic Investment Dataset...")
    investment_data = manager.load_dataset('synthetic-investment', 
                                         n_users=100, n_items=50, avg_interactions_per_user=15)
    
    if investment_data:
        info = investment_data['dataset_info']
        print(f"  ✅ Generated: {info.n_users} users, {info.n_items} items")
        print(f"  📈 Interactions: {info.n_interactions}")
        print(f"  🕳️ Sparsity: {info.sparsity:.3f}")
        
        # Sample user
        sample_user = list(investment_data['user_data'].keys())[0]
        user_data = investment_data['user_data'][sample_user]
        print(f"  👤 Sample user: {len(user_data['interactions'])} interactions")
        print(f"      Risk tolerance: {user_data['risk_tolerance']}")
        print(f"      Preferred sectors: {user_data['preferred_sectors'][:3]}...")
    
    # Test MovieLens 100K (smaller, faster to test)
    print("\n🎬 Testing MovieLens 100K Dataset...")
    try:
        movielens_data = manager.load_dataset('movielens-100k')
        if movielens_data:
            info = movielens_data['dataset_info']
            print(f"  ✅ Loaded: {info.n_users} users, {info.n_items} items")
            print(f"  📈 Interactions: {info.n_interactions}")
            print(f"  🕳️ Sparsity: {info.sparsity:.3f}")
        else:
            print("  ⚠️ MovieLens 100K not downloaded (network/download issue)")
    except Exception as e:
        print(f"  ⚠️ MovieLens 100K error: {str(e)}")
    
    print("\n✅ Dataset Manager testing completed!")

if __name__ == "__main__":
    test_dataset_manager()
