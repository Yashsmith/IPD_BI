"""
Experimental Methodology Framework for GRPO-GRPO-P Research
Comprehensive experimental design for rigorous academic evaluation
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import json
from datetime import datetime
from collections import defaultdict
import itertools
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.evaluation.research_evaluator import RecommendationEvaluator, EvaluationMetrics, StatisticalAnalyzer
from src.evaluation.baseline_methods import BaselineRecommender, create_all_baselines, create_all_baselines_with_modern
from src.agents.hybrid_system import HybridRecommendationSystem, SystemUser
from .sector_mapping import get_sector_for_startup

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    experiment_name: str
    method_name: str
    hyperparameters: Dict[str, Any]
    dataset_config: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    random_seed: int = 42

@dataclass
class DatasetSplit:
    """Train/validation/test split for experiments"""
    train_users: List[str]
    val_users: List[str]
    test_users: List[str]
    train_data: Dict[str, Any]
    val_data: Dict[str, Any]
    test_data: Dict[str, Any]
    split_timestamp: str

@dataclass
class EnhancedDatasetSplit:
    """Enhanced dataset split with stratification information for Phase 2"""
    train_users: List[str]
    val_users: List[str] 
    test_users: List[str]
    train_data: Dict[str, Any]
    val_data: Dict[str, Any]
    test_data: Dict[str, Any]
    split_timestamp: str
    fold_id: int
    random_seed: int
    stratification_info: Dict[str, Any]
    split_metadata: Dict[str, Any]

@dataclass
class CrossValidationConfig:
    """Configuration for enhanced cross-validation"""
    n_folds: int = 10
    n_seeds: int = 5
    stratify_by: Optional[str] = 'interaction_count'  # 'interaction_count', 'rating_mean', 'activity_level'
    min_interactions_per_user: int = 5
    validation_ratio: float = 0.2
    ensure_balanced_folds: bool = True
    random_state_base: int = 42

@dataclass
class ExperimentRun:
    """Results from a single experimental run"""
    experiment_config: ExperimentConfig
    run_id: int
    fold_id: Optional[int]  # For cross-validation
    train_time: float
    test_time: float
    memory_usage: float
    test_metrics: EvaluationMetrics
    val_metrics: Optional[EvaluationMetrics]
    detailed_results: Dict[str, Any]
    timestamp: str
    status: str  # "completed", "failed", "running"
    error_message: Optional[str] = None

class EnhancedCrossValidator:
    """
    Enhanced cross-validation with stratified sampling, multiple seeds, and comprehensive reporting
    """
    
    def __init__(self, config: CrossValidationConfig = None):
        self.config = config or CrossValidationConfig()
        
    def _compute_user_statistics(self, user_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Compute statistics for each user for stratification"""
        user_stats = {}
        
        for user_id, data in user_data.items():
            interactions = data.get('interactions', [])
            
            stats = {
                'interaction_count': len(interactions),
                'rating_mean': np.mean([i.get('rating', 0) for i in interactions]) if interactions else 0,
                'rating_std': np.std([i.get('rating', 0) for i in interactions]) if interactions else 0,
                'unique_items': len(set(i.get('item_id', '') for i in interactions)),
                'activity_level': self._classify_activity_level(len(interactions))
            }
            
            user_stats[user_id] = stats
            
        return user_stats
    
    def _classify_activity_level(self, interaction_count: int) -> str:
        """Classify user activity level based on interaction count"""
        if interaction_count < 5:
            return 'low'
        elif interaction_count < 20:
            return 'medium'
        else:
            return 'high'
    
    def _create_stratification_labels(self, user_stats: Dict[str, Dict[str, Any]], 
                                    stratify_by: str) -> Dict[str, int]:
        """Create stratification labels for users"""
        user_labels = {}
        
        if stratify_by == 'interaction_count':
            # Stratify by interaction count quartiles
            counts = [stats['interaction_count'] for stats in user_stats.values()]
            quartiles = np.percentile(counts, [25, 50, 75])
            
            for user_id, stats in user_stats.items():
                count = stats['interaction_count']
                if count <= quartiles[0]:
                    label = 0  # Q1
                elif count <= quartiles[1]:
                    label = 1  # Q2
                elif count <= quartiles[2]:
                    label = 2  # Q3
                else:
                    label = 3  # Q4
                user_labels[user_id] = label
                
        elif stratify_by == 'activity_level':
            # Stratify by activity level
            activity_map = {'low': 0, 'medium': 1, 'high': 2}
            for user_id, stats in user_stats.items():
                user_labels[user_id] = activity_map[stats['activity_level']]
                
        elif stratify_by == 'rating_mean':
            # Stratify by mean rating quartiles
            ratings = [stats['rating_mean'] for stats in user_stats.values()]
            quartiles = np.percentile(ratings, [25, 50, 75])
            
            for user_id, stats in user_stats.items():
                rating = stats['rating_mean']
                if rating <= quartiles[0]:
                    label = 0
                elif rating <= quartiles[1]:
                    label = 1
                elif rating <= quartiles[2]:
                    label = 2
                else:
                    label = 3
                user_labels[user_id] = label
                
        return user_labels
    
    def _filter_users(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter users based on minimum interaction requirements"""
        filtered_data = {}
        
        for user_id, data in user_data.items():
            interactions = data.get('interactions', [])
            if len(interactions) >= self.config.min_interactions_per_user:
                filtered_data[user_id] = data
                
        return filtered_data
    
    def create_stratified_splits(self, user_data: Dict[str, Any], 
                               random_seed: int) -> List[EnhancedDatasetSplit]:
        """Create stratified k-fold cross-validation splits"""
        
        # Filter users
        filtered_data = self._filter_users(user_data)
        
        # Compute user statistics
        user_stats = self._compute_user_statistics(filtered_data)
        
        # Create stratification labels
        if self.config.stratify_by:
            user_labels = self._create_stratification_labels(user_stats, self.config.stratify_by)
            
            # Prepare data for sklearn
            user_ids = list(filtered_data.keys())
            labels = [user_labels[user_id] for user_id in user_ids]
            
            # Create stratified folds
            skf = StratifiedKFold(
                n_splits=self.config.n_folds, 
                shuffle=True, 
                random_state=random_seed
            )
            
            fold_indices = list(skf.split(user_ids, labels))
        else:
            # Regular k-fold
            user_ids = list(filtered_data.keys())
            kf = KFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=random_seed
            )
            
            fold_indices = list(kf.split(user_ids))
            user_labels = {user_id: 0 for user_id in user_ids}  # Dummy labels
        
        # Create enhanced splits
        splits = []
        
        for fold_idx, (train_val_idx, test_idx) in enumerate(fold_indices):
            # Get user IDs for this fold
            train_val_users = [user_ids[i] for i in train_val_idx]
            test_users = [user_ids[i] for i in test_idx]
            
            # Split train_val into train and validation
            np.random.seed(random_seed + fold_idx)  # Ensure reproducibility
            np.random.shuffle(train_val_users)
            
            val_size = int(len(train_val_users) * self.config.validation_ratio)
            val_users = train_val_users[:val_size]
            train_users = train_val_users[val_size:]
            
            # Create data dictionaries
            train_data = {user_id: filtered_data[user_id] for user_id in train_users}
            val_data = {user_id: filtered_data[user_id] for user_id in val_users}
            test_data = {user_id: filtered_data[user_id] for user_id in test_users}
            
            # Compute stratification info for this fold
            stratification_info = self._compute_fold_stratification_info(
                train_users, val_users, test_users, user_stats, user_labels
            )
            
            # Create split metadata
            split_metadata = {
                'train_size': len(train_users),
                'val_size': len(val_users),
                'test_size': len(test_users),
                'total_interactions_train': sum(len(data['interactions']) for data in train_data.values()),
                'total_interactions_val': sum(len(data['interactions']) for data in val_data.values()),
                'total_interactions_test': sum(len(data['interactions']) for data in test_data.values()),
            }
            
            split = EnhancedDatasetSplit(
                train_users=train_users,
                val_users=val_users,
                test_users=test_users,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                split_timestamp=datetime.now().isoformat(),
                fold_id=fold_idx,
                random_seed=random_seed,
                stratification_info=stratification_info,
                split_metadata=split_metadata
            )
            
            splits.append(split)
            
        return splits
    
    def _compute_fold_stratification_info(self, train_users: List[str], 
                                        val_users: List[str], 
                                        test_users: List[str],
                                        user_stats: Dict[str, Dict[str, Any]],
                                        user_labels: Dict[str, int]) -> Dict[str, Any]:
        """Compute stratification information for a fold"""
        
        def get_label_distribution(users):
            labels = [user_labels[user_id] for user_id in users]
            distribution = defaultdict(int)
            for label in labels:
                distribution[label] += 1
            return dict(distribution)
        
        def get_stats_summary(users):
            if not users:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            
            interactions = [user_stats[user_id]['interaction_count'] for user_id in users]
            return {
                'mean': np.mean(interactions),
                'std': np.std(interactions),
                'min': np.min(interactions),
                'max': np.max(interactions)
            }
        
        return {
            'train_label_distribution': get_label_distribution(train_users),
            'val_label_distribution': get_label_distribution(val_users),
            'test_label_distribution': get_label_distribution(test_users),
            'train_interaction_stats': get_stats_summary(train_users),
            'val_interaction_stats': get_stats_summary(val_users),
            'test_interaction_stats': get_stats_summary(test_users)
        }
    
    def create_multiple_seed_splits(self, user_data: Dict[str, Any]) -> Dict[int, List[EnhancedDatasetSplit]]:
        """Create cross-validation splits with multiple random seeds"""
        
        all_splits = {}
        
        for seed_idx in range(self.config.n_seeds):
            seed = self.config.random_state_base + seed_idx
            logger.info(f"Creating CV splits with seed {seed} ({seed_idx + 1}/{self.config.n_seeds})")
            
            splits = self.create_stratified_splits(user_data, seed)
            all_splits[seed] = splits
            
        return all_splits
    
    def generate_cv_report(self, all_splits: Dict[int, List[EnhancedDatasetSplit]]) -> Dict[str, Any]:
        """Generate comprehensive cross-validation report"""
        
        report = {
            'cv_configuration': {
                'n_folds': self.config.n_folds,
                'n_seeds': self.config.n_seeds,
                'stratify_by': self.config.stratify_by,
                'min_interactions_per_user': self.config.min_interactions_per_user,
                'validation_ratio': self.config.validation_ratio
            },
            'fold_statistics': {},
            'seed_consistency': {},
            'overall_statistics': {}
        }
        
        # Analyze fold statistics across seeds
        for seed, splits in all_splits.items():
            seed_stats = []
            
            for split in splits:
                fold_stats = {
                    'fold_id': split.fold_id,
                    'train_size': split.split_metadata['train_size'],
                    'val_size': split.split_metadata['val_size'],
                    'test_size': split.split_metadata['test_size'],
                    'train_interactions': split.split_metadata['total_interactions_train'],
                    'val_interactions': split.split_metadata['total_interactions_val'],
                    'test_interactions': split.split_metadata['total_interactions_test']
                }
                seed_stats.append(fold_stats)
                
            report['fold_statistics'][seed] = seed_stats
        
        # Compute overall statistics
        all_train_sizes = []
        all_test_sizes = []
        all_val_sizes = []
        
        for seed, splits in all_splits.items():
            for split in splits:
                all_train_sizes.append(split.split_metadata['train_size'])
                all_val_sizes.append(split.split_metadata['val_size'])
                all_test_sizes.append(split.split_metadata['test_size'])
        
        report['overall_statistics'] = {
            'train_size_stats': {
                'mean': np.mean(all_train_sizes),
                'std': np.std(all_train_sizes),
                'min': np.min(all_train_sizes),
                'max': np.max(all_train_sizes)
            },
            'val_size_stats': {
                'mean': np.mean(all_val_sizes),
                'std': np.std(all_val_sizes),
                'min': np.min(all_val_sizes),
                'max': np.max(all_val_sizes)
            },
            'test_size_stats': {
                'mean': np.mean(all_test_sizes),
                'std': np.std(all_test_sizes),
                'min': np.min(all_test_sizes),
                'max': np.max(all_test_sizes)
            }
        }
        
        return report

class ExperimentalFramework:
    """
    Comprehensive experimental framework for recommendation system research
    Handles dataset preparation, cross-validation, hyperparameter tuning, and evaluation
    """
    
    def __init__(self, 
                 results_dir: str = "experimental_results",
                 random_seed: int = 42):
        """
        Initialize experimental framework
        
        Args:
            results_dir: Directory to store experimental results
            random_seed: Global random seed for reproducibility
        """
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Initialize components
        self.evaluator = RecommendationEvaluator()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Initialize enhanced cross-validator for Phase 2
        self.enhanced_cv = None  # Will be initialized when needed
        
        # Experiment tracking
        self.experiment_results: List[ExperimentRun] = []
        self.baseline_methods = create_all_baselines_with_modern()  # Phase 2 - Modern baselines
        
        logger.info(f"ExperimentalFramework initialized with results dir: {results_dir}")
    
    def create_dataset_splits(self, 
                            user_data: Dict[str, Any],
                            split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
                            temporal_split: bool = False,
                            stratify_by: Optional[str] = None) -> List[DatasetSplit]:
        """
        Create train/validation/test splits for experiments
        
        Args:
            user_data: Dictionary with user interaction data
            split_ratios: (train, val, test) ratios
            temporal_split: Whether to split by time (chronological)
            stratify_by: Attribute to stratify split by (e.g., user_type)
            
        Returns:
            List of DatasetSplit objects
        """
        
        user_ids = list(user_data.keys())
        n_users = len(user_ids)
        
        train_ratio, val_ratio, test_ratio = split_ratios
        
        if temporal_split:
            # Sort users by their first interaction timestamp
            def get_first_timestamp(user_id):
                interactions = user_data[user_id].get('interactions', [])
                if interactions:
                    return min(interaction.get('timestamp', '9999-12-31') for interaction in interactions)
                return '9999-12-31'
            
            user_ids.sort(key=get_first_timestamp)
        else:
            # Random shuffle for random split
            np.random.shuffle(user_ids)
        
        # Calculate split indices
        train_end = int(n_users * train_ratio)
        val_end = int(n_users * (train_ratio + val_ratio))
        
        train_users = user_ids[:train_end]
        val_users = user_ids[train_end:val_end]
        test_users = user_ids[val_end:]
        
        # Create data splits
        train_data = {user_id: user_data[user_id] for user_id in train_users}
        val_data = {user_id: user_data[user_id] for user_id in val_users}
        test_data = {user_id: user_data[user_id] for user_id in test_users}
        
        split = DatasetSplit(
            train_users=train_users,
            val_users=val_users,
            test_users=test_users,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            split_timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Created dataset split: {len(train_users)} train, {len(val_users)} val, {len(test_users)} test")
        
        return [split]
    
    def create_cross_validation_splits(self, 
                                     user_data: Dict[str, Any],
                                     n_folds: int = 5,
                                     stratify_by: Optional[str] = None) -> List[DatasetSplit]:
        """
        Create k-fold cross-validation splits
        
        Args:
            user_data: Dictionary with user interaction data
            n_folds: Number of folds for cross-validation
            stratify_by: Attribute to stratify by
            
        Returns:
            List of DatasetSplit objects (one per fold)
        """
        
        user_ids = list(user_data.keys())
        n_users = len(user_ids)
        
        # Shuffle users
        np.random.shuffle(user_ids)
        
        # Create folds
        fold_size = n_users // n_folds
        splits = []
        
        for fold in range(n_folds):
            # Test set for this fold
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_users
            test_users = user_ids[test_start:test_end]
            
            # Training + validation set
            remaining_users = user_ids[:test_start] + user_ids[test_end:]
            
            # Split remaining into train and validation
            val_size = len(remaining_users) // 5  # 20% for validation
            val_users = remaining_users[:val_size]
            train_users = remaining_users[val_size:]
            
            # Create data splits
            train_data = {user_id: user_data[user_id] for user_id in train_users}
            val_data = {user_id: user_data[user_id] for user_id in val_users}
            test_data = {user_id: user_data[user_id] for user_id in test_users}
            
            split = DatasetSplit(
                train_users=train_users,
                val_users=val_users,
                test_users=test_users,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                split_timestamp=datetime.now().isoformat()
            )
            
            splits.append(split)
        
        logger.info(f"Created {n_folds}-fold cross-validation splits")
        return splits
    
    def create_enhanced_cross_validation_splits(self, 
                                              user_data: Dict[str, Any],
                                              cv_config: CrossValidationConfig = None) -> Dict[int, List[EnhancedDatasetSplit]]:
        """
        Create enhanced cross-validation splits with stratification and multiple seeds (Phase 2)
        
        Args:
            user_data: Dictionary with user interaction data
            cv_config: Configuration for enhanced cross-validation
            
        Returns:
            Dictionary mapping seeds to lists of EnhancedDatasetSplit objects
        """
        
        if cv_config is None:
            cv_config = CrossValidationConfig()
        
        # Initialize enhanced cross-validator if needed
        if self.enhanced_cv is None:
            self.enhanced_cv = EnhancedCrossValidator(cv_config)
        
        logger.info(f"Creating enhanced CV splits: {cv_config.n_folds} folds, {cv_config.n_seeds} seeds, stratify by {cv_config.stratify_by}")
        
        # Create multiple seed splits
        all_splits = self.enhanced_cv.create_multiple_seed_splits(user_data)
        
        logger.info(f"Created enhanced {cv_config.n_folds}-fold CV splits with {cv_config.n_seeds} random seeds")
        return all_splits
    
    def generate_cv_report(self, all_splits: Dict[int, List[EnhancedDatasetSplit]]) -> Dict[str, Any]:
        """Generate comprehensive cross-validation report for Phase 2"""
        
        if self.enhanced_cv is None:
            raise ValueError("Enhanced cross-validator not initialized")
            
        return self.enhanced_cv.generate_cv_report(all_splits)
    
    def generate_hyperparameter_grid(self, 
                                   method_name: str,
                                   param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate hyperparameter grid for grid search
        
        Args:
            method_name: Name of the method
            param_ranges: Dictionary with parameter names and value ranges
            
        Returns:
            List of hyperparameter combinations
        """
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        hyperparameter_grid = []
        for combination in combinations:
            params = dict(zip(param_names, combination))
            hyperparameter_grid.append(params)
        
        logger.info(f"Generated {len(hyperparameter_grid)} hyperparameter combinations for {method_name}")
        return hyperparameter_grid
    
    def run_baseline_experiment(self, 
                              baseline: BaselineRecommender,
                              dataset_split: DatasetSplit,
                              config: ExperimentConfig) -> ExperimentRun:
        """
        Run experiment with a baseline method
        
        Args:
            baseline: Baseline recommendation method
            dataset_split: Train/test split
            config: Experiment configuration
            
        Returns:
            ExperimentRun with results
        """
        
        start_time = datetime.now()
        
        try:
            # Convert ALL data to format expected by baseline (need mappings for all users)
            all_data = {**dataset_split.train_data, **dataset_split.test_data}
            full_matrix, user_mapping, item_mapping = self._convert_to_matrix_format(all_data)
            
            # Create training matrix with only training users
            train_user_indices = [user_mapping[uid] for uid in dataset_split.train_users if uid in user_mapping]
            train_matrix = full_matrix[train_user_indices, :]
            
            # Train baseline
            train_start = datetime.now()
            baseline.fit(train_matrix)
            train_time = (datetime.now() - train_start).total_seconds()
            
            # Generate recommendations for test users
            test_start = datetime.now()
            
            # Dual-level baseline evaluation
            sector_recommendations = {}
            item_recommendations = {}
            sector_ground_truth = {}
            item_ground_truth = {}
            
            for user_id in dataset_split.test_users:
                if user_id in user_mapping:
                    matrix_user_id = user_mapping[user_id]
                    recommendations = baseline.recommend(matrix_user_id, n_recommendations=10)
                    
                    # Convert back to original item IDs
                    item_ids = [item_mapping[item_idx] for item_idx, score in recommendations 
                              if item_idx in item_mapping]
                    
                    # Item-level recommendations (direct from baseline)
                    item_recommendations[user_id] = item_ids
                    
                    # Convert item IDs to sectors for sector-level evaluation
                    from .sector_mapping import get_sector_for_startup
                    item_sectors = []
                    for item_id in item_ids:
                        item_sectors.append(get_sector_for_startup(item_id))
                    
                    # Remove duplicates while preserving order for sector-level
                    seen = set()
                    unique_sectors = []
                    for sector in item_sectors:
                        if sector not in seen:
                            seen.add(sector)
                            unique_sectors.append(sector)
                    
                    sector_recommendations[user_id] = unique_sectors
                    
                    # Get dual-level ground truth
                    user_data = dataset_split.test_data[user_id]
                    sector_ground_truth[user_id] = self._extract_relevant_sectors(user_data)
                    item_ground_truth[user_id] = self._extract_relevant_items(user_data)
            
            test_time = (datetime.now() - test_start).total_seconds()
            
            # Dual-level evaluation for baselines
            sector_metrics = self._evaluate_recommendations(sector_recommendations, sector_ground_truth)
            item_metrics = self._evaluate_recommendations(item_recommendations, item_ground_truth)
            
            # Combine metrics
            combined_metrics = self._combine_dual_level_metrics(sector_metrics, item_metrics)
            
            # Create experiment run
            experiment_run = ExperimentRun(
                experiment_config=config,
                run_id=len(self.experiment_results),
                fold_id=None,
                train_time=train_time,
                test_time=test_time,
                memory_usage=0.0,  # Could implement memory tracking
                test_metrics=combined_metrics,
                val_metrics=None,
                detailed_results={
                    'n_train_users': len(dataset_split.train_users),
                    'n_test_users': len(dataset_split.test_users),
                    'baseline_name': baseline.get_name(),
                    'dual_level_evaluation': {
                        'sector_metrics': sector_metrics,
                        'item_metrics': item_metrics
                    }
                },
                timestamp=start_time.isoformat(),
                status="completed"
            )
            
            logger.info(f"Completed baseline experiment: {baseline.get_name()}")
            return experiment_run
            
        except Exception as e:
            logger.error(f"Baseline experiment failed: {e}")
            
            # Return failed experiment run
            experiment_run = ExperimentRun(
                experiment_config=config,
                run_id=len(self.experiment_results),
                fold_id=None,
                train_time=0.0,
                test_time=0.0,
                memory_usage=0.0,
                test_metrics=EvaluationMetrics({}, {}, {}, 0.0, 0.0, 0.0, 0.0, 0.0),
                val_metrics=None,
                detailed_results={},
                timestamp=start_time.isoformat(),
                status="failed",
                error_message=str(e)
            )
            
            return experiment_run
    
    def run_grpo_grpo_p_experiment(self, 
                                 dataset_split: DatasetSplit,
                                 config: ExperimentConfig) -> ExperimentRun:
        """
        Run experiment with GRPO-GRPO-P hybrid system
        
        Args:
            dataset_split: Train/test split
            config: Experiment configuration
            
        Returns:
            ExperimentRun with results
        """
        
        start_time = datetime.now()
        
        try:
            # Initialize hybrid system
            hybrid_system = HybridRecommendationSystem()
            
            # Register training users
            train_start = datetime.now()
            for user_id in dataset_split.train_users:
                user_data = dataset_split.train_data[user_id]
                system_user = self._convert_to_system_user(user_id, user_data)
                hybrid_system.register_user(system_user)
                
                # Simulate training interactions
                self._simulate_training_interactions(hybrid_system, user_id, user_data)
            
            train_time = (datetime.now() - train_start).total_seconds()
            
            # Generate recommendations for test users
            test_start = datetime.now()
            
            # Dual-level evaluation: collect both sector and item recommendations
            sector_recommendations = {}
            item_recommendations = {}
            sector_ground_truth = {}
            item_ground_truth = {}
            
            for user_id in dataset_split.test_users:
                user_data = dataset_split.test_data[user_id]
                system_user = self._convert_to_system_user(user_id, user_data)
                hybrid_system.register_user(system_user)
                
                # Get recommendations from hybrid system
                recommendations = hybrid_system.get_recommendations(user_id, max_recommendations=10)
                
                # Extract sector-level recommendations
                recommended_sectors = [rec.sector for rec in recommendations]
                sector_recommendations[user_id] = recommended_sectors
                
                # Extract item-level recommendations (from all recommended sectors)
                recommended_items = []
                for rec in recommendations:
                    if hasattr(rec, 'recommended_items') and rec.recommended_items:
                        recommended_items.extend(rec.recommended_items[:3])  # Top 3 items per sector
                item_recommendations[user_id] = recommended_items[:10]  # Limit to 10 total items
                
                # Get dual-level ground truth
                sector_ground_truth[user_id] = self._extract_relevant_sectors(user_data)
                item_ground_truth[user_id] = self._extract_relevant_items(user_data)
            
            test_time = (datetime.now() - test_start).total_seconds()
            
            # Dual-level evaluation
            sector_metrics = self._evaluate_recommendations(sector_recommendations, sector_ground_truth)
            item_metrics = self._evaluate_recommendations(item_recommendations, item_ground_truth)
            
            # Combine metrics with prefixes
            combined_metrics = self._combine_dual_level_metrics(sector_metrics, item_metrics)
            
            # Get system statistics
            system_stats = hybrid_system.get_system_stats()
            
            # Create experiment run
            experiment_run = ExperimentRun(
                experiment_config=config,
                run_id=len(self.experiment_results),
                fold_id=None,
                train_time=train_time,
                test_time=test_time,
                memory_usage=0.0,
                test_metrics=combined_metrics,
                val_metrics=None,
                detailed_results={
                    'n_train_users': len(dataset_split.train_users),
                    'n_test_users': len(dataset_split.test_users),
                    'system_stats': system_stats,
                    'hyperparameters': config.hyperparameters,
                    'dual_level_evaluation': {
                        'sector_metrics': sector_metrics,
                        'item_metrics': item_metrics
                    }
                },
                timestamp=start_time.isoformat(),
                status="completed"
            )
            
            logger.info(f"Completed GRPO-GRPO-P experiment")
            return experiment_run
            
        except Exception as e:
            logger.error(f"GRPO-GRPO-P experiment failed: {e}")
            
            experiment_run = ExperimentRun(
                experiment_config=config,
                run_id=len(self.experiment_results),
                fold_id=None,
                train_time=0.0,
                test_time=0.0,
                memory_usage=0.0,
                test_metrics=EvaluationMetrics({}, {}, {}, 0.0, 0.0, 0.0, 0.0, 0.0),
                val_metrics=None,
                detailed_results={},
                timestamp=start_time.isoformat(),
                status="failed",
                error_message=str(e)
            )
            
            return experiment_run
    
    def run_ablation_study(self, 
                          dataset_splits: List[DatasetSplit],
                          ablation_configs: List[Dict[str, Any]]) -> List[ExperimentRun]:
        """
        Run ablation study to understand component contributions
        
        Args:
            dataset_splits: List of dataset splits
            ablation_configs: Different configurations to test
            
        Returns:
            List of ExperimentRun results
        """
        
        ablation_results = []
        
        for config in ablation_configs:
            config_name = config.get('name', 'unnamed_ablation')
            logger.info(f"Running ablation: {config_name}")
            
            for split_idx, split in enumerate(dataset_splits):
                experiment_config = ExperimentConfig(
                    experiment_name=f"ablation_{config_name}",
                    method_name="GRPO-GRPO-P-Ablation",
                    hyperparameters=config,
                    dataset_config={},
                    evaluation_config={}
                )
                
                result = self.run_grpo_grpo_p_experiment(split, experiment_config)
                result.fold_id = split_idx
                ablation_results.append(result)
        
        return ablation_results
    
    def run_hyperparameter_search(self, 
                                dataset_splits: List[DatasetSplit],
                                param_grid: List[Dict[str, Any]],
                                method_name: str = "GRPO-GRPO-P") -> List[ExperimentRun]:
        """
        Run hyperparameter search
        
        Args:
            dataset_splits: List of dataset splits
            param_grid: Grid of hyperparameters to search
            method_name: Name of the method
            
        Returns:
            List of ExperimentRun results
        """
        
        search_results = []
        
        for param_idx, params in enumerate(param_grid):
            logger.info(f"Running hyperparameter config {param_idx + 1}/{len(param_grid)}: {params}")
            
            for split_idx, split in enumerate(dataset_splits):
                experiment_config = ExperimentConfig(
                    experiment_name=f"hyperparam_search_{param_idx}",
                    method_name=method_name,
                    hyperparameters=params,
                    dataset_config={},
                    evaluation_config={}
                )
                
                result = self.run_grpo_grpo_p_experiment(split, experiment_config)
                result.fold_id = split_idx
                search_results.append(result)
        
        return search_results
    
    def compare_methods(self, 
                       method_results: Dict[str, List[ExperimentRun]],
                       primary_metric: str = "ndcg_at_k") -> Dict[str, Any]:
        """
        Compare different methods statistically
        
        Args:
            method_results: Dictionary mapping method names to experiment results
            primary_metric: Primary metric for comparison
            
        Returns:
            Dictionary with comparison results
        """
        
        comparison_results = {}
        method_names = list(method_results.keys())
        
        # Extract metric values for each method
        method_scores = {}
        for method_name, results in method_results.items():
            scores = []
            for result in results:
                if result.status == "completed":
                    metric_value = getattr(result.test_metrics, primary_metric, {})
                    if isinstance(metric_value, dict) and 5 in metric_value:
                        scores.append(metric_value[5])  # Use @5 metric
                    elif isinstance(metric_value, (int, float)):
                        scores.append(metric_value)
            method_scores[method_name] = scores
        
        # Pairwise comparisons
        pairwise_comparisons = {}
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                method_a = method_names[i]
                method_b = method_names[j]
                
                scores_a = method_scores[method_a]
                scores_b = method_scores[method_b]
                
                if len(scores_a) > 0 and len(scores_b) > 0:
                    comparison = self.statistical_analyzer.compare_methods(
                        scores_a, scores_b, method_a, method_b, primary_metric
                    )
                    
                    pair_key = f"{method_a}_vs_{method_b}"
                    pairwise_comparisons[pair_key] = comparison
        
        # Summary statistics
        summary_stats = {}
        for method_name, scores in method_scores.items():
            if len(scores) > 0:
                summary_stats[method_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'median': np.median(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'n_runs': len(scores)
                }
        
        comparison_results = {
            'summary_statistics': summary_stats,
            'pairwise_comparisons': pairwise_comparisons,
            'primary_metric': primary_metric,
            'method_scores': method_scores
        }
        
        return comparison_results
    
    def save_experiment_results(self, results: List[ExperimentRun], filename: str):
        """Save experiment results to file"""
        
        results_path = self.results_dir / filename
        
        # Convert to serializable format
        serializable_results = []
        for result in results:
            result_dict = {
                'experiment_config': {
                    'experiment_name': result.experiment_config.experiment_name,
                    'method_name': result.experiment_config.method_name,
                    'hyperparameters': result.experiment_config.hyperparameters,
                    'random_seed': result.experiment_config.random_seed
                },
                'run_id': result.run_id,
                'fold_id': result.fold_id,
                'train_time': result.train_time,
                'test_time': result.test_time,
                'memory_usage': result.memory_usage,
                'test_metrics': {
                    'precision_at_k': result.test_metrics.precision_at_k,
                    'recall_at_k': result.test_metrics.recall_at_k,
                    'ndcg_at_k': result.test_metrics.ndcg_at_k,
                    'map_score': result.test_metrics.map_score,
                    'mrr_score': result.test_metrics.mrr_score,
                    'diversity': result.test_metrics.diversity,
                    'novelty': result.test_metrics.novelty,
                    'coverage': result.test_metrics.coverage
                },
                'timestamp': result.timestamp,
                'status': result.status,
                'error_message': result.error_message
            }
            serializable_results.append(result_dict)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved {len(results)} experiment results to {results_path}")
    
    def _convert_to_matrix_format(self, user_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict, Dict]:
        """Convert user data to matrix format for baselines"""
        
        # Extract all users and items
        all_users = list(user_data.keys())
        all_items = set()
        
        for user_id, data in user_data.items():
            interactions = data.get('interactions', [])
            for interaction in interactions:
                all_items.add(interaction.get('item_id', ''))
        
        all_items = list(all_items)
        
        # Create mappings
        user_mapping = {user_id: idx for idx, user_id in enumerate(all_users)}
        item_mapping = {item_id: idx for idx, item_id in enumerate(all_items)}
        reverse_item_mapping = {idx: item_id for item_id, idx in item_mapping.items()}
        
        # Create matrix
        matrix = np.zeros((len(all_users), len(all_items)))
        
        for user_id, data in user_data.items():
            user_idx = user_mapping[user_id]
            interactions = data.get('interactions', [])
            
            for interaction in interactions:
                item_id = interaction.get('item_id', '')
                if item_id in item_mapping:
                    item_idx = item_mapping[item_id]
                    rating = interaction.get('rating', 1.0)
                    matrix[user_idx, item_idx] = rating
        
        return matrix, user_mapping, reverse_item_mapping
    
    def _convert_to_system_user(self, user_id: str, user_data: Dict[str, Any]) -> SystemUser:
        """Convert user data to SystemUser object"""
        
        return SystemUser(
            user_id=user_id,
            role=user_data.get('role', 'investor'),
            sectors=user_data.get('sectors', ['technology', 'finance']),
            location=user_data.get('location', 'mumbai'),
            capital_range=user_data.get('capital_range', 'medium'),
            risk_appetite=user_data.get('risk_appetite', 'moderate'),
            experience_level=user_data.get('experience_level', 'intermediate')
        )
    
    def _simulate_training_interactions(self, system: HybridRecommendationSystem, 
                                      user_id: str, user_data: Dict[str, Any]):
        """Simulate training interactions for a user"""
        
        interactions = user_data.get('interactions', [])
        for interaction in interactions:
            feedback_score = interaction.get('rating', 1.0)
            # Normalize to [-1, 1] range
            feedback_score = (feedback_score - 2.5) / 2.5
            
            system.provide_feedback(
                user_id=user_id,
                recommendation_id=interaction.get('item_id', ''),
                feedback_type='rated',
                feedback_score=feedback_score
            )
    
    def _extract_relevant_items(self, user_data: Dict[str, Any]) -> List[str]:
        """Extract relevant items from user data"""
        
        interactions = user_data.get('interactions', [])
        relevant_items = []
        
        for interaction in interactions:
            rating = interaction.get('rating', 0.0)
            if rating >= 3.0:  # Consider rating >= 3 as relevant
                relevant_items.append(interaction.get('item_id', ''))
        
        return relevant_items
    
    def _extract_relevant_sectors(self, user_data: Dict[str, Any]) -> List[str]:
        """Extract relevant sectors from user data by mapping high-rated startups to sectors"""
        
        interactions = user_data.get('interactions', [])
        relevant_sectors = set()
        
        # Simple mapping: startup ID to sector (in real system this would be from a database)
        startup_to_sector = {
            'startup_1': 'technology', 'startup_2': 'finance', 'startup_3': 'healthcare',
            'startup_4': 'technology', 'startup_5': 'finance', 'startup_6': 'healthcare', 
            'startup_7': 'retail', 'startup_8': 'technology', 'startup_9': 'finance',
            'startup_10': 'healthcare', 'startup_11': 'technology', 'startup_12': 'finance',
            'startup_13': 'healthcare', 'startup_14': 'retail', 'startup_15': 'technology',
            'startup_16': 'finance', 'startup_17': 'healthcare', 'startup_18': 'retail',
            'startup_19': 'technology', 'startup_20': 'finance', 'startup_21': 'healthcare',
            'startup_22': 'retail', 'startup_23': 'technology', 'startup_24': 'finance',
            'startup_25': 'healthcare', 'startup_26': 'retail', 'startup_27': 'technology',
            'startup_28': 'finance', 'startup_29': 'healthcare', 'startup_30': 'retail',
            'startup_31': 'technology', 'startup_32': 'finance', 'startup_33': 'healthcare',
            'startup_34': 'retail', 'startup_35': 'technology', 'startup_36': 'finance',
            'startup_37': 'healthcare', 'startup_38': 'retail', 'startup_39': 'technology',
            'startup_40': 'finance', 'startup_41': 'healthcare', 'startup_42': 'retail',
            'startup_43': 'technology', 'startup_44': 'finance', 'startup_45': 'healthcare',
            'startup_46': 'retail', 'startup_47': 'technology', 'startup_48': 'finance',
            'startup_49': 'healthcare', 'startup_50': 'retail', 'startup_51': 'technology',
            'startup_52': 'finance', 'startup_53': 'healthcare', 'startup_54': 'retail',
            'startup_55': 'technology', 'startup_56': 'finance', 'startup_57': 'healthcare',
            'startup_58': 'retail', 'startup_59': 'technology', 'startup_60': 'finance',
            'startup_61': 'healthcare', 'startup_62': 'retail', 'startup_63': 'technology',
            'startup_64': 'finance', 'startup_65': 'healthcare', 'startup_66': 'retail',
            'startup_67': 'technology', 'startup_68': 'finance', 'startup_69': 'healthcare',
            'startup_70': 'retail', 'startup_71': 'technology', 'startup_72': 'finance',
            'startup_73': 'healthcare', 'startup_74': 'retail', 'startup_75': 'technology',
            'startup_76': 'finance', 'startup_77': 'healthcare', 'startup_78': 'retail',
            'startup_79': 'technology', 'startup_80': 'finance', 'startup_81': 'healthcare',
            'startup_82': 'retail', 'startup_83': 'technology', 'startup_84': 'finance',
            'startup_85': 'healthcare', 'startup_86': 'retail', 'startup_87': 'technology',
            'startup_88': 'finance', 'startup_89': 'healthcare', 'startup_90': 'retail',
            'startup_91': 'technology', 'startup_92': 'finance', 'startup_93': 'healthcare',
            'startup_94': 'retail', 'startup_95': 'technology', 'startup_96': 'finance',
            'startup_97': 'healthcare', 'startup_98': 'retail', 'startup_99': 'technology',
            'startup_100': 'finance'
        }
        
        for interaction in interactions:
            rating = interaction.get('rating', 0.0)
            if rating >= 3.0:  # Consider rating >= 3 as relevant
                startup_id = interaction.get('item_id', '')
                sector = startup_to_sector.get(startup_id, 'technology')  # Default to technology
                relevant_sectors.add(sector)
        
        return list(relevant_sectors)
    
    def _extract_relevant_items(self, user_data: Dict[str, Any]) -> List[str]:
        """Extract relevant items (startup IDs) from user interaction data"""
        
        interactions = user_data.get('interactions', [])
        relevant_items = []
        
        for interaction in interactions:
            rating = interaction.get('rating', 0.0)
            if rating >= 3.0:  # Consider rating >= 3 as relevant
                item_id = interaction.get('item_id', '')
                if item_id:
                    relevant_items.append(item_id)
        
        return relevant_items
    
    def _combine_dual_level_metrics(self, 
                                   sector_metrics: EvaluationMetrics, 
                                   item_metrics: EvaluationMetrics) -> EvaluationMetrics:
        """
        Combine sector-level and item-level metrics into a single EvaluationMetrics object
        
        Args:
            sector_metrics: Evaluation metrics at sector level
            item_metrics: Evaluation metrics at item level
            
        Returns:
            Combined metrics with prefixed keys
        """
        
        # Combine precision@k metrics
        combined_precision = {}
        for k, v in sector_metrics.precision_at_k.items():
            combined_precision[f"sector_precision@{k}"] = v
        for k, v in item_metrics.precision_at_k.items():
            combined_precision[f"item_precision@{k}"] = v
        
        # Combine recall@k metrics
        combined_recall = {}
        for k, v in sector_metrics.recall_at_k.items():
            combined_recall[f"sector_recall@{k}"] = v
        for k, v in item_metrics.recall_at_k.items():
            combined_recall[f"item_recall@{k}"] = v
        
        # Combine ndcg@k metrics
        combined_ndcg = {}
        for k, v in sector_metrics.ndcg_at_k.items():
            combined_ndcg[f"sector_ndcg@{k}"] = v
        for k, v in item_metrics.ndcg_at_k.items():
            combined_ndcg[f"item_ndcg@{k}"] = v
        
        # Use sector-level for aggregate metrics (as they're more business-relevant)
        return EvaluationMetrics(
            precision_at_k=combined_precision,
            recall_at_k=combined_recall,
            ndcg_at_k=combined_ndcg,
            map_score=sector_metrics.map_score,  # Use sector-level MAP as primary
            mrr_score=sector_metrics.mrr_score,  # Use sector-level MRR as primary
            diversity=sector_metrics.diversity,
            novelty=sector_metrics.novelty,
            coverage=sector_metrics.coverage,
            user_satisfaction=sector_metrics.user_satisfaction,
            engagement_metrics=sector_metrics.engagement_metrics
        )
    
    def _evaluate_recommendations(self, recommendations: Dict[str, List[str]], 
                                ground_truth: Dict[str, List[str]]) -> EvaluationMetrics:
        """Evaluate recommendations against ground truth"""
        
        user_metrics = self.evaluator.evaluate_system(
            recommendations, ground_truth, catalog_size=1000
        )
        
        return self.evaluator.aggregate_metrics(user_metrics)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Experimental Framework")
    print("=" * 50)
    
    # Initialize framework
    framework = ExperimentalFramework()
    
    # Create sample user data
    sample_user_data = {}
    for user_id in range(50):
        sample_user_data[f"user_{user_id}"] = {
            'role': np.random.choice(['entrepreneur', 'investor', 'business_owner']),
            'sectors': ['technology', 'finance'],
            'risk_appetite': np.random.choice(['low', 'moderate', 'high']),
            'interactions': [
                {
                    'item_id': f"item_{np.random.randint(0, 20)}",
                    'rating': np.random.choice([1, 2, 3, 4, 5]),
                    'timestamp': '2024-01-01'
                }
                for _ in range(np.random.randint(5, 15))
            ]
        }
    
    print(f"📊 Created sample dataset with {len(sample_user_data)} users")
    
    # Test dataset splitting
    splits = framework.create_dataset_splits(sample_user_data)
    print(f"✅ Created dataset split: {len(splits[0].train_users)} train, {len(splits[0].test_users)} test")
    
    # Test cross-validation splits
    cv_splits = framework.create_cross_validation_splits(sample_user_data, n_folds=3)
    print(f"✅ Created {len(cv_splits)} cross-validation folds")
    
    # Test hyperparameter grid generation
    param_grid = framework.generate_hyperparameter_grid(
        "GRPO-GRPO-P",
        {
            'learning_rate': [0.01, 0.05, 0.1],
            'exploration_rate': [0.1, 0.2, 0.3]
        }
    )
    print(f"✅ Generated hyperparameter grid with {len(param_grid)} combinations")
    
    print("\n🎯 Experimental Framework Ready!")
    print("Ready to run comprehensive research experiments!")
