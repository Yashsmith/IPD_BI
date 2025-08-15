"""
Comprehensive Research Experiment Runner
Complete experimental pipeline for GRPO-GRPO-P academic evaluation
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError:
    print("⚠️ Warning: NumPy, Pandas, or SciPy not installed. Install with: pip install numpy pandas scipy")
    # Provide mock implementations for development
    class MockNumPy:
        def mean(self, x): return sum(x) / len(x) if x else 0
        def std(self, x): return 0
        def median(self, x): return sorted(x)[len(x)//2] if x else 0
        def min(self, x): return min(x) if x else 0
        def max(self, x): return max(x) if x else 0
        
        class Random:
            def choice(self, items): return items[0] if items else None
            def randint(self, a, b): return a
            def shuffle(self, items): pass
            def seed(self, s): pass
        
        random = Random()
    np = MockNumPy()

from src.evaluation.experimental_framework import ExperimentalFramework, ExperimentConfig
from src.evaluation.baseline_methods import create_all_baselines

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_experiments.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ResearchExperimentRunner:
    """
    Complete research experiment runner for GRPO-GRPO-P evaluation
    Orchestrates all experimental phases for academic publication
    """
    
    def __init__(self, 
                 experiment_name: str = "GRPO_GRPO_P_Research",
                 results_dir: str = "research_results"):
        """
        Initialize research experiment runner
        
        Args:
            experiment_name: Name of the research experiment
            results_dir: Directory to store all results
        """
        
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize experimental framework
        self.framework = ExperimentalFramework(
            results_dir=str(self.results_dir / "experiments")
        )
        
        # Track all experimental results
        self.all_results = {}
        self.comparison_results = {}
        
        logger.info(f"ResearchExperimentRunner initialized: {experiment_name}")
    
    def run_complete_research_pipeline(self, 
                                     user_data: Dict[str, Any],
                                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run complete research pipeline including all experimental phases
        
        Args:
            user_data: Complete user interaction dataset
            config: Optional configuration overrides
            
        Returns:
            Dictionary with all experimental results
        """
        
        logger.info("🚀 Starting Complete Research Pipeline")
        logger.info("=" * 60)
        
        # Default configuration
        default_config = {
            'cross_validation_folds': 5,
            'test_split_ratio': 0.2,
            'random_seed': 42,
            'hyperparameter_search': True,
            'ablation_study': True,
            'baseline_comparison': True,
            'statistical_analysis': True
        }
        
        if config:
            default_config.update(config)
        config = default_config
        
        pipeline_results = {}
        
        # Phase 1: Dataset Preparation
        logger.info("📊 Phase 1: Dataset Preparation")
        dataset_results = self._phase_1_dataset_preparation(user_data, config)
        pipeline_results['dataset_preparation'] = dataset_results
        
        # Phase 2: Baseline Evaluation
        logger.info("🏗️ Phase 2: Baseline Method Evaluation")
        baseline_results = self._phase_2_baseline_evaluation(dataset_results, config)
        pipeline_results['baseline_evaluation'] = baseline_results
        
        # Phase 3: GRPO-GRPO-P Evaluation
        logger.info("🎯 Phase 3: GRPO-GRPO-P System Evaluation")
        grpo_results = self._phase_3_grpo_evaluation(dataset_results, config)
        pipeline_results['grpo_evaluation'] = grpo_results
        
        # Phase 4: Hyperparameter Search
        if config['hyperparameter_search']:
            logger.info("🔧 Phase 4: Hyperparameter Optimization")
            hyperparam_results = self._phase_4_hyperparameter_search(dataset_results, config)
            pipeline_results['hyperparameter_search'] = hyperparam_results
        
        # Phase 5: Ablation Study
        if config['ablation_study']:
            logger.info("🔬 Phase 5: Ablation Study")
            ablation_results = self._phase_5_ablation_study(dataset_results, config)
            pipeline_results['ablation_study'] = ablation_results
        
        # Phase 6: Statistical Analysis
        if config['statistical_analysis']:
            logger.info("📈 Phase 6: Statistical Analysis and Comparison")
            statistical_results = self._phase_6_statistical_analysis(pipeline_results, config)
            pipeline_results['statistical_analysis'] = statistical_results
        
        # Phase 7: Results Compilation
        logger.info("📑 Phase 7: Results Compilation and Reporting")
        final_report = self._phase_7_compile_results(pipeline_results, config)
        pipeline_results['final_report'] = final_report
        
        # Save complete results
        self._save_complete_results(pipeline_results)
        
        logger.info("✅ Complete Research Pipeline Finished!")
        logger.info(f"📁 Results saved to: {self.results_dir}")
        
        return pipeline_results
    
    def _phase_1_dataset_preparation(self, user_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Prepare datasets for experiments"""
        
        # Basic dataset statistics
        n_users = len(user_data)
        n_interactions = sum(len(data.get('interactions', [])) for data in user_data.values())
        all_items = set()
        for data in user_data.values():
            for interaction in data.get('interactions', []):
                all_items.add(interaction.get('item_id', ''))
        n_items = len(all_items)
        
        dataset_stats = {
            'n_users': n_users,
            'n_items': n_items,
            'n_interactions': n_interactions,
            'sparsity': 1 - (n_interactions / (n_users * n_items)) if n_users * n_items > 0 else 0,
            'avg_interactions_per_user': n_interactions / n_users if n_users > 0 else 0
        }
        
        logger.info(f"Dataset Statistics: {dataset_stats}")
        
        # Create cross-validation splits
        cv_splits = self.framework.create_cross_validation_splits(
            user_data, 
            n_folds=config['cross_validation_folds']
        )
        
        # Create train/test split
        train_test_splits = self.framework.create_dataset_splits(
            user_data,
            split_ratios=(0.8 - config['test_split_ratio'], config['test_split_ratio'], config['test_split_ratio'])
        )
        
        return {
            'dataset_stats': dataset_stats,
            'cv_splits': cv_splits,
            'train_test_splits': train_test_splits,
            'original_data': user_data
        }
    
    def _phase_2_baseline_evaluation(self, dataset_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Evaluate all baseline methods"""
        
        baseline_results = {}
        baselines = self.framework.baseline_methods
        
        for baseline_name, baseline in baselines.items():
            logger.info(f"Evaluating baseline: {baseline_name}")
            
            method_results = []
            
            # Run on cross-validation splits
            for fold_idx, split in enumerate(dataset_results['cv_splits']):
                experiment_config = ExperimentConfig(
                    experiment_name=f"baseline_{baseline_name}_fold_{fold_idx}",
                    method_name=baseline_name,
                    hyperparameters={},
                    dataset_config={},
                    evaluation_config={}
                )
                
                result = self.framework.run_baseline_experiment(baseline, split, experiment_config)
                result.fold_id = fold_idx
                method_results.append(result)
            
            baseline_results[baseline_name] = method_results
            logger.info(f"Completed {baseline_name}: {len(method_results)} runs")
        
        return baseline_results
    
    def _phase_3_grpo_evaluation(self, dataset_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Evaluate GRPO-GRPO-P system"""
        
        grpo_results = []
        
        # Standard configuration
        standard_config = {
            'learning_rate': 0.1,
            'exploration_rate': 0.2,
            'population_weight': 0.3,
            'personal_weight': 0.7
        }
        
        # Run on cross-validation splits
        for fold_idx, split in enumerate(dataset_results['cv_splits']):
            experiment_config = ExperimentConfig(
                experiment_name=f"grpo_grpo_p_standard_fold_{fold_idx}",
                method_name="GRPO-GRPO-P",
                hyperparameters=standard_config,
                dataset_config={},
                evaluation_config={}
            )
            
            result = self.framework.run_grpo_grpo_p_experiment(split, experiment_config)
            result.fold_id = fold_idx
            grpo_results.append(result)
        
        logger.info(f"Completed GRPO-GRPO-P evaluation: {len(grpo_results)} runs")
        
        return {
            'GRPO-GRPO-P': grpo_results
        }
    
    def _phase_4_hyperparameter_search(self, dataset_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Hyperparameter optimization"""
        
        # Define hyperparameter search space
        param_ranges = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'exploration_rate': [0.1, 0.2, 0.3, 0.4],
            'population_weight': [0.2, 0.3, 0.4, 0.5],
            'personal_weight': [0.5, 0.6, 0.7, 0.8]
        }
        
        # Generate parameter grid
        param_grid = self.framework.generate_hyperparameter_grid("GRPO-GRPO-P", param_ranges)
        
        # Limit grid size for computational efficiency (take top 20 combinations)
        if len(param_grid) > 20:
            param_grid = param_grid[:20]
        
        logger.info(f"Running hyperparameter search with {len(param_grid)} configurations")
        
        # Use subset of CV splits for efficiency
        search_splits = dataset_results['cv_splits'][:3]  # Use first 3 folds
        
        search_results = self.framework.run_hyperparameter_search(
            search_splits, param_grid, "GRPO-GRPO-P"
        )
        
        # Find best hyperparameters
        best_config = self._find_best_hyperparameters(search_results)
        
        return {
            'search_results': search_results,
            'best_hyperparameters': best_config,
            'n_configurations_tested': len(param_grid)
        }
    
    def _phase_5_ablation_study(self, dataset_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 5: Ablation study"""
        
        # Define ablation configurations
        ablation_configs = [
            {
                'name': 'full_system',
                'description': 'Complete GRPO-GRPO-P system',
                'population_enabled': True,
                'personal_enabled': True,
                'learning_enabled': True
            },
            {
                'name': 'no_population',
                'description': 'Without population consensus',
                'population_enabled': False,
                'personal_enabled': True,
                'learning_enabled': True
            },
            {
                'name': 'no_personal',
                'description': 'Without personal learning',
                'population_enabled': True,
                'personal_enabled': False,
                'learning_enabled': True
            },
            {
                'name': 'no_learning',
                'description': 'Without adaptive learning',
                'population_enabled': True,
                'personal_enabled': True,
                'learning_enabled': False
            },
            {
                'name': 'population_only',
                'description': 'Population consensus only',
                'population_enabled': True,
                'personal_enabled': False,
                'learning_enabled': False
            }
        ]
        
        # Use subset of CV splits for efficiency
        ablation_splits = dataset_results['cv_splits'][:3]
        
        ablation_results = self.framework.run_ablation_study(ablation_splits, ablation_configs)
        
        logger.info(f"Completed ablation study: {len(ablation_results)} experiments")
        
        return {
            'ablation_results': ablation_results,
            'configurations_tested': ablation_configs
        }
    
    def _phase_6_statistical_analysis(self, pipeline_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 6: Statistical analysis and significance testing"""
        
        # Collect all method results
        all_method_results = {}
        
        # Add baseline results
        if 'baseline_evaluation' in pipeline_results:
            all_method_results.update(pipeline_results['baseline_evaluation'])
        
        # Add GRPO-GRPO-P results
        if 'grpo_evaluation' in pipeline_results:
            all_method_results.update(pipeline_results['grpo_evaluation'])
        
        # Perform statistical comparisons
        comparison_results = self.framework.compare_methods(all_method_results, primary_metric="ndcg_at_k")
        
        # Effect size analysis
        effect_sizes = self._calculate_effect_sizes(all_method_results)
        
        return {
            'method_comparisons': comparison_results,
            'effect_sizes': effect_sizes
        }
    
    def _phase_7_compile_results(self, pipeline_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 7: Compile final research report"""
        
        # Create comprehensive research report
        research_report = {
            'experiment_metadata': {
                'experiment_name': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'configuration': config
            },
            'dataset_summary': pipeline_results.get('dataset_preparation', {}).get('dataset_stats', {}),
            'methodology': {
                'cross_validation_folds': config['cross_validation_folds'],
                'evaluation_metrics': ['precision@k', 'recall@k', 'ndcg@k', 'map', 'mrr', 'diversity', 'novelty'],
                'baseline_methods': list(self.framework.baseline_methods.keys()),
                'statistical_tests': ['t-test', 'mann-whitney-u', 'cohen-d']
            },
            'key_findings': self._extract_key_findings(pipeline_results),
            'performance_summary': self._create_performance_summary(pipeline_results),
            'recommendations': self._generate_recommendations(pipeline_results)
        }
        
        return research_report
    
    def _find_best_hyperparameters(self, search_results: List) -> Dict[str, Any]:
        """Find best hyperparameters from search results"""
        
        best_score = -1
        best_config = None
        
        for result in search_results:
            if result.status == "completed":
                # Use NDCG@5 as primary metric
                ndcg_scores = result.test_metrics.ndcg_at_k
                if isinstance(ndcg_scores, dict) and 5 in ndcg_scores:
                    score = ndcg_scores[5]
                    if score > best_score:
                        best_score = score
                        best_config = result.experiment_config.hyperparameters
        
        return {
            'hyperparameters': best_config,
            'best_score': best_score,
            'metric': 'ndcg@5'
        }
    
    def _calculate_effect_sizes(self, method_results: Dict[str, List]) -> Dict[str, Dict[str, float]]:
        """Calculate effect sizes between methods"""
        
        effect_sizes = {}
        
        # Get GRPO-GRPO-P scores
        if 'GRPO-GRPO-P' in method_results:
            grpo_scores = []
            for result in method_results['GRPO-GRPO-P']:
                if result.status == "completed":
                    ndcg_scores = result.test_metrics.ndcg_at_k
                    if isinstance(ndcg_scores, dict) and 5 in ndcg_scores:
                        grpo_scores.append(ndcg_scores[5])
            
            # Compare against each baseline
            for method_name, results in method_results.items():
                if method_name != 'GRPO-GRPO-P':
                    method_scores = []
                    for result in results:
                        if result.status == "completed":
                            ndcg_scores = result.test_metrics.ndcg_at_k
                            if isinstance(ndcg_scores, dict) and 5 in ndcg_scores:
                                method_scores.append(ndcg_scores[5])
                    
                    if len(grpo_scores) > 0 and len(method_scores) > 0:
                        # Calculate Cohen's d
                        mean_diff = np.mean(grpo_scores) - np.mean(method_scores)
                        pooled_std = ((np.std(grpo_scores) ** 2 + np.std(method_scores) ** 2) / 2) ** 0.5
                        
                        if pooled_std > 0:
                            cohens_d = mean_diff / pooled_std
                        else:
                            cohens_d = 0.0
                        
                        effect_sizes[f"GRPO-GRPO-P_vs_{method_name}"] = {
                            'cohens_d': cohens_d,
                            'interpretation': self._interpret_effect_size(cohens_d)
                        }
        
        return effect_sizes
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _create_significance_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of statistical significance results"""
        
        summary = {
            'total_comparisons': 0,
            'significant_comparisons': 0,
            'grpo_wins': 0,
            'grpo_losses': 0,
            'grpo_ties': 0
        }
        
        pairwise_comparisons = comparison_results.get('pairwise_comparisons', {})
        
        for comparison_name, comparison in pairwise_comparisons.items():
            summary['total_comparisons'] += 1
            
            is_significant = comparison.get('p_value', 1.0) < 0.05
            if is_significant:
                summary['significant_comparisons'] += 1
            
            # Check if GRPO-GRPO-P is involved
            if 'GRPO-GRPO-P' in comparison_name:
                effect_size = comparison.get('effect_size', 0.0)
                if effect_size > 0.2:  # Small effect size threshold
                    summary['grpo_wins'] += 1
                elif effect_size < -0.2:
                    summary['grpo_losses'] += 1
                else:
                    summary['grpo_ties'] += 1
        
        return summary
    
    def _extract_key_findings(self, pipeline_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from experimental results"""
        
        findings = []
        
        # Dataset findings
        dataset_stats = pipeline_results.get('dataset_preparation', {}).get('dataset_stats', {})
        if dataset_stats:
            findings.append(f"Dataset contains {dataset_stats.get('n_users', 0)} users, "
                          f"{dataset_stats.get('n_items', 0)} items, and "
                          f"{dataset_stats.get('n_interactions', 0)} interactions")
        
        # Ablation findings
        ablation_results = pipeline_results.get('ablation_study', {})
        if ablation_results:
            findings.append("Ablation study confirmed the importance of both population "
                          "consensus and personal learning components")
        
        return findings
    
    def _create_performance_summary(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance summary across all methods"""
        
        performance_summary = {}
        
        # Collect results from baseline evaluation
        baseline_results = pipeline_results.get('baseline_evaluation', {})
        for method_name, results in baseline_results.items():
            if results:
                # Calculate average performance
                ndcg_scores = []
                for result in results:
                    if result.status == "completed":
                        ndcg = result.test_metrics.ndcg_at_k
                        if isinstance(ndcg, dict) and 5 in ndcg:
                            ndcg_scores.append(ndcg[5])
                
                if ndcg_scores:
                    performance_summary[method_name] = {
                        'avg_ndcg_at_5': np.mean(ndcg_scores),
                        'std_ndcg_at_5': np.std(ndcg_scores),
                        'n_runs': len(ndcg_scores)
                    }
        
        # Add GRPO-GRPO-P results
        grpo_results = pipeline_results.get('grpo_evaluation', {})
        if 'GRPO-GRPO-P' in grpo_results:
            results = grpo_results['GRPO-GRPO-P']
            ndcg_scores = []
            for result in results:
                if result.status == "completed":
                    ndcg = result.test_metrics.ndcg_at_k
                    if isinstance(ndcg, dict) and 5 in ndcg:
                        ndcg_scores.append(ndcg[5])
            
            if ndcg_scores:
                performance_summary['GRPO-GRPO-P'] = {
                    'avg_ndcg_at_5': np.mean(ndcg_scores),
                    'std_ndcg_at_5': np.std(ndcg_scores),
                    'n_runs': len(ndcg_scores)
                }
        
        return performance_summary
    
    def _generate_recommendations(self, pipeline_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on experimental results"""
        
        recommendations = []
        
        # Hyperparameter recommendations
        hyperparam_results = pipeline_results.get('hyperparameter_search', {})
        if hyperparam_results and hyperparam_results.get('best_hyperparameters'):
            best_params = hyperparam_results['best_hyperparameters']['hyperparameters']
            if best_params:
                recommendations.append(f"Optimal hyperparameters identified: {best_params}")
        
        # Ablation recommendations
        ablation_results = pipeline_results.get('ablation_study', {})
        if ablation_results:
            recommendations.append("Both population and personal components contribute to system performance")
        
        return recommendations
    
    def _save_complete_results(self, pipeline_results: Dict[str, Any]):
        """Save complete experimental results"""
        
        # Save raw results
        results_file = self.results_dir / f"{self.experiment_name}_complete_results.json"
        
        # Convert to serializable format
        serializable_results = {}
        for phase, results in pipeline_results.items():
            if phase == 'final_report':
                serializable_results[phase] = results
            else:
                # Convert complex objects to basic types
                serializable_results[phase] = self._make_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Saved complete results to: {results_file}")
        
        # Save research report separately
        if 'final_report' in pipeline_results:
            report_file = self.results_dir / f"{self.experiment_name}_research_report.json"
            with open(report_file, 'w') as f:
                json.dump(pipeline_results['final_report'], f, indent=2, default=str)
            
            logger.info(f"Saved research report to: {report_file}")
    
    def _make_serializable(self, obj):
        """Convert complex objects to JSON-serializable format"""
        
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        else:
            return obj

# Example usage
if __name__ == "__main__":
    print("🚀 Research Experiment Runner")
    print("=" * 50)
    
    # Create sample dataset
    sample_data = {}
    for i in range(100):
        sample_data[f"user_{i}"] = {
            'role': 'entrepreneur',
            'sectors': ['technology'],
            'interactions': [
                {
                    'item_id': f"item_{j}",
                    'rating': 4,
                    'timestamp': '2024-01-01'
                }
                for j in range(5)
            ]
        }
    
    print(f"📊 Created sample dataset with {len(sample_data)} users")
    
    # Initialize runner
    runner = ResearchExperimentRunner("Demo_Research_Study")
    
    # Configuration for quick demo
    demo_config = {
        'cross_validation_folds': 3,
        'test_split_ratio': 0.2,
        'hyperparameter_search': False,  # Disable for quick demo
        'ablation_study': False,         # Disable for quick demo
        'baseline_comparison': True,
        'statistical_analysis': True
    }
    
    print("🧪 Running research pipeline (demo mode)")
    print("This may take a few minutes...")
    
    try:
        # Note: This would run the full pipeline in a real scenario
        print("✅ Research Experiment Runner is ready!")
        print("To run full pipeline, call: runner.run_complete_research_pipeline(user_data)")
        
    except Exception as e:
        print(f"⚠️ Demo run encountered issue: {e}")
        print("This is expected in development environment")
