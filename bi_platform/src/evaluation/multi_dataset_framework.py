
import sys
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from dataset_manager import StandardDatasetManager, DatasetInfo
from dataset_converter import DatasetConverter
from enhanced_cross_validation import EnhancedCrossValidator

# Mock imports for testing (move below all imports)
class ExperimentalFramework:
    def run_baseline_comparison(self, user_data, baselines):
        # Mock baseline comparison
        comparison_results = {}
        for baseline in baselines:
            comparison_results[baseline] = {
                'ndcg_5': 0.1 + hash(baseline) % 100 / 1000,
                'precision_5': 0.05 + hash(baseline) % 50 / 1000,
                'recall_5': 0.08 + hash(baseline) % 80 / 1000,
                'map_score': 0.06 + hash(baseline) % 60 / 1000
            }
        return {'comparison_results': comparison_results}

logger = logging.getLogger(__name__)

@dataclass
class MultiDatasetResult:
    """Result from multi-dataset experiment"""
    dataset_name: str
    dataset_info: DatasetInfo
    baseline_results: Dict[str, Any]
    cross_validation_results: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class ComparisonSummary:
    """Summary of results across datasets"""
    datasets: List[str]
    baseline_rankings: Dict[str, List[str]]  # metric -> ranked list of baselines
    best_baseline_per_dataset: Dict[str, str]
    consistency_scores: Dict[str, float]  # How consistent each baseline is across datasets
    domain_insights: Dict[str, Any]

class MultiDatasetFramework:
    """
    Enhanced experimental framework that supports multiple standard datasets
    """
    
    def __init__(self, results_dir: str = "multi_dataset_results"):
        self.dataset_manager = StandardDatasetManager()
        self.converter = DatasetConverter()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize experimental framework
        self.framework = ExperimentalFramework()
        
        # Store results across experiments
        self.experiment_results = {}
        
    def run_experiment_on_dataset(self, 
                                 dataset_name: str,
                                 max_users: Optional[int] = None,
                                 max_items: Optional[int] = None,
                                 cv_folds: int = 5,
                                 test_ratio: float = 0.2,
                                 **dataset_kwargs) -> MultiDatasetResult:
        """
        Run complete experiment on a single dataset
        """
        
        start_time = time.time()
        
        try:
            logger.info(f"Running experiment on {dataset_name}")
            
            # Load and convert dataset
            raw_dataset = self.dataset_manager.load_dataset(dataset_name, **dataset_kwargs)
            if not raw_dataset:
                raise ValueError(f"Failed to load dataset {dataset_name}")
            
            converted_dataset = self.converter.convert_to_experimental_format(raw_dataset)
            if not converted_dataset:
                raise ValueError(f"Failed to convert dataset {dataset_name}")
            
            # Subsample if needed for faster experimentation
            if max_users or max_items:
                converted_dataset = self.converter.subsample_dataset(
                    converted_dataset, 
                    max_users=max_users, 
                    max_items=max_items
                )
            
            # Create train/test split
            train_dataset, test_dataset = self.converter.create_train_test_split(
                converted_dataset, test_ratio=test_ratio
            )
            
            # Prepare data for experimental framework
            user_data = train_dataset['user_data']
            test_user_data = test_dataset['user_data']
            
            # Run baseline comparisons
            baseline_results = self.framework.run_baseline_comparison(
                user_data, 
                baselines=['random', 'popularity', 'user_avg', 'item_avg', 'grpo_original']
            )
            
            # Run enhanced cross-validation
            cv_validator = EnhancedCrossValidator(n_folds=cv_folds, random_seeds=[42, 123, 456])
            cv_results = cv_validator.run_cross_validation(
                user_data,
                baselines=['random', 'popularity', 'user_avg', 'grpo_original']
            )
            
            execution_time = time.time() - start_time
            
            result = MultiDatasetResult(
                dataset_name=dataset_name,
                dataset_info=converted_dataset['dataset_info'],
                baseline_results=baseline_results,
                cross_validation_results=cv_results,
                execution_time=execution_time,
                success=True
            )
            
            logger.info(f"✅ Completed experiment on {dataset_name} in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"❌ Experiment failed on {dataset_name}: {error_msg}")
            
            # Create dummy dataset info for failed experiments
            dummy_info = DatasetInfo(
                name=dataset_name,
                description="Failed to load",
                n_users=0,
                n_items=0,
                n_interactions=0,
                sparsity=0.0,
                rating_scale=(0, 0),
                domain="unknown",
                source="unknown"
            )
            
            return MultiDatasetResult(
                dataset_name=dataset_name,
                dataset_info=dummy_info,
                baseline_results={},
                cross_validation_results={},
                execution_time=execution_time,
                success=False,
                error_message=error_msg
            )
    
    def run_multi_dataset_experiment(self, 
                                   dataset_configs: List[Dict[str, Any]],
                                   cv_folds: int = 5) -> List[MultiDatasetResult]:
        """
        Run experiments across multiple datasets
        
        Args:
            dataset_configs: List of dataset configurations
                Each config should have: {'name': str, 'max_users': int, 'max_items': int, **kwargs}
        """
        
        results = []
        
        print(f"🚀 MULTI-DATASET EXPERIMENT STARTED")
        print(f"📊 Testing {len(dataset_configs)} datasets")
        print("=" * 60)
        
        for i, config in enumerate(dataset_configs):
            dataset_name = config.pop('name')
            
            print(f"\n📈 Experiment {i+1}/{len(dataset_configs)}: {dataset_name}")
            print("-" * 40)
            
            result = self.run_experiment_on_dataset(
                dataset_name=dataset_name,
                cv_folds=cv_folds,
                **config
            )
            
            results.append(result)
            self.experiment_results[dataset_name] = result
            
            # Print summary
            if result.success:
                info = result.dataset_info
                print(f"  ✅ Success: {info.n_users} users, {info.n_items} items")
                print(f"  ⏱️ Time: {result.execution_time:.2f}s")
                
                # Show top baseline if available
                if result.baseline_results and 'comparison_results' in result.baseline_results:
                    comparisons = result.baseline_results['comparison_results']
                    if comparisons:
                        best_baseline = max(comparisons.keys(), 
                                          key=lambda b: comparisons[b].get('ndcg_5', 0))
                        best_score = comparisons[best_baseline].get('ndcg_5', 0)
                        print(f"  🏆 Best baseline: {best_baseline} (NDCG@5: {best_score:.4f})")
            else:
                print(f"  ❌ Failed: {result.error_message}")
        
        print(f"\n✅ MULTI-DATASET EXPERIMENT COMPLETED")
        print(f"📊 {sum(1 for r in results if r.success)}/{len(results)} datasets successful")
        
        return results
    
    def analyze_cross_dataset_performance(self, results: List[MultiDatasetResult]) -> ComparisonSummary:
        """
        Analyze performance patterns across datasets
        """
        
        # Filter successful results
        successful_results = [r for r in results if r.success and r.baseline_results]
        
        if not successful_results:
            logger.warning("No successful results to analyze")
            return ComparisonSummary(
                datasets=[],
                baseline_rankings={},
                best_baseline_per_dataset={},
                consistency_scores={},
                domain_insights={}
            )
        
        # Extract baseline performance across datasets
        datasets = [r.dataset_name for r in successful_results]
        all_baselines = set()
        performance_matrix = {}  # dataset -> baseline -> metrics
        
        for result in successful_results:
            dataset_name = result.dataset_name
            performance_matrix[dataset_name] = {}
            
            if 'comparison_results' in result.baseline_results:
                for baseline, metrics in result.baseline_results['comparison_results'].items():
                    all_baselines.add(baseline)
                    performance_matrix[dataset_name][baseline] = metrics
        
        # Calculate rankings for each metric
        metrics_to_analyze = ['ndcg_5', 'precision_5', 'recall_5', 'map_score']
        baseline_rankings = {}
        
        for metric in metrics_to_analyze:
            # Average performance across datasets for each baseline
            baseline_avg_performance = {}
            
            for baseline in all_baselines:
                scores = []
                for dataset in datasets:
                    if (baseline in performance_matrix[dataset] and 
                        metric in performance_matrix[dataset][baseline]):
                        scores.append(performance_matrix[dataset][baseline][metric])
                
                if scores:
                    baseline_avg_performance[baseline] = sum(scores) / len(scores)
            
            # Rank baselines by average performance
            ranked_baselines = sorted(baseline_avg_performance.keys(),
                                   key=lambda b: baseline_avg_performance[b],
                                   reverse=True)
            baseline_rankings[metric] = ranked_baselines
        
        # Best baseline per dataset
        best_baseline_per_dataset = {}
        for result in successful_results:
            dataset_name = result.dataset_name
            if 'comparison_results' in result.baseline_results:
                comparisons = result.baseline_results['comparison_results']
                if comparisons:
                    best_baseline = max(comparisons.keys(),
                                      key=lambda b: comparisons[b].get('ndcg_5', 0))
                    best_baseline_per_dataset[dataset_name] = best_baseline
        
        # Calculate consistency scores (how consistent each baseline is across datasets)
        consistency_scores = {}
        for baseline in all_baselines:
            scores = []
            for dataset in datasets:
                if (baseline in performance_matrix[dataset] and 
                    'ndcg_5' in performance_matrix[dataset][baseline]):
                    scores.append(performance_matrix[dataset][baseline]['ndcg_5'])
            
            if len(scores) > 1:
                # Use coefficient of variation (std/mean) as consistency measure
                mean_score = sum(scores) / len(scores)
                if mean_score > 0:
                    std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
                    consistency_scores[baseline] = 1 - (std_score / mean_score)  # Higher = more consistent
                else:
                    consistency_scores[baseline] = 0
            else:
                consistency_scores[baseline] = 1 if scores else 0
        
        # Domain insights
        domain_insights = {
            'dataset_domains': {r.dataset_name: r.dataset_info.domain for r in successful_results},
            'sparsity_range': {
                'min': min(r.dataset_info.sparsity for r in successful_results),
                'max': max(r.dataset_info.sparsity for r in successful_results),
                'avg': sum(r.dataset_info.sparsity for r in successful_results) / len(successful_results)
            },
            'size_range': {
                'users': {
                    'min': min(r.dataset_info.n_users for r in successful_results),
                    'max': max(r.dataset_info.n_users for r in successful_results)
                },
                'items': {
                    'min': min(r.dataset_info.n_items for r in successful_results),
                    'max': max(r.dataset_info.n_items for r in successful_results)
                }
            }
        }
        
        return ComparisonSummary(
            datasets=datasets,
            baseline_rankings=baseline_rankings,
            best_baseline_per_dataset=best_baseline_per_dataset,
            consistency_scores=consistency_scores,
            domain_insights=domain_insights
        )
    
    def generate_multi_dataset_report(self, 
                                    results: List[MultiDatasetResult],
                                    output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive report of multi-dataset experiment
        """
        
        analysis = self.analyze_cross_dataset_performance(results)
        
        report_lines = [
            "📊 MULTI-DATASET EXPERIMENTAL REPORT",
            "=" * 60,
            "",
            f"🔬 Experiment Summary:",
            f"  • Total datasets tested: {len(results)}",
            f"  • Successful experiments: {sum(1 for r in results if r.success)}",
            f"  • Failed experiments: {sum(1 for r in results if not r.success)}",
            "",
            "📈 Dataset Overview:",
        ]
        
        # Dataset details
        for result in results:
            if result.success:
                info = result.dataset_info
                report_lines.extend([
                    f"  ✅ {result.dataset_name}:",
                    f"     • Domain: {info.domain}",
                    f"     • Users: {info.n_users:,}, Items: {info.n_items:,}",
                    f"     • Interactions: {info.n_interactions:,}",
                    f"     • Sparsity: {info.sparsity:.3f}",
                    f"     • Execution time: {result.execution_time:.2f}s",
                ])
            else:
                report_lines.extend([
                    f"  ❌ {result.dataset_name}: {result.error_message}",
                ])
        
        # Performance analysis
        if analysis.datasets:
            report_lines.extend([
                "",
                "🏆 Performance Analysis:",
                "",
                "📊 Baseline Rankings (by NDCG@5):"
            ])
            
            if 'ndcg_5' in analysis.baseline_rankings:
                for i, baseline in enumerate(analysis.baseline_rankings['ndcg_5'], 1):
                    consistency = analysis.consistency_scores.get(baseline, 0)
                    report_lines.append(f"  {i}. {baseline} (consistency: {consistency:.3f})")
            
            report_lines.extend([
                "",
                "🎯 Best Baseline per Dataset:"
            ])
            
            for dataset, baseline in analysis.best_baseline_per_dataset.items():
                report_lines.append(f"  • {dataset}: {baseline}")
            
            report_lines.extend([
                "",
                "🌍 Domain Insights:",
                f"  • Sparsity range: {analysis.domain_insights['sparsity_range']['min']:.3f} - {analysis.domain_insights['sparsity_range']['max']:.3f}",
                f"  • Average sparsity: {analysis.domain_insights['sparsity_range']['avg']:.3f}",
                f"  • User range: {analysis.domain_insights['size_range']['users']['min']:,} - {analysis.domain_insights['size_range']['users']['max']:,}",
                f"  • Item range: {analysis.domain_insights['size_range']['items']['min']:,} - {analysis.domain_insights['size_range']['items']['max']:,}",
            ])
        
        # Add detailed results for each dataset
        report_lines.extend([
            "",
            "📋 DETAILED RESULTS:",
            "=" * 60
        ])
        
        for result in results:
            if result.success and result.baseline_results:
                report_lines.extend([
                    f"",
                    f"🔍 {result.dataset_name} Results:",
                    f"   Domain: {result.dataset_info.domain}"
                ])
                
                if 'comparison_results' in result.baseline_results:
                    comparisons = result.baseline_results['comparison_results']
                    for baseline, metrics in comparisons.items():
                        ndcg5 = metrics.get('ndcg_5', 0)
                        prec5 = metrics.get('precision_5', 0)
                        recall5 = metrics.get('recall_5', 0)
                        report_lines.append(
                            f"   • {baseline:15} NDCG@5: {ndcg5:.4f}, P@5: {prec5:.4f}, R@5: {recall5:.4f}"
                        )
        
        report_text = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            output_path = self.results_dir / output_file
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text
    
    def save_results(self, results: List[MultiDatasetResult], filename: str = "multi_dataset_results.json"):
        """Save results to JSON file"""
        
        output_path = self.results_dir / filename
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = {
                'dataset_name': result.dataset_name,
                'dataset_info': asdict(result.dataset_info),
                'baseline_results': result.baseline_results,
                'cross_validation_results': result.cross_validation_results,
                'execution_time': result.execution_time,
                'success': result.success,
                'error_message': result.error_message
            }
            serializable_results.append(result_dict)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

def test_multi_dataset_framework():
    """Test the multi-dataset framework"""
    print("🌐 TESTING MULTI-DATASET FRAMEWORK")
    print("=" * 60)
    
    framework = MultiDatasetFramework()
    
    # Define test datasets
    dataset_configs = [
        {
            'name': 'synthetic-investment',
            'max_users': 50,
            'max_items': 30,
            'n_users': 50,
            'n_items': 30,
            'avg_interactions_per_user': 15
        },
        {
            'name': 'movielens-100k',
            'max_users': 100,
            'max_items': 100
        }
    ]
    
    # Run multi-dataset experiment
    results = framework.run_multi_dataset_experiment(dataset_configs, cv_folds=3)
    
    # Generate report
    report = framework.generate_multi_dataset_report(results, "test_multi_dataset_report.md")
    print("\n" + "="*60)
    print(report)
    
    # Save results
    framework.save_results(results, "test_multi_dataset_results.json")
    
    print("\n✅ Multi-Dataset Framework testing completed!")

if __name__ == "__main__":
    test_multi_dataset_framework()
