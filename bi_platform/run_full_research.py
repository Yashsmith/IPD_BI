"""
Run Complete Research Evaluation Pipeline
Execute comprehensive academic evaluation of GRPO-GRPO-P
"""

import sys
import os

# Add src to path
sys.path.append('src')

from evaluation.research_runner import ResearchExperimentRunner

def create_sample_dataset():
    """Create a sample dataset for testing"""
    import random
    
    sample_data = {}
    
    # Create 200 users with varied interaction patterns
    for user_id in range(200):
        # Vary user characteristics
        roles = ['entrepreneur', 'investor', 'business_owner', 'startup_founder']
        sectors = [['technology'], ['finance'], ['healthcare'], ['technology', 'finance'], ['retail']]
        risk_levels = ['low', 'moderate', 'high']
        
        # Create interaction history
        n_interactions = random.randint(5, 20)
        interactions = []
        
        for _ in range(n_interactions):
            interactions.append({
                'item_id': f"startup_{random.randint(1, 100)}",
                'rating': random.randint(1, 5),
                'timestamp': f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                'interaction_type': random.choice(['view', 'like', 'invest', 'contact'])
            })
        
        sample_data[f"user_{user_id}"] = {
            'role': random.choice(roles),
            'sectors': random.choice(sectors),
            'location': random.choice(['mumbai', 'delhi', 'bangalore', 'pune', 'hyderabad']),
            'capital_range': random.choice(['low', 'medium', 'high']),
            'risk_appetite': random.choice(risk_levels),
            'experience_level': random.choice(['beginner', 'intermediate', 'expert']),
            'interactions': interactions
        }
    
    return sample_data

def main():
    print("🚀 Starting GRPO-GRPO-P Research Evaluation")
    print("=" * 60)
    
    # Create sample dataset
    print("📊 Creating sample dataset...")
    user_data = create_sample_dataset()
    print(f"✅ Created dataset with {len(user_data)} users")
    
    # Initialize research runner
    runner = ResearchExperimentRunner("GRPO_GRPO_P_Comprehensive_Study")
    
    # Configuration for comprehensive evaluation
    research_config = {
        'cross_validation_folds': 5,
        'test_split_ratio': 0.2,
        'random_seed': 42,
        'hyperparameter_search': True,
        'ablation_study': True,
        'baseline_comparison': True,
        'statistical_analysis': True
    }
    
    print("🔬 Running comprehensive research pipeline...")
    print("This includes:")
    print("  ✓ Dataset preparation and analysis")
    print("  ✓ Baseline method evaluation (6 methods)")
    print("  ✓ GRPO-GRPO-P system evaluation")
    print("  ✓ Hyperparameter optimization")
    print("  ✓ Ablation studies")
    print("  ✓ Statistical significance testing")
    print("  ✓ Research report generation")
    print()
    print("⏰ This may take 15-30 minutes...")
    
    try:
        # Run complete research pipeline
        results = runner.run_complete_research_pipeline(user_data, research_config)
        
        print("\n🎉 Research Evaluation Complete!")
        print("=" * 60)
        
        # Display key results
        final_report = results.get('final_report', {})
        
        print("📊 Key Findings:")
        key_findings = final_report.get('key_findings', [])
        for finding in key_findings:
            print(f"  • {finding}")
        
        print("\n📈 Performance Summary:")
        performance_summary = final_report.get('performance_summary', {})
        for method, metrics in performance_summary.items():
            if 'avg_ndcg_at_5' in metrics:
                print(f"  • {method}: NDCG@5 = {metrics['avg_ndcg_at_5']:.3f} ± {metrics['std_ndcg_at_5']:.3f}")
        
        print("\n💡 Recommendations:")
        recommendations = final_report.get('recommendations', [])
        for rec in recommendations:
            print(f"  • {rec}")
        
        print(f"\n📁 Results saved to: research_results/")
        print("🎯 Ready for academic paper writing!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("This might be due to missing dependencies or data format issues.")
        print("Check the error details above and ensure all requirements are installed.")

if __name__ == "__main__":
    main()
