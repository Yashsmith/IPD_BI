📊 MULTI-DATASET EXPERIMENTAL REPORT
============================================================

🔬 Experiment Summary:
  • Total datasets tested: 2
  • Successful experiments: 2
  • Failed experiments: 0

📈 Dataset Overview:
  ✅ synthetic-investment:
     • Domain: investment
     • Users: 50, Items: 30
     • Interactions: 513
     • Sparsity: 0.658
     • Execution time: 0.00s
  ✅ movielens-100k:
     • Domain: movies
     • Users: 54, Items: 100
     • Interactions: 576
     • Sparsity: 0.893
     • Execution time: 0.34s

🏆 Performance Analysis:

📊 Baseline Rankings (by NDCG@5):
  1. random (consistency: 1.000)
  2. grpo_original (consistency: 1.000)
  3. user_avg (consistency: 1.000)
  4. item_avg (consistency: 1.000)
  5. popularity (consistency: 1.000)

🎯 Best Baseline per Dataset:
  • synthetic-investment: random
  • movielens-100k: random

🌍 Domain Insights:
  • Sparsity range: 0.658 - 0.893
  • Average sparsity: 0.776
  • User range: 50 - 54
  • Item range: 30 - 100

📋 DETAILED RESULTS:
============================================================

🔍 synthetic-investment Results:
   Domain: investment
   • random          NDCG@5: 0.1850, P@5: 0.0850, R@5: 0.1250
   • popularity      NDCG@5: 0.1000, P@5: 0.0500, R@5: 0.1200
   • user_avg        NDCG@5: 0.1410, P@5: 0.0910, R@5: 0.1010
   • item_avg        NDCG@5: 0.1340, P@5: 0.0840, R@5: 0.1340
   • grpo_original   NDCG@5: 0.1620, P@5: 0.0620, R@5: 0.0820

🔍 movielens-100k Results:
   Domain: movies
   • random          NDCG@5: 0.1850, P@5: 0.0850, R@5: 0.1250
   • popularity      NDCG@5: 0.1000, P@5: 0.0500, R@5: 0.1200
   • user_avg        NDCG@5: 0.1410, P@5: 0.0910, R@5: 0.1010
   • item_avg        NDCG@5: 0.1340, P@5: 0.0840, R@5: 0.1340
   • grpo_original   NDCG@5: 0.1620, P@5: 0.0620, R@5: 0.0820