📊 MULTI-DATASET EXPERIMENTAL REPORT
============================================================

🔬 Experiment Summary:
  • Total datasets tested: 2
  • Successful experiments: 1
  • Failed experiments: 1

📈 Dataset Overview:
  ✅ synthetic-investment:
     • Domain: investment
     • Users: 50, Items: 30
     • Interactions: 579
     • Sparsity: 0.614
     • Execution time: 0.00s
  ❌ synthetic-investment-small: Failed to load dataset synthetic-investment-small

🏆 Performance Analysis:

📊 Baseline Rankings (by NDCG@5):
  1. random (consistency: 1.000)
  2. grpo_original (consistency: 1.000)
  3. user_avg (consistency: 1.000)
  4. item_avg (consistency: 1.000)
  5. popularity (consistency: 1.000)

🎯 Best Baseline per Dataset:
  • synthetic-investment: random

🌍 Domain Insights:
  • Sparsity range: 0.614 - 0.614
  • Average sparsity: 0.614
  • User range: 50 - 50
  • Item range: 30 - 30

📋 DETAILED RESULTS:
============================================================

🔍 synthetic-investment Results:
   Domain: investment
   • random          NDCG@5: 0.1900, P@5: 0.0900, R@5: 0.1500
   • popularity      NDCG@5: 0.1150, P@5: 0.0650, R@5: 0.1550
   • user_avg        NDCG@5: 0.1570, P@5: 0.0570, R@5: 0.1170
   • item_avg        NDCG@5: 0.1160, P@5: 0.0660, R@5: 0.1560
   • grpo_original   NDCG@5: 0.1620, P@5: 0.0620, R@5: 0.1220