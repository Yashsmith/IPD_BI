#!/usr/bin/env python3
"""
Interactive CLI for GRPO–GRPO-P Hybrid Recommendation System
Usage examples:
  python interactive_recommend.py --user-id alice --role entrepreneur \
    --sectors technology finance --location mumbai --capital-range medium \
    --risk high --experience expert --num-recs 3

Optionally provide feedback after seeing recommendations with --feedback yes
"""
import sys
import os
import argparse
from typing import List

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.hybrid_system import HybridRecommendationSystem, SystemUser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive GRPO–GRPO-P recommendations")
    parser.add_argument("--user-id", required=True, help="Unique user id")
    parser.add_argument("--role", required=True, choices=[
        "entrepreneur", "investor", "business_owner", "analyst"
    ])
    parser.add_argument("--sectors", nargs="+", required=True, help="Preferred sectors, e.g., technology finance")
    parser.add_argument("--location", required=True)
    parser.add_argument("--capital-range", required=True, choices=["low", "medium", "high"])
    parser.add_argument("--risk", required=True, choices=["low", "moderate", "high"], help="Risk appetite")
    parser.add_argument("--experience", required=True, choices=["beginner", "intermediate", "expert"])
    parser.add_argument("--num-recs", type=int, default=3, help="Number of recommendations to return")
    parser.add_argument("--feedback", choices=["yes", "no"], default="no", help="Prompt for feedback after recommendations")
    return parser.parse_args()


def main():
    args = parse_args()

    system = HybridRecommendationSystem()

    user = SystemUser(
        user_id=args.user_id,
        role=args.role,
        sectors=args.sectors,
        location=args.location,
        capital_range=args.capital_range,
        risk_appetite=args.risk,
        experience_level=args.experience,
    )

    system.register_user(user)

    recommendations = system.get_recommendations(args.user_id, max_recommendations=args.num_recs)

    print("\n=== Recommendations ===")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. Sector: {rec.sector} | Action: {rec.recommendation_type} | Confidence: {rec.confidence:.2f}")
        print(f"   Blend: {rec.grpo_weight:.0%} GRPO + {rec.grpo_p_weight:.0%} GRPO-P | Strategy: {rec.blending_strategy}")
        print(f"   Risk: {rec.risk_level}\n")

    if args.feedback == "yes" and len(recommendations) > 0:
        try:
            choice = input("Provide feedback on the first recommendation (p=positive, n=negative, s=skip): ").strip().lower()
            if choice in ("p", "n"):
                score = 0.8 if choice == "p" else -0.8
                system.provide_feedback(
                    user_id=args.user_id,
                    recommendation_id="rec_1",
                    feedback_type="rated_positive" if choice == "p" else "rated_negative",
                    feedback_score=score,
                )
                print("Feedback recorded.")
        except EOFError:
            pass


if __name__ == "__main__":
    main() 