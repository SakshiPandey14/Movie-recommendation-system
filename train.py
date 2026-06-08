#!/usr/bin/env python3
"""
train.py — Offline model training script.

Usage
-----
# With real TMDB data:
python train.py --movies data/tmdb_5000_movies.csv \
                --credits data/tmdb_5000_credits.csv

# With built-in demo data:
python train.py --demo
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from preprocess import run_pipeline, engineer_features, build_display_df
from recommender import ContentRecommender
from sample_data import generate_sample_dataset


def train(movies_path: str | None, credits_path: str | None, demo: bool):
    if demo:
        print("► Using built-in demo dataset (30 movies)")
        raw = generate_sample_dataset()
        df = engineer_features(raw)
        df = build_display_df(df)
    else:
        if not movies_path or not credits_path:
            print("ERROR: Provide --movies and --credits paths, or use --demo")
            sys.exit(1)
        print(f"► Loading: {movies_path}")
        print(f"► Loading: {credits_path}")
        df = run_pipeline(movies_path, credits_path)

    print(f"► Dataset shape: {df.shape}")
    print("► Fitting ContentRecommender …")
    model = ContentRecommender(max_features=5000)
    model.fit(df)
    print(f"► {model}")

    model.save("model/recommender.pkl")
    print("► Training complete. Model saved to model/recommender.pkl")

    # Quick sanity-check
    sample_title = df["title"].iloc[0]
    print(f"\n► Quick test — recommendations for '{sample_title}':")
    recs = model.recommend(sample_title, n=5)
    for _, row in recs.iterrows():
        print(f"   #{int(row['rank'])}  {row['title']}  ({row['similarity_score']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CineMatch recommender.")
    parser.add_argument("--movies",  help="Path to tmdb_5000_movies.csv")
    parser.add_argument("--credits", help="Path to tmdb_5000_credits.csv")
    parser.add_argument("--demo",    action="store_true", help="Use demo dataset")
    args = parser.parse_args()
    train(args.movies, args.credits, args.demo)
