"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script implements the recommendation system for the MovieLens 1M dataset. It includes
functions for loading data, preprocessing temporal information, training an SVD-based
recommendation model, and evaluating its performance using metrics such as RMSE, Precision@K,
and Recall@K. The best-performing model is saved for later use.
"""

import logging
import os
import pickle
import sqlite3
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from helpers import perform_cross_validation, precision_recall_at_k
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import GridSearchCV

LOG_FILE = "recommendation_system.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log file
        logging.StreamHandler(),  # Console output
    ],
)


def load_data_from_sqlite(db_path: str) -> pd.DataFrame:
    """
    Loads the ratings data from the SQLite database.

    :param db_path: Path to the SQLite database file.
    :return: A DataFrame containing the ratings data.
    """
    logging.info("Loading ratings data from SQLite database at %s", db_path)

    conn = sqlite3.connect(db_path)
    try:
        # Load ratings table with appropriate data types
        ratings_df = pd.read_sql(
            "SELECT UserID, MovieID, Rating, Timestamp FROM ratings", conn
        )
        ratings_df = ratings_df.astype(
            {
                "UserID": "int32",
                "MovieID": "int32",
                "Rating": "float32",
                "Timestamp": "datetime64[ns]",
            }
        )
        logging.debug(
            "Loaded ratings table with %d rows of ratings data, including Timestamp.",
            len(ratings_df),
        )

    except Exception as e:
        logging.error("Failed to load ratings data: %s", e, exc_info=True)
        raise
    finally:
        conn.close()
        logging.info("Database connection closed.")
    return ratings_df


def preprocess_with_timestamp(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the ratings DataFrame to include temporal information.

    :param ratings_df: DataFrame containing the ratings data with a Timestamp column.
    :return: Preprocessed DataFrame with an additional 'Recency' feature.
    """
    logging.info("Preprocessing ratings data to include temporal information.")

    # Convert Timestamp to datetime if not already done
    if not pd.api.types.is_datetime64_any_dtype(ratings_df["Timestamp"]):
        ratings_df["Timestamp"] = pd.to_datetime(ratings_df["Timestamp"], unit="s")

    # Calculate the recency of each rating (days since the earliest rating)
    earliest_timestamp = ratings_df["Timestamp"].min()
    ratings_df["Recency"] = (ratings_df["Timestamp"] - earliest_timestamp).dt.days

    logging.debug(
        "Added 'Recency' feature to ratings DataFrame. Earliest timestamp: %s",
        earliest_timestamp,
    )
    return ratings_df


def save_model(algo, model_path: str) -> None:
    """
    Saves the trained model to a file.

    :param algo: The trained Surprise model.
    :param model_path: Path to save the model file.
    """
    logging.info("Saving the trained model to %s", model_path)
    try:
        with open(model_path, "wb") as model_file:
            pickle.dump(algo, model_file)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error("Failed to save the model: %s", e, exc_info=True)
        raise


def soft_weight_by_recency(ratings_df, decay_rate=0.01):
    """
    Assigns a soft weight to each rating using exponential decay based on recency.
    Returns a DataFrame sampled according to these weights.
    """
    max_recency = ratings_df["Recency"].max()
    weights = np.exp(decay_rate * (ratings_df["Recency"] - max_recency))
    weights = weights / weights.sum()
    sampled_idx = np.random.choice(
        ratings_df.index, size=len(ratings_df), replace=True, p=weights
    )
    return ratings_df.loc[sampled_idx].reset_index(drop=True)


def filter_noise_outliers(ratings_df, min_user_ratings=10, min_item_ratings=10):
    """
    Removes users and items with very few ratings.
    """
    user_counts = ratings_df["UserID"].value_counts()
    item_counts = ratings_df["MovieID"].value_counts()
    filtered = ratings_df[
        ratings_df["UserID"].isin(user_counts[user_counts >= min_user_ratings].index)
        & ratings_df["MovieID"].isin(item_counts[item_counts >= min_item_ratings].index)
    ]
    return filtered.reset_index(drop=True)


def time_based_split(ratings_df, test_ratio=0.2):
    """
    Chronologically splits ratings_df into train and test sets.
    """
    ratings_df = ratings_df.sort_values("Timestamp")
    split_idx = int(len(ratings_df) * (1 - test_ratio))
    train = ratings_df.iloc[:split_idx]
    test = ratings_df.iloc[split_idx:]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def get_coverage(recommendations, all_items):
    """
    Percentage of unique items recommended at least once.
    """
    recommended_items = set([iid for recs in recommendations.values() for iid in recs])
    return len(recommended_items) / len(all_items)


def get_diversity(recommendations, item_similarity):
    """
    Diversity: average pairwise dissimilarity among recommended items (1 - similarity).
    item_similarity: dict of (item1, item2) -> similarity (0-1)
    """
    diversities = []
    for recs in recommendations.values():
        if len(recs) < 2:
            continue
        pairs = [(recs[i], recs[j]) for i in range(len(recs)) for j in range(i + 1, len(recs))]
        dissimilarities = [1 - item_similarity.get((a, b), 0) for a, b in pairs]
        if dissimilarities:
            diversities.append(np.mean(dissimilarities))
    return np.mean(diversities) if diversities else 0


def get_novelty(recommendations, item_popularity):
    """
    Novelty: average popularity rank of recommended items (lower is less novel).
    item_popularity: dict of item -> count
    """
    all_counts = np.array(list(item_popularity.values()))
    ranks = {iid: (all_counts > item_popularity[iid]).sum() + 1 for iid in item_popularity}
    novelty_scores = []
    for recs in recommendations.values():
        novelty_scores += [ranks.get(iid, 0) for iid in recs]
    return np.mean(novelty_scores) if novelty_scores else 0


def recommend_top_k(algo, users, items, k=10):
    """
    For each user, recommend top-k items not yet rated.
    Returns: dict user -> list of item ids
    """
    recommendations = {}
    for uid in users:
        user_items = set(items)
        # Remove items already rated by user
        # (Assume you have a ratings_df in scope)
        rated = set(ratings_df[ratings_df["UserID"] == uid]["MovieID"])
        candidates = list(user_items - rated)
        preds = [(iid, algo.predict(uid, iid).est) for iid in candidates]
        top_k = sorted(preds, key=lambda x: x[1], reverse=True)[:k]
        recommendations[uid] = [iid for iid, _ in top_k]
    return recommendations


def svd_grid_search(data):
    """
    Performs grid search to find the best SVD hyperparameters.
    Returns the best algorithm and the best parameters.
    """
    param_grid = {
        "n_factors": [50, 100, 150],
        "lr_all": [0.002, 0.005, 0.01],
        "reg_all": [0.02, 0.05, 0.1],
        "n_epochs": [20, 30, 40],
    }
    gs = GridSearchCV(
        SVD, param_grid, measures=["rmse"], cv=3, n_jobs=4, joblib_verbose=1
    )
    gs.fit(data)
    logging.info("Best RMSE score from grid search: %.4f" % gs.best_score["rmse"])
    logging.info("Best SVD parameters: %s" % gs.best_params["rmse"])
    return gs.best_estimator["rmse"], gs.best_params["rmse"]


def train_and_evaluate_svd(
    ratings_df: pd.DataFrame, model_path: str, apply_recency: bool = True
) -> dict:
    """
    Trains and evaluates an SVD-based recommendation system using temporal information.
    Returns a dictionary of evaluation metrics for comparison.
    """
    logging.info(
        "Starting training and evaluation of SVD-based recommendation system with temporal information."
    )
    # Preprocess
    ratings_df = preprocess_with_timestamp(ratings_df)
    ratings_df = filter_noise_outliers(ratings_df)
    if apply_recency:
        ratings_df = soft_weight_by_recency(ratings_df, decay_rate=0.01)
    # Time-based split
    train_df, test_df = time_based_split(ratings_df, test_ratio=0.2)
    reader = Reader(rating_scale=(ratings_df["Rating"].min(), ratings_df["Rating"].max()))
    train_data = Dataset.load_from_df(train_df[["UserID", "MovieID", "Rating"]], reader)
    testset = list(test_df[["UserID", "MovieID", "Rating"]].itertuples(index=False, name=None))
    # Grid search for best SVD hyperparameters
    best_algo, best_params = svd_grid_search(train_data)
    trainset = train_data.build_full_trainset()
    best_algo.fit(trainset)
    logging.info(f"SVD model trained with best params: {best_params}")
    # Evaluate
    predictions = best_algo.test(testset)
    rmse = accuracy.rmse(predictions)
    logging.info("RMSE on test set: %.4f", rmse)
    # Save the best-performing model
    save_model(best_algo, model_path)
    logging.info("Best-performing model saved to %s", model_path)
    # Calculate Precision and Recall at K for the best model
    precision, recall = precision_recall_at_k(predictions, k=10, threshold=3.5)
    logging.info("Precision@10: %.4f, Recall@10: %.4f", precision, recall)
    # Additional metrics
    users = test_df["UserID"].unique()
    items = ratings_df["MovieID"].unique()
    item_popularity = Counter(train_df["MovieID"])
    item_similarity = {(i, j): 0 for i in items for j in items if i != j}
    recommendations = recommend_top_k(best_algo, users, items, k=10)
    coverage = get_coverage(recommendations, items)
    diversity = get_diversity(recommendations, item_similarity)
    novelty = get_novelty(recommendations, item_popularity)
    logging.info(
        "Coverage: %.4f, Diversity: %.4f, Novelty: %.2f",
        coverage,
        diversity,
        novelty,
    )
    return {
        "RMSE": rmse,
        "Precision@10": precision,
        "Recall@10": recall,
        "Coverage": coverage,
        "Diversity": diversity,
        "Novelty": novelty,
        "BestParams": best_params
    }


def compare_models(ratings_df, model_path_with, model_path_without):
    """
    Trains, saves, and compares models with and without recency weighting.
    Logs and prints their evaluation metrics.
    """
    logging.info("Training model WITH recency weighting...")
    metrics_with = train_and_evaluate_svd(
        ratings_df.copy(), model_path_with, apply_recency=True
    )
    logging.info("Training model WITHOUT recency weighting...")
    metrics_without = train_and_evaluate_svd(
        ratings_df.copy(), model_path_without, apply_recency=False
    )
    # Print comparison
    print("Model Comparison:")
    print("With Recency:", metrics_with)
    print("Without Recency:", metrics_without)
    # Save to CSV for tracking
    df = pd.DataFrame([metrics_with, metrics_without], index=["With Recency", "Without Recency"])
    df.to_csv("model_comparison.csv")
    logging.info("Model comparison saved to model_comparison.csv")


if __name__ == "__main__":
    logging.info("Starting the recommendation system pipeline.")
    try:
        # Path to the SQLite database
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")
        model_path_with = os.path.join(project_dir, "models", "svd_movielens_1m_with_recency.pkl")
        model_path_without = os.path.join(project_dir, "models", "svd_movielens_1m_without_recency.pkl")
        logging.debug(
            "Project directory: %s, DB path: %s, Model path (with recency): %s, Model path (without recency): %s",
            project_dir,
            db_path,
            model_path_with,
            model_path_without,
        )
        # Load the ratings data
        ratings_df = load_data_from_sqlite(db_path)
        # Train, save, and compare both models
        compare_models(ratings_df, model_path_with, model_path_without)
        logging.info("Recommendation system pipeline completed successfully.")
    except Exception as e:
        logging.critical("Pipeline failed: %s", e, exc_info=True)
