"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script trains the hybrid recommendation system for the MovieLens 1M dataset. It includes
functions for loading data, preprocessing temporal information, training an SVD-based
recommendation model, and calculating content-based similarity. The best-performing model
is saved for later use.
"""

import logging
import os
import sqlite3

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from collections import defaultdict, Counter
from surprise.model_selection import train_test_split

from helpers import (
    calculate_content_similarity,
    calculate_sentiment_scores,
    get_movie_details,
    perform_cross_validation,
    save_model,
    load_model,
)

LOG_FILE = "hybrid_recommendation_system.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log file
        logging.StreamHandler(),  # Console output
    ],
)


def get_dynamic_alpha(user_id, ratings_df, min_alpha=0.3, max_alpha=1.0, cold_start_threshold=5):
    """
    Returns a dynamic alpha for a user: alpha=1 for new users (cold start),
    otherwise increases from min_alpha to max_alpha with number of ratings.
    """
    n_ratings = len(ratings_df[ratings_df['UserID'] == user_id])
    if n_ratings <= cold_start_threshold:
        return 1.0
    # Linear scaling for demonstration (can be changed to log or other)
    alpha = min(max_alpha, min_alpha + 0.01 * n_ratings)
    return alpha


def apply_time_decay_to_similarity(similarity_df, movies_df, decay_rate=0.001):
    """
    Applies exponential time decay to content-based similarity based on movie release year (if available).
    Assumes movies_df has columns: MovieID, Year.
    """
    if 'Year' not in movies_df.columns:
        return similarity_df  # No year info, skip
    year_dict = movies_df.set_index('MovieID')['Year'].to_dict()
    max_year = max(year_dict.values())
    for i in similarity_df.index:
        for j in similarity_df.columns:
            year_i = year_dict.get(i, max_year)
            year_j = year_dict.get(j, max_year)
            avg_year = (year_i + year_j) / 2
            decay = np.exp(-decay_rate * (max_year - avg_year))
            similarity_df.loc[i, j] *= decay
    return similarity_df


def rerank_for_diversity_novelty(recommendations, item_similarity, item_popularity, lambda_div=0.5, lambda_nov=0.5):
    """
    Rerank recommendations to increase diversity and novelty.
    lambda_div: weight for diversity, lambda_nov: weight for novelty.
    """
    reranked = {}
    for uid, recs in recommendations.items():
        if not recs:
            reranked[uid] = []
            continue
        selected = [recs[0]]
        for _ in range(1, len(recs)):
            candidates = [iid for iid in recs if iid not in selected]
            scores = []
            for iid in candidates:
                # Diversity: min similarity to already selected
                div = min([1 - item_similarity.get((iid, sid), 0) for sid in selected]) if selected else 1
                # Novelty: inverse popularity
                nov = 1 / (item_popularity.get(iid, 1))
                scores.append((iid, lambda_div * div + lambda_nov * nov))
            if scores:
                best_iid = max(scores, key=lambda x: x[1])[0]
                selected.append(best_iid)
        reranked[uid] = selected
    return reranked


def calculate_hybrid_scores(
    algo,
    ratings_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    alpha_func=None,
) -> list:
    """
    Calculates hybrid scores for all user-movie pairs in the dataset, using a dynamic alpha per user.
    :param algo: The trained collaborative filtering model.
    :param ratings_df: DataFrame containing the ratings data.
    :param similarity_df: DataFrame containing content-based similarity scores.
    :param alpha_func: Function to compute alpha per user.
    :return: A list of tuples containing (UserID, MovieID, HybridScore).
    """
    logging.info("Calculating hybrid scores for all user-movie pairs (dynamic alpha).")
    hybrid_scores = []
    for _, row in ratings_df.iterrows():
        user_id = row["UserID"]
        movie_id = row["MovieID"]
        alpha = alpha_func(user_id, ratings_df) if alpha_func else 0.7
        # Collaborative filtering score
        cf_score = algo.predict(user_id, movie_id).est
        # Content-based filtering score
        cb_score = 0
        user_rated_movies = ratings_df[ratings_df["UserID"] == user_id]["MovieID"].unique()
        for rated_movie_id in user_rated_movies:
            cb_score += similarity_df.loc[movie_id, rated_movie_id]
        cb_score /= len(user_rated_movies) if len(user_rated_movies) > 0 else 1
        # Hybrid score (weighted combination)
        hybrid_score = alpha * cf_score + (1 - alpha) * cb_score
        hybrid_scores.append((user_id, movie_id, hybrid_score))
    logging.info("Hybrid scores calculated successfully.")
    return hybrid_scores


def train_collaborative_model(ratings_df: pd.DataFrame) -> SVD:
    """
    Trains a collaborative filtering model using the Surprise library.

    :param ratings_df: DataFrame containing the ratings data.
    :return: The trained collaborative filtering model.
    """
    logging.info("Training collaborative filtering model.")

    # Prepare the data for Surprise
    reader = Reader(
        rating_scale=(ratings_df["Rating"].min(), ratings_df["Rating"].max())
    )
    data = Dataset.load_from_df(ratings_df[["UserID", "MovieID", "Rating"]], reader)

    # Perform cross-validation and track the best model
    best_model, best_rmse = perform_cross_validation(data, kfolds=5)
    logging.info("Best RMSE from cross-validation: %.4f", best_rmse)

    return best_model


def evaluate_hybrid_model(hybrid_scores, test_ratings_df, k=10, threshold=3.5):
    """
    Avalia o modelo híbrido usando Precision@K e Recall@K.
    :param hybrid_scores: lista de tuplas (UserID, MovieID, HybridScore)
    :param test_ratings_df: DataFrame com ratings reais do conjunto de teste
    :param k: número de recomendações
    :param threshold: limiar para considerar relevante
    """
    user_scores = defaultdict(list)
    for user_id, movie_id, score in hybrid_scores:
        user_scores[user_id].append((movie_id, score))

    precisions, recalls = [], []

    for user_id in test_ratings_df["UserID"].unique():
        relevant_items = set(
            test_ratings_df[
                (test_ratings_df["UserID"] == user_id)
                & (test_ratings_df["Rating"] >= threshold)
            ]["MovieID"]
        )
        if not relevant_items:
            continue
        ranked_items = sorted(user_scores[user_id], key=lambda x: x[1], reverse=True)
        recommended_items = [movie_id for movie_id, _ in ranked_items[:k]]
        n_rel = len(relevant_items)
        n_rec_k = len(recommended_items)
        n_rel_and_rec_k = len(set(recommended_items) & relevant_items)
        precisions.append(n_rel_and_rec_k / n_rec_k if n_rec_k else 0)
        recalls.append(n_rel_and_rec_k / n_rel if n_rel else 0)
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    print(f"Hybrid Precision@{k}: {avg_precision:.4f}, Recall@{k}: {avg_recall:.4f}")
    return avg_precision, avg_recall


def train_and_save_hybrid_model(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    model_path: str,
    similarity_method: str = "count",
    alpha: float = 0.7,
):
    """
    Trains and saves the hybrid recommendation system.

    :param ratings_df: DataFrame containing the ratings data.
    :param movies_df: DataFrame containing movie information.
    :param model_path: Path to save the trained model.
    :param similarity_method: The method to use for content-based similarity ("count" or "tfidf").
    :param alpha: Weight for combining collaborative and content-based scores (0 <= alpha <= 1).
    """
    logging.info("Starting training of the hybrid recommendation system.")

    # Train the collaborative filtering model
    collaborative_model = train_collaborative_model(ratings_df)

    # # Calculate sentiment scores
    # movies_df = calculate_sentiment_scores(movies_df, reviews_df)

    # Calculate content-based similarity
    similarity_df = calculate_content_similarity(movies_df, method=similarity_method)

    # Save the collaborative filtering model
    save_model(collaborative_model, model_path)
    logging.info("Collaborative filtering model saved to %s", model_path)

    logging.info("Hybrid recommendation system training completed.")


def get_coverage(recommendations, all_items):
    recommended_items = set([iid for recs in recommendations.values() for iid in recs])
    return len(recommended_items) / len(all_items)


def get_diversity(recommendations, item_similarity):
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
    all_counts = np.array(list(item_popularity.values()))
    ranks = {iid: (all_counts > item_popularity[iid]).sum() + 1 for iid in item_popularity}
    novelty_scores = []
    for recs in recommendations.values():
        novelty_scores += [ranks.get(iid, 0) for iid in recs]
    return np.mean(novelty_scores) if novelty_scores else 0


def recommend_top_k(hybrid_scores, users, items, k=10):
    from collections import defaultdict
    user_scores = defaultdict(list)
    for user_id, movie_id, score in hybrid_scores:
        user_scores[user_id].append((movie_id, score))
    recommendations = {}
    for uid in users:
        # Remove items already rated by user
        rated = set([movie_id for movie_id, _ in user_scores[uid]])
        candidates = [(mid, s) for mid, s in user_scores[uid] if mid not in rated]
        ranked = sorted(user_scores[uid], key=lambda x: x[1], reverse=True)
        recommendations[uid] = [mid for mid, _ in ranked[:k]]
    return recommendations


def cold_start_recommendations(user_id, movies_df, ratings_df, similarity_df, k=10):
    """
    Recommend top-k most popular or most similar items for cold-start users.
    """
    # Recommend most popular items not yet rated
    rated = set(ratings_df[ratings_df['UserID'] == user_id]['MovieID'])
    all_items = set(movies_df['MovieID'])
    not_rated = list(all_items - rated)
    # Popularity by number of ratings
    item_popularity = ratings_df['MovieID'].value_counts()
    popular_items = [iid for iid in item_popularity.index if iid in not_rated][:k]
    # If not enough, fill with random or similar items
    if len(popular_items) < k:
        extra = [iid for iid in not_rated if iid not in popular_items][:k-len(popular_items)]
        popular_items += extra
    return popular_items[:k]


def cold_start_item_recommendations(item_id, ratings_df, similarity_df, k=10):
    """
    Recommend top-k users for a new item based on content similarity or popularity.
    """
    # Recommend to most active users
    user_activity = ratings_df['UserID'].value_counts()
    return list(user_activity.index[:k])


if __name__ == "__main__":
    logging.info("Starting the hybrid recommendation system evaluation pipeline.")
    try:
        # Paths
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")
        model_path_with = os.path.join(project_dir, "models", "svd_movielens_1m_with_recency.pkl")
        model_path_without = os.path.join(project_dir, "models", "svd_movielens_1m_without_recency.pkl")
        # Load the ratings and movies data
        conn = sqlite3.connect(db_path)
        ratings_df = pd.read_sql("SELECT UserID, MovieID, Rating FROM ratings", conn)
        movies_df = pd.read_sql("SELECT MovieID, Title, Genres FROM movies", conn)
        conn.close()
        # Specify the similarity method (count or tfidf)
        similarity_method = "tfidf"  # or "count"
        # Calculate content-based similarity
        similarity_df = calculate_content_similarity(movies_df, method=similarity_method)
        # Tune alpha and similarity method
        best_alpha = 0.7
        best_sim_method = similarity_method
        best_metric = -1
        for alpha in np.arange(0.1, 1.05, 0.1):
            for sim_method in ["tfidf", "count"]:
                sim_df = calculate_content_similarity(movies_df, method=sim_method)
                # Load pretrained SVD model (with recency for tfidf, without for count as example)
                if sim_method == "tfidf":
                    svd_model = load_model(model_path_with)
                else:
                    svd_model = load_model(model_path_without)
                # Use dynamic alpha
                hybrid_scores = calculate_hybrid_scores(
                    svd_model, ratings_df, sim_df,
                    alpha_func=lambda uid, df: alpha
                )
                # Evaluate on a holdout set (here, use all ratings for simplicity)
                users = ratings_df['UserID'].unique()
                items = ratings_df['MovieID'].unique()
                recommendations = recommend_top_k(hybrid_scores, users, items, k=10)
                # Use coverage as main metric for tuning (could use others)
                coverage = get_coverage(recommendations, items)
                if coverage > best_metric:
                    best_metric = coverage
                    best_alpha = alpha
                    best_sim_method = sim_method
        logging.info(f"Best alpha: {best_alpha}, Best similarity method: {best_sim_method}, Best coverage: {best_metric:.4f}")
        # Final hybrid model with best params
        similarity_df = calculate_content_similarity(movies_df, method=best_sim_method)
        svd_model = load_model(model_path_with if best_sim_method=="tfidf" else model_path_without)
        hybrid_scores = calculate_hybrid_scores(
            svd_model, ratings_df, similarity_df,
            alpha_func=lambda uid, df: best_alpha if len(ratings_df[ratings_df['UserID']==uid])>5 else 1.0
        )
        users = ratings_df['UserID'].unique()
        items = ratings_df['MovieID'].unique()
        recommendations = recommend_top_k(hybrid_scores, users, items, k=10)
        # Post-processing: diversity/novelty reranking and metrics
        item_popularity = Counter(ratings_df['MovieID'])
        item_similarity = {(i, j): similarity_df.loc[i, j] if i in similarity_df.index and j in similarity_df.columns else 0 for i in items for j in items if i != j}
        reranked_recommendations = rerank_for_diversity_novelty(recommendations, item_similarity, item_popularity, lambda_div=0.5, lambda_nov=0.5)
        coverage = get_coverage(reranked_recommendations, items)
        diversity = get_diversity(reranked_recommendations, item_similarity)
        novelty = get_novelty(reranked_recommendations, item_popularity)
        logging.info(f"Final Hybrid Coverage: {coverage:.4f}, Diversity: {diversity:.4f}, Novelty: {novelty:.2f}")
        # Example: handle cold-start for a new user
        new_user_id = max(users) + 1
        cold_recs = cold_start_recommendations(new_user_id, movies_df, ratings_df, similarity_df, k=10)
        logging.info(f"Cold-start recommendations for new user: {cold_recs}")
        # Example: handle cold-start for a new item
        new_item_id = max(items) + 1
        cold_item_recs = cold_start_item_recommendations(new_item_id, ratings_df, similarity_df, k=10)
        logging.info(f"Cold-start recommendations for new item: {cold_item_recs}")
        logging.info("Hybrid recommendation system evaluation pipeline completed successfully.")
    except Exception as e:
        logging.critical("Pipeline failed: %s", e, exc_info=True)
