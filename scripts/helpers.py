"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script provides helper functions for the recommendation system. These functions
include utilities for printing tables, validating data, loading models, and retrieving
user and movie details.
"""

import logging
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, accuracy
from tabulate import tabulate
from textblob import TextBlob


# --- Data Validation and Loading ---
def print_table(df: pd.DataFrame, title: str) -> None:
    """
    Prints a DataFrame in a tabular format using the tabulate library.

    :param df: The DataFrame to print.
    :param title: Title of the table.
    """
    print(f"\n{title}")
    print(tabulate(df.head(10), headers="keys", tablefmt="grid"))


def validate_data(
    users_df: pd.DataFrame, ratings_df: pd.DataFrame, movies_df: pd.DataFrame
) -> None:
    """
    Validates the cleaned DataFrames to ensure data integrity.

    :param users_df: DataFrame containing user data.
    :param ratings_df: DataFrame containing rating data.
    :param movies_df: DataFrame containing movie data.
    """
    logging.info("Validating cleaned data.")

    # Check for duplicate UserIDs
    if users_df["UserID"].duplicated().any():
        raise ValueError("Duplicate UserIDs found in users DataFrame.")

    # Check for duplicate MovieIDs
    if movies_df["MovieID"].duplicated().any():
        raise ValueError("Duplicate MovieIDs found in movies DataFrame.")

    # Check for invalid ratings
    if not ratings_df["Rating"].between(0.5, 5.0).all():
        raise ValueError("Invalid ratings found in ratings DataFrame.")

    logging.info("Data validation passed successfully.")


def reverse_gender_mapping(gender: int) -> str:
    """
    Reverses the gender mapping (0 -> F, 1 -> M).

    :param gender: Mapped gender value (0 or 1).
    :return: Original gender value ('F' or 'M').
    """
    return {0: "F", 1: "M"}.get(gender, "Unknown")


def load_model(model_path: str):
    """
    Loads a trained model from a file.

    :param model_path: Path to the saved model file.
    :return: The loaded Surprise model.
    """
    logging.info("Loading the trained model from %s", model_path)
    try:
        with open(model_path, "rb") as model_file:
            algo = pickle.load(model_file)
        logging.info("Model loaded successfully.")
        return algo
    except Exception as e:
        logging.error("Failed to load the model: %s", e, exc_info=True)
        raise


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


# --- User and Movie Details ---
def get_user_details(user_id: int, users_df: pd.DataFrame) -> dict:
    """
    Retrieves user details from the users DataFrame.

    :param user_id: The ID of the user.
    :param users_df: DataFrame containing user information.
    :return: A dictionary with user details (Gender, Age, Occupation).
    """
    user_details = users_df[users_df["UserID"] == user_id].iloc[0].to_dict()
    return user_details


def get_movie_details(movie_id: int, movies_df: pd.DataFrame) -> dict:
    """
    Retrieves movie details from the movies DataFrame.

    :param movie_id: The ID of the movie.
    :param movies_df: DataFrame containing movie information.
    :return: A dictionary with movie details (Title, Year, Genre).
    """
    movie_details = movies_df[movies_df["MovieID"] == movie_id].iloc[0].to_dict()
    return movie_details


# --- Recommendation Metrics ---
def calculate_rmse(predictions) -> float:
    """
    Calculates the RMSE for a given set of predictions.

    :param predictions: List of predictions from the Surprise library.
    :return: The RMSE value.
    """

    rmse = accuracy.rmse(predictions, verbose=False)
    logging.info("Calculated RMSE: %.4f", rmse)
    return rmse


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Calculates Precision and Recall at K.

    :param predictions: List of predictions from the Surprise library.
    :param k: Number of recommendations to consider.
    :param threshold: Rating threshold to consider a recommendation as relevant.
    :return: Precision and Recall at K.
    """
    logging.info("Calculating Precision and Recall at K.")
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in top_k)
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold)) for (est, true_r) in top_k
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    avg_precision = sum(precisions.values()) / len(precisions)
    avg_recall = sum(recalls.values()) / len(recalls)
    logging.info("Precision@K: %.4f, Recall@K: %.4f", avg_precision, avg_recall)
    return avg_precision, avg_recall


def evaluate_recommendations(
    recommendations: pd.DataFrame, user_ratings: pd.DataFrame, n: int = 10
) -> dict:
    """
    Evaluates the quality of recommendations using Precision@N, Recall@N, and F1-Score.

    :param recommendations: DataFrame containing the recommended movies.
    :param user_ratings: DataFrame containing the user's actual ratings.
    :param n: The number of recommendations to evaluate.
    :return: A dictionary containing evaluation metrics.
    """
    logging.info("Evaluating recommendations.")

    # Get the top N recommended movie IDs
    recommended_movie_ids = recommendations["MovieID"].head(n).tolist()

    # Get the relevant movie IDs (movies rated 4 or higher by the user)
    relevant_movie_ids = user_ratings[user_ratings["Rating"] >= 4]["MovieID"].tolist()

    # Calculate Precision@N
    true_positives = len(set(recommended_movie_ids) & set(relevant_movie_ids))
    precision_at_n = true_positives / n if n > 0 else 0

    # Calculate Recall@N
    recall_at_n = true_positives / len(relevant_movie_ids) if relevant_movie_ids else 0

    # Calculate F1-Score
    if precision_at_n + recall_at_n > 0:
        f1_score = 2 * (precision_at_n * recall_at_n) / (precision_at_n + recall_at_n)
    else:
        f1_score = 0

    logging.info(
        "Evaluation Metrics - Precision@%d: %.2f, Recall@%d: %.2f, F1-Score: %.2f",
        n,
        precision_at_n,
        n,
        recall_at_n,
        f1_score,
    )

    return {
        "Precision@N": precision_at_n,
        "Recall@N": recall_at_n,
        "F1-Score": f1_score,
    }


def get_coverage(recommendations, all_items):
    recommended_items = set([iid for recs in recommendations.values() for iid in recs])
    return len(recommended_items) / len(all_items)


def get_diversity(recommendations, item_similarity):
    diversities = []
    for recs in recommendations.values():
        if len(recs) < 2:
            continue
        pairs = [
            (recs[i], recs[j])
            for i in range(len(recs))
            for j in range(i + 1, len(recs))
        ]
        dissimilarities = [1 - item_similarity.get((a, b), 0) for a, b in pairs]
        if dissimilarities:
            diversities.append(np.mean(dissimilarities))
    return np.mean(diversities) if diversities else 0


def get_novelty(recommendations, item_popularity):
    all_counts = np.array(list(item_popularity.values()))
    ranks = {
        iid: (all_counts > item_popularity[iid]).sum() + 1 for iid in item_popularity
    }
    novelty_scores = []
    for recs in recommendations.values():
        novelty_scores += [ranks.get(iid, 0) for iid in recs]
    return np.mean(novelty_scores) if novelty_scores else 0


# --- Content-Based Similarity ---
def calculate_content_similarity(
    movies_df: pd.DataFrame, method: str = "count", use_sentiment: bool = False
) -> pd.DataFrame:
    """
    Calculates the cosine similarity between movies based on their genres.

    :param movies_df: DataFrame containing movie information.
    :param method: The method to use for vectorization ("count" or "tfidf").
    :param use_sentiment: Whether to include SentimentScore in the similarity computation.
    :return: A DataFrame containing the similarity scores between movies.

    """
    logging.info("Calculating content-based similarity for movies.")

    # Extract the year from the title if not already present
    if "Year" not in movies_df.columns:
        movies_df["Year"] = movies_df["Title"].str.extract(r"\((\d{4})\)").fillna("")

    # Combine Genres, Title, and Year into a single string for each movie
    movies_df["CombinedFeatures"] = (
        movies_df["Genres"].fillna("")
        + "|"
        + movies_df["Title"].fillna("")
        + "|"
        + movies_df["Year"].fillna("")
    )

    # Optionally include SentimentScore with a weight
    if use_sentiment and "SentimentScore" in movies_df.columns:
        sentiment_weight = 0.5  # Adjust the weight as needed
        movies_df["CombinedFeatures"] += "|" + (
            movies_df["SentimentScore"] * sentiment_weight
        ).astype(str)

    # Define a custom tokenizer to split on spaces
    def custom_tokenizer(text):
        return text.split("|")

    # Choose the vectorizer based on the method
    if method == "count":
        vectorizer = CountVectorizer(tokenizer=custom_tokenizer, token_pattern=None)
    elif method == "tfidf":
        vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, token_pattern=None)
    else:
        raise ValueError("Invalid method. Choose 'count' or 'tfidf'.")

    # Create the feature matrix
    feature_matrix = vectorizer.fit_transform(movies_df["CombinedFeatures"])

    # Calculate cosine similarity between movies
    similarity_matrix = cosine_similarity(feature_matrix)

    # Convert to DataFrame for easier handling
    similarity_df = pd.DataFrame(
        similarity_matrix, index=movies_df["MovieID"], columns=movies_df["MovieID"]
    )
    logging.info("Content-based similarity calculation completed.")
    return similarity_df


def apply_time_decay_to_similarity(similarity_df, movies_df, decay_rate=0.001):
    """
    Applies exponential time decay to content-based similarity based on movie release year (if available).
    Assumes movies_df has columns: MovieID, Year.
    """
    if "Year" not in movies_df.columns:
        return similarity_df  # No year info, skip
    year_dict = movies_df.set_index("MovieID")["Year"].to_dict()
    max_year = max(year_dict.values())
    for i in similarity_df.index:
        for j in similarity_df.columns:
            year_i = year_dict.get(i, max_year)
            year_j = year_dict.get(j, max_year)
            avg_year = (year_i + year_j) / 2
            decay = np.exp(-decay_rate * (max_year - avg_year))
            similarity_df.loc[i, j] *= decay
    return similarity_df


def calculate_sentiment_scores(
    movies_df: pd.DataFrame, reviews_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates sentiment scores for movies based on user reviews.

    :param movies_df: DataFrame containing movie information.
    :param reviews_df: DataFrame containing user reviews with MovieID and ReviewText.
    :return: DataFrame with an additional SentimentScore column for each movie.
    """
    logging.info("Calculating sentiment scores for movies.")

    # Aggregate reviews for each movie
    reviews_agg = (
        reviews_df.groupby("MovieID")["ReviewText"].apply(" ".join).reset_index()
    )

    # Calculate sentiment scores using TextBlob
    reviews_agg["SentimentScore"] = reviews_agg["ReviewText"].apply(
        lambda text: TextBlob(text).sentiment.polarity
    )

    # Merge sentiment scores with the movies DataFrame
    movies_df = movies_df.merge(
        reviews_agg[["MovieID", "SentimentScore"]], on="MovieID", how="left"
    )
    movies_df["SentimentScore"] = movies_df["SentimentScore"].fillna(
        0
    )  # Default to neutral sentiment

    logging.info("Sentiment scores calculated successfully.")
    return movies_df


def save_similarity_matrix(similarity_df, path):
    similarity_df.to_pickle(path)


def load_similarity_matrix(path):
    return pd.read_pickle(path)


# --- Hybrid Recommendation Utilities ---
def get_dynamic_alpha(
    user_id, ratings_df, min_alpha=0.3, max_alpha=1.0, cold_start_threshold=5
):
    """
    Returns a dynamic alpha for a user: alpha=1 for new users (cold start),
    otherwise increases from min_alpha to max_alpha with number of ratings.
    """
    n_ratings = len(ratings_df[ratings_df["UserID"] == user_id])
    if n_ratings <= cold_start_threshold:
        return 1.0
    alpha = min(max_alpha, min_alpha + 0.01 * n_ratings)
    return alpha


def recommend_top_k(hybrid_scores, users, items, k=10):
    from collections import defaultdict

    user_scores = defaultdict(list)
    for user_id, movie_id, score in hybrid_scores:
        user_scores[user_id].append((movie_id, score))
    recommendations = {}
    for uid in users:
        rated = set([movie_id for movie_id, _ in user_scores[uid]])
        candidates = [(mid, s) for mid, s in user_scores[uid] if mid not in rated]
        ranked = sorted(user_scores[uid], key=lambda x: x[1], reverse=True)
        recommendations[uid] = [mid for mid, _ in ranked[:k]]
    return recommendations


def precompute_recommendations(
    svd_model,
    ratings_df,
    similarity_df,
    users,
    items,
    k=10,
    alpha=0.7,
    cache_path="models/user_recommendations.pkl",
):
    hybrid_scores = []
    for user_id in users:
        user_rated = set(ratings_df[ratings_df["UserID"] == user_id]["MovieID"])
        for movie_id in items:
            if movie_id in user_rated:
                continue
            cf_score = svd_model.predict(user_id, movie_id).est
            cb_score = 0
            for rated_movie_id in user_rated:
                cb_score += similarity_df.loc[movie_id, rated_movie_id]
            cb_score /= len(user_rated) if user_rated else 1
            a = get_dynamic_alpha(
                user_id, ratings_df, min_alpha=alpha
            )  # Use min_alpha for weighting
            hybrid_score = a * cf_score + (1 - a) * cb_score
            hybrid_scores.append((user_id, movie_id, hybrid_score))
    user_scores = defaultdict(list)
    for user_id, movie_id, score in hybrid_scores:
        user_scores[user_id].append((movie_id, score))
    recommendations = {}
    for uid in users:
        ranked = sorted(user_scores[uid], key=lambda x: x[1], reverse=True)
        recommendations[uid] = [mid for mid, _ in ranked[:k]]
    with open(cache_path, "wb") as f:
        pickle.dump(recommendations, f)
    return recommendations


def cold_start_recommendations(user_id, movies_df, ratings_df, similarity_df, k=10):
    rated = set(ratings_df[ratings_df["UserID"] == user_id]["MovieID"])
    all_items = set(movies_df["MovieID"])
    not_rated = list(all_items - rated)
    item_popularity = ratings_df["MovieID"].value_counts()
    popular_items = [iid for iid in item_popularity.index if iid in not_rated][:k]
    if len(popular_items) < k:
        extra = [iid for iid in not_rated if iid not in popular_items][
            : k - len(popular_items)
        ]
        popular_items += extra
    return popular_items[:k]


def cold_start_item_recommendations(item_id, ratings_df, similarity_df, k=10):
    user_activity = ratings_df["UserID"].value_counts()
    return list(user_activity.index[:k])


def rerank_for_diversity_novelty(
    recommendations, item_similarity, item_popularity, lambda_div=0.5, lambda_nov=0.5
):
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
                div = (
                    min([1 - item_similarity.get((iid, sid), 0) for sid in selected])
                    if selected
                    else 1
                )
                nov = 1 / (item_popularity.get(iid, 1))
                scores.append((iid, lambda_div * div + lambda_nov * nov))
            if scores:
                best_iid = max(scores, key=lambda x: x[1])[0]
                selected.append(best_iid)
        reranked[uid] = selected
    return reranked


def cache_popular_items(
    ratings_df, movies_df, k=10, cache_path="models/popular_items.pkl"
):
    item_popularity = ratings_df["MovieID"].value_counts()
    popular_items = [
        iid for iid in item_popularity.index if iid in set(movies_df["MovieID"])
    ][:k]
    import pickle

    with open(cache_path, "wb") as f:
        pickle.dump(popular_items, f)
    return popular_items


# --- Recommendation Utilities ---
def get_top_n_recommendations(algo, ratings_df, movies_df, user_id, n=10):
    """
    Generates the top N recommendations for a given user.
    :param algo: The trained Surprise model.
    :param ratings_df: DataFrame containing the ratings data.
    :param movies_df: DataFrame containing movie information.
    :param user_id: The ID of the user for whom to generate recommendations.
    :param n: The number of recommendations to generate.
    :return: A list of top N recommended movies with details.
    """
    all_movie_ids = movies_df["MovieID"].unique()
    rated_movie_ids = ratings_df[ratings_df["UserID"] == user_id]["MovieID"].unique()
    recommendations = []
    for movie_id in all_movie_ids:
        if movie_id not in rated_movie_ids:
            pred = algo.predict(user_id, movie_id)
            recommendations.append((movie_id, pred.est))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_n = recommendations[:n]
    top_n_details = [
        {
            "MovieID": movie_id,
            "PredictedRating": predicted_rating,
            **get_movie_details(movie_id, movies_df),
        }
        for movie_id, predicted_rating in top_n
    ]
    return top_n_details


def compare_real_and_predicted_ratings(algo, ratings_df, movies_df, user_id):
    """
    Compares real ratings with predicted ratings for movies the user has already rated.
    :param algo: The trained Surprise model.
    :param ratings_df: DataFrame containing the ratings data.
    :param movies_df: DataFrame containing movie information.
    :param user_id: The ID of the user.
    :return: A DataFrame with real and predicted ratings for movies the user has rated.
    """
    user_ratings = ratings_df[ratings_df["UserID"] == user_id]
    comparisons = []
    for _, row in user_ratings.iterrows():
        movie_id = row["MovieID"]
        real_rating = row["Rating"]
        pred = algo.predict(user_id, movie_id)
        movie_details = get_movie_details(movie_id, movies_df)
        comparisons.append(
            {
                "MovieID": movie_id,
                "Title": movie_details["Title"],
                "Genres": movie_details["Genres"],
                "RealRating": real_rating,
                "PredictedRating": pred.est,
            }
        )
    comparisons_df = pd.DataFrame(comparisons)
    return comparisons_df


def get_user_id(ratings_df) -> int | None:
    try:
        associate = (
            input("Deseja associar a recomendação ao seu histórico? (s/n): ")
            .strip()
            .lower()
        )
        if associate == "s":
            user_id_input = input(
                "Digite seu UserID (ou pressione Enter para ignorar): "
            ).strip()
            if user_id_input:
                user_id = int(user_id_input)
                if user_id not in ratings_df["UserID"].values:
                    print(
                        "UserID não encontrado. Recomendação será feita sem personalização."
                    )
                    return None
                return user_id
            else:
                print(
                    "UserID não informado. Recomendação será feita sem personalização."
                )
    except Exception:
        print("Erro ao processar UserID. Recomendação será feita sem personalização.")
    return None


def collect_user_feedback():
    try:
        feedback = input("Você gostou dessas recomendações? (s/n): ").strip().lower()
        if feedback == "s":
            print("Obrigado pelo seu feedback positivo!")
        elif feedback == "n":
            print("Obrigado pelo seu feedback. Vamos trabalhar para melhorar!")
        else:
            print("Feedback não reconhecido. Obrigado mesmo assim!")
    except Exception:
        print("Erro ao registrar feedback.")
