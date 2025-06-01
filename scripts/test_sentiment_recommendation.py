"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script provides an interactive testing environment for the hybrid recommendation system.
It allows users to generate hybrid recommendations by combining collaborative filtering,
content-based filtering, and sentiment analysis based on user input.
"""

import logging
import os
import sqlite3

import pandas as pd
from googletrans import Translator
from helpers import load_model, print_table
from textblob import TextBlob

LOG_FILE = "test_hybrid_recommendation.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log file
        logging.StreamHandler(),  # Console output
    ],
)


def analyze_user_input(user_input: str) -> float:
    """
    Analyzes the sentiment of the user's input text.

    :param user_input: The text input from the user describing their mood and preferences.
    :return: A sentiment polarity score (-1.0 to 1.0).
    """
    translator = Translator()
    translation = translator.translate(user_input, dest="en")
    logging.info("Translating user input to English: %s", translation.text)
    blob = TextBlob(translation.text)

    logging.info("Analyzing user input sentiment.")
    sentiment_score = blob.sentiment.polarity
    logging.info("User input sentiment score: %.2f", sentiment_score)
    return sentiment_score


# Emotion to genre mapping
EMOTION_GENRE_MAP = {
    "happy": ["Comedy", "Adventure", "Family"],
    "sad": ["Drama", "Romance"],
    "excited": ["Action", "Sci-Fi", "Thriller"],
    "calm": ["Documentary", "History"],
    "fear": ["Thriller", "Horror"],
    "angry": ["Crime", "Action"],
    "surprise": ["Mystery", "Fantasy"],
    "neutral": ["Drama", "Documentary", "History"],
}


# Simple emotion classifier
def classify_emotion(user_input: str) -> str:
    """
    Classifies the user's emotion based on keywords and sentiment polarity.
    Returns one of the keys in EMOTION_GENRE_MAP.
    """
    blob = TextBlob(user_input)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    text = user_input.lower()
    # Keyword-based detection (expand as needed)
    if any(word in text for word in ["happy", "joy", "excited", "great", "awesome"]):
        return "happy"
    if any(word in text for word in ["sad", "down", "depressed", "unhappy"]):
        return "sad"
    if any(word in text for word in ["calm", "relaxed", "peaceful"]):
        return "calm"
    if any(word in text for word in ["scared", "afraid", "fear", "terrified"]):
        return "fear"
    if any(word in text for word in ["angry", "mad", "furious"]):
        return "angry"
    if any(word in text for word in ["surprised", "shocked"]):
        return "surprise"
    if any(word in text for word in ["action", "adrenaline", "thrill"]):
        return "excited"
    # Fallback to polarity
    if polarity > 0.2:
        return "happy"
    elif polarity < -0.2:
        return "sad"
    else:
        return "neutral"


def recommend_movies_based_on_emotion(
    movies_df: pd.DataFrame,
    emotion: str,
    n: int = 10,
) -> pd.DataFrame:
    """
    Recommends movies by boosting scores for genres matching the detected emotion.
    Does not exclude other genres, but prioritizes matches.
    """
    genres = EMOTION_GENRE_MAP.get(emotion, EMOTION_GENRE_MAP["neutral"])

    # Score: 2 for primary genre match, 1 for secondary, 0 otherwise
    def score_row(row):
        for g in genres:
            if g in row["Genres"]:
                return 2
        return 0

    movies_df = movies_df.copy()
    movies_df["EmotionScore"] = movies_df.apply(score_row, axis=1)
    # Sort by score, then by popularity (if available), then by title
    recommendations = movies_df.sort_values(
        by=["EmotionScore", "Title"], ascending=[False, True]
    ).head(n)
    return recommendations


def recommend_movies_based_on_sentiment(
    movies_df: pd.DataFrame,
    # similarity_df: pd.DataFrame,
    sentiment_score: float,
    n: int = 10,
) -> pd.DataFrame:
    """
    Recommends movies based on the user's sentiment score.

    :param movies_df: DataFrame containing movie information.
    :param similarity_df: DataFrame containing content-based similarity scores.
    :param sentiment_score: The sentiment polarity score from the user's input.
    :param n: The number of recommendations to generate.
    :return: A DataFrame containing the top N recommended movies with details.
    """
    logging.info("Recommending movies based on sentiment score %.2f.", sentiment_score)

    # # Filter movies based on sentiment score
    # if sentiment_score > 0:
    #     # Positive sentiment: Recommend movies with high sentiment scores
    #     filtered_movies = movies_df[movies_df["SentimentScore"] > 0]
    # elif sentiment_score < 0:
    #     # Negative sentiment: Recommend movies with low sentiment scores
    #     filtered_movies = movies_df[movies_df["SentimentScore"] < 0]
    # else:
    #     # Neutral sentiment: Recommend movies with average sentiment scores
    #     filtered_movies = movies_df[
    #         (movies_df["SentimentScore"] >= -0.1) & (movies_df["SentimentScore"] <= 0.1)
    #     ]

    # # Calculate the average similarity score for each movie
    # avg_similarity = similarity_df.mean(axis=1).sort_values(ascending=False)

    # # Get the top N movies
    # filtered_movies["AvgSimilarity"] = filtered_movies["MovieID"].map(avg_similarity)
    # recommendations = filtered_movies.sort_values(by="AvgSimilarity", ascending=False).head(n)

    # Filter movies based on sentiment score
    if sentiment_score > 0:
        # Positive sentiment: Recommend uplifting genres
        filtered_movies = movies_df[
            movies_df["Genres"].str.contains("Comedy|Adventure|Family", na=False)
        ]
    elif sentiment_score < 0:
        # Negative sentiment: Recommend reflective genres
        filtered_movies = movies_df[
            movies_df["Genres"].str.contains("Drama|Romance|Thriller", na=False)
        ]
    else:
        # Neutral sentiment: Recommend popular genres
        filtered_movies = movies_df[
            movies_df["Genres"].str.contains("Action|Sci-Fi|Fantasy", na=False)
        ]

    # Get the top N movies
    recommendations = filtered_movies.head(n)

    return recommendations


if __name__ == "__main__":
    logging.info("Starting the hybrid recommendation system testing script.")

    try:
        # Paths
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")
        model_paths = [
            os.path.join(project_dir, "models", "svd_movielens_1m_with_recency.pkl"),
            os.path.join(project_dir, "models", "svd_movielens_1m_without_recency.pkl"),
        ]
        # Load the trained model
        loaded_models = [load_model(path) for path in model_paths]

        # Load the ratings, movies, and reviews data
        conn = sqlite3.connect(db_path)
        ratings_df = pd.read_sql("SELECT UserID, MovieID, Rating FROM ratings", conn)
        movies_df = pd.read_sql("SELECT MovieID, Title, Genres FROM movies", conn)
        conn.close()

        # Calculate content-based similarity
        # similarity_method = "tfidf"  # Change to "count" if needed
        # similarity_df = calculate_content_similarity(
        # #     movies_df, method=similarity_method
        # )

        # Interactive testing
        user_input = input("Descreva seu humor e o que você gostaria de assistir hoje: ")
        try:
            emotion = classify_emotion(user_input)
            logging.info(f"Detected emotion: {emotion}")
            print(f"\nDetected emotion: {emotion.capitalize()}")
            print(f"Recommending movies in genres: {', '.join(EMOTION_GENRE_MAP[emotion])}")
            recommendations = recommend_movies_based_on_emotion(
                movies_df, emotion, n=10
            )
            if recommendations.empty:
                print("No recommendations found for your mood. Showing popular movies instead.")
                recommendations = movies_df.head(10)
        except Exception as e:
            print("Could not detect emotion or recommend based on your input. Showing popular movies.")
            logging.error(f"Emotion detection/recommendation failed: {e}")
            recommendations = movies_df.head(10)

        # Display recommendations
        print_table(recommendations, "Personalized Recommendations Based on Your Mood")

    except Exception as e:
        logging.critical("Interactive testing failed: %s", e, exc_info=True)
