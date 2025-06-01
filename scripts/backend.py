import os
import sqlite3
import pandas as pd
from scripts.helpers import load_model
from scripts.hybrid_recommendation_system import calculate_content_similarity

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_DIR, "dataset", "sqlite", "movielens_1m.db")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "svd_movielens_1m_with_recency.pkl")

# Load data and model
def load_data():
    conn = sqlite3.connect(DB_PATH)
    ratings_df = pd.read_sql("SELECT UserID, MovieID, Rating FROM ratings", conn)
    movies_df = pd.read_sql("SELECT MovieID, Title, Genres FROM movies", conn)
    conn.close()
    return ratings_df, movies_df

def load_backend():
    ratings_df, movies_df = load_data()
    algo = load_model(MODEL_PATH)
    similarity_method = "tfidf"
    similarity_df = calculate_content_similarity(movies_df, method=similarity_method)
    return ratings_df, movies_df, algo, similarity_df