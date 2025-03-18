"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This module provides functions to load the MovieLens 1M dataset and save it to a SQLite database.

Functions:
    extract_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Extracts the MovieLens 1M dataset and returns DataFrames for users, movies, and ratings.

    save_to_sqlite(users: pd.DataFrame, ratings: pd.DataFrame, movies: pd.DataFrame, db_path: str) -> None:
        Saves the DataFrames to a SQLite database.
"""

import sqlite3
import os
from typing import Tuple

import pandas as pd


def extract_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extracts the MovieLens 1M dataset and returns DataFrames for users, movies, and ratings.

    :param data_path: Path to the directory containing the dataset files.
    :return: A tuple containing three DataFrames: users, movies, and ratings.
    """
    # Define column names for each dataset
    users_cols = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
    ratings_cols = ["UserID", "MovieID", "Rating", "Timestamp"]
    movies_cols = ["MovieID", "Title", "Genres"]

    # Construct file paths
    users_file = os.path.join(data_path, "users.dat")
    ratings_file = os.path.join(data_path, "ratings.dat")
    movies_file = os.path.join(data_path, "movies.dat")

    # Load the files into DataFrames
    users_df = pd.read_csv(users_file, sep="::", names=users_cols, engine="python")
    ratings_df = pd.read_csv(
        ratings_file, sep="::", names=ratings_cols, engine="python"
    )
    movies_df = pd.read_csv(
        movies_file, sep="::", names=movies_cols, engine="python", encoding="latin-1"
    )

    return users_df, ratings_df, movies_df


def clean_data(
    users_df: pd.DataFrame, ratings_df: pd.DataFrame, movies_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleans the DataFrames by removing unnecessary columns and rows.

    :param users_df: DataFrame containing user data.
    :param ratings_df: DataFrame containing rating data.
    :param movies_df: DataFrame containing movie data.
    :return: A tuple containing the cleaned DataFrames: users, movies, and ratings.
    """
    # Remove lines with missing values
    users_df.dropna(inplace=True)
    ratings_df.dropna(inplace=True)
    movies_df.dropna(inplace=True)

    # Ensure column datatypes
    users_df = users_df.astype(
        {"UserID": "int32", "Age": "int32", "Occupation": "int32"}
    )
    ratings_df = ratings_df.astype(
        {"UserID": "int32", "MovieID": "int32", "Rating": "float32"}
    )
    movies_df = movies_df.astype({"MovieID": "int32"})

    # Normalize Gender column (0 for F, 1 for M)
    users_df["Gender"] = users_df["Gender"].map({"F": 0, "M": 1})

    # Convert Timestamp column to datetime
    ratings_df["Timestamp"] = pd.to_datetime(ratings_df["Timestamp"], unit="s")

    # Remove the Zip-code column from the users DataFrame
    users_df = users_df.drop(columns=["Zip-code"])

    # Create a new column for the year of release
    movies_df["Year"] = movies_df["Title"].str.extract(r"\((\d{4})\)").astype("Int32")

    # Create a new column for the genres
    genre_list = set("|".join(movies_df["Genres"]).split("|"))
    for genre in genre_list:
        movies_df[genre] = movies_df["Genres"].apply(lambda x: int(genre in x))

    return users_df, ratings_df, movies_df


def save_to_sqlite(
    users_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    db_path: str,
) -> None:
    """
    Saves the DataFrames to a SQLite database.

    :param users_df: DataFrame containing user data.
    :param ratings_df: DataFrame containing rating data.
    :param movies_df: DataFrame containing movie data.
    :param db_path: Path to the SQLite database file.
    """
    # Ensure the directory for the database file exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        users_df.to_sql("users", conn, if_exists="replace", index=False)
        ratings_df.to_sql("ratings", conn, if_exists="replace", index=False)
        movies_df.to_sql("movies", conn, if_exists="replace", index=False)
        print(f"Data cleaned and saved to {db_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, "dataset", "ml-1m")
    db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")

    users, ratings, movies = extract_data(data_path)
    save_to_sqlite(users, ratings, movies, db_path)
