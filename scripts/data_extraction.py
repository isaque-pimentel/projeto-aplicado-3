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
import logging
import pandas as pd

LOG_FILE = "data_extraction.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log file
        logging.StreamHandler(),  # Console output
    ],
)


def extract_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extracts the MovieLens 1M dataset and returns DataFrames for users, movies, and ratings.

    :param data_path: Path to the directory containing the dataset files.
    :return: A tuple containing three DataFrames: users, movies, and ratings.
    """
    logging.info("Starting data extraction from %s", data_path)

    # Define column names for each dataset
    users_cols = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
    ratings_cols = ["UserID", "MovieID", "Rating", "Timestamp"]
    movies_cols = ["MovieID", "Title", "Genres"]

    try:
        # Construct file paths
        users_file = os.path.join(data_path, "users.dat")
        ratings_file = os.path.join(data_path, "ratings.dat")
        movies_file = os.path.join(data_path, "movies.dat")
        logging.debug(
            "Constructed file paths: users_file=%s, ratings_file=%s, movies_file=%s",
            users_file,
            ratings_file,
            movies_file,
        )

        # Load the files into DataFrames
        users_df = pd.read_csv(users_file, sep="::", names=users_cols, engine="python")
        ratings_df = pd.read_csv(
            ratings_file, sep="::", names=ratings_cols, engine="python"
        )
        movies_df = pd.read_csv(
            movies_file,
            sep="::",
            names=movies_cols,
            engine="python",
            encoding="latin-1",
        )
        logging.debug(
            "Loaded data into DataFrames: users_df=%d rows, ratings_df=%d rows, movies_df=%d rows",
            len(users_df),
            len(ratings_df),
            len(movies_df),
        )

        logging.info("Data successfully extracted.")
        return users_df, ratings_df, movies_df
    except Exception as e:
        logging.error("Failed to extract data: %s", e, exc_info=True)
        raise


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
    logging.info("Starting data cleaning process.")

    try:

        # Remove lines with missing values
        users_df.dropna(inplace=True)
        ratings_df.dropna(inplace=True)
        movies_df.dropna(inplace=True)
        logging.debug(
            "Dropped rows with missing values: users_df=%d rows, ratings_df=%d rows, movies_df=%d rows",
            len(users_df),
            len(ratings_df),
            len(movies_df),
        )

        # Ensure column datatypes
        users_df = users_df.astype(
            {"UserID": "int32", "Age": "int32", "Occupation": "int32"}
        )
        ratings_df = ratings_df.astype(
            {"UserID": "int32", "MovieID": "int32", "Rating": "float32"}
        )
        movies_df = movies_df.astype({"MovieID": "int32"})
        logging.debug("Ensured efficient datatypes for columns.")

        # Normalize Gender column (0 for F, 1 for M)
        users_df["Gender"] = users_df["Gender"].map({"F": 0, "M": 1})
        logging.debug("Normalized 'Gender' column.")

        # Convert Timestamp column to datetime
        ratings_df["Timestamp"] = pd.to_datetime(ratings_df["Timestamp"], unit="s")
        logging.debug("Converted 'Timestamp' column to datetime format.")

        # Remove the Zip-code column from the users DataFrame
        users_df = users_df.drop(columns=["Zip-code"])
        logging.debug("Dropped 'Zip-code' column from users DataFrame.")

        # Create a new column for the year of release
        movies_df["Year"] = (
            movies_df["Title"].str.extract(r"\((\d{4})\)").astype("Int32")
        )
        logging.debug("Extracted release year from movie titles.")

        # Create a new column for the genres
        genre_list = set("|".join(movies_df["Genres"]).split("|"))
        for genre in genre_list:
            movies_df[genre] = movies_df["Genres"].apply(lambda x: int(genre in x))
        logging.debug("Created genre columns for movies DataFrame.")

        logging.info("Data cleaning process completed successfully.")
        return users_df, ratings_df, movies_df

    except Exception as e:
        logging.error(
            "An error occurred while during data cleaning: %s", e, exc_info=True
        )
        raise


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
    logging.info("Saving data to SQLite database at %s", db_path)

    # Ensure the directory for the database file exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        users_df.to_sql("users", conn, if_exists="replace", index=False)
        ratings_df.to_sql("ratings", conn, if_exists="replace", index=False)
        movies_df.to_sql("movies", conn, if_exists="replace", index=False)
        logging.debug(
            "Data saved to SQLite database: users=%d rows, ratings=%d rows, movies=%d rows",
            len(users_df),
            len(ratings_df),
            len(movies_df),
        )

        logging.info("Data successfully saved to SQLite database.")
    except Exception as e:
        logging.error(
            "An error occurred while saving to database: %s", e, exc_info=True
        )
        raise
    finally:
        conn.close()
        logging.info("Database connection closed.")


if __name__ == "__main__":
    logging.info("Starting the HistFlix data processing pipeline.")

    try:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_dir, "dataset", "ml-1m")
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")

        logging.debug(
            "Project directory: %s, Data path: %s, DB path: %s",
            project_dir,
            data_path,
            db_path,
        )

        users, ratings, movies = extract_data(data_path)
        users, ratings, movies = clean_data(users, ratings, movies)
        save_to_sqlite(users, ratings, movies, db_path)
        logging.info("Pipeline completed successfully.")
    except Exception as e:
        logging.critical("Pipeline failed: %s", e, exc_info=True)
