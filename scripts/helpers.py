import logging
import pickle
import pandas as pd
from tabulate import tabulate


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