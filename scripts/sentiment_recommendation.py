"""
Sentiment-based recommendation core functions for HistFlix.
Authors: B Baltuilhe, I Pimentel, K Pena

This module provides functions for emotion classification and sentiment-based movie recommendation.
"""

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from typing import Tuple, Optional
import ast
import os

try:
    from googletrans import Translator
    translator = Translator()
except ImportError:
    translator = None

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

EMOTION_KEYWORDS = {
    "happy": ["happy", "joy", "delighted", "cheerful", "content", "satisfied", "pleased", "funny", "amused", "smile", "laugh"],
    "sad": ["sad", "down", "depressed", "unhappy", "melancholy", "blue", "cry", "lonely", "heartbroken", "bittersweet", "nostalgic"],
    "calm": ["calm", "relaxed", "peaceful", "serene", "chill", "tranquil", "at ease"],
    "fear": ["scared", "afraid", "fear", "terrified", "anxious", "worried", "tense", "nervous"],
    "angry": ["angry", "mad", "furious", "annoyed", "irritated", "frustrated", "resentful"],
    "surprise": ["surprised", "shocked", "amazed", "astonished", "unexpected", "startled"],
    "excited": ["excited", "adrenaline", "thrill", "eager", "enthusiastic", "pumped", "curious", "energetic"],
    "bored": ["bored", "dull", "uninterested", "tired", "monotonous", "meh"],
    "neutral": ["ok", "fine", "neutral", "average", "so-so", "normal"]
}

analyzer = SentimentIntensityAnalyzer()

def classify_emotion(user_input: str) -> str:
    """
    Classifies the user's emotion based on keywords, sentiment polarity, and subjectivity.
    Returns one of the keys in EMOTION_GENRE_MAP.
    If the input is too short or only whitespace, returns 'neutral'.
    """
    if not user_input or not user_input.strip() or len(user_input.strip()) < 3:
        return "neutral"
    sentiment = TextBlob(user_input).sentiment
    polarity = getattr(sentiment, 'polarity', 0)
    subjectivity = getattr(sentiment, 'subjectivity', 0)
    text = user_input.lower()
    detected_emotions = []
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(word in text for word in keywords):
            detected_emotions.append(emotion)
    if subjectivity < 0.3:
        return "neutral"
    if detected_emotions:
        if 'excited' in detected_emotions and polarity > 0.5:
            return 'excited'
        return detected_emotions[0]
    if polarity > 0.5 and subjectivity > 0.5:
        return "excited"
    elif polarity > 0.2:
        return "happy"
    elif polarity < -0.5 and subjectivity > 0.5:
        return "angry"
    elif polarity < -0.2:
        return "sad"
    elif subjectivity > 0.7:
        return "surprise"
    else:
        return "neutral"

def preprocess_user_input(user_input: str) -> Tuple[str, Optional[str]]:
    """
    Preprocesses user input: trims, lowercases, and translates to English if needed.
    Returns (processed_text, translation_error_message)
    """
    text = user_input.strip()
    if not text:
        return "", None
    # Try to translate if not in English and translator is available
    if translator:
        try:
            detected = translator.detect(text)
            if detected.lang != 'en':
                translated = translator.translate(text, src=detected.lang, dest='en')
                return translated.text, None
        except Exception as e:
            return text, f"Erro ao traduzir o texto: {str(e)}"
    return text, None

def detect_emotions_with_weights(text: str):
    """
    Detects multiple emotions in the text and assigns a weight to each based on keyword frequency and VADER compound score.
    Returns a dict: emotion -> weight (normalized to sum to 1).
    """
    text_lower = text.lower()
    emotion_scores = {emotion: 0.0 for emotion in EMOTION_KEYWORDS}  # Use float for all
    # Keyword-based scoring
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                emotion_scores[emotion] += 1.0
    # VADER sentiment
    vader_scores = analyzer.polarity_scores(text)
    # Map VADER compound to emotion weights
    if vader_scores['compound'] > 0.5:
        emotion_scores['happy'] += float(vader_scores['compound'])
        emotion_scores['excited'] += float(vader_scores['pos'])
    elif vader_scores['compound'] < -0.5:
        emotion_scores['sad'] += abs(float(vader_scores['compound']))
        emotion_scores['angry'] += float(vader_scores['neg'])
    elif abs(vader_scores['compound']) < 0.2:
        emotion_scores['neutral'] += 1.0
    # Normalize
    total = sum(emotion_scores.values())
    if total == 0:
        return {'neutral': 1.0}
    return {emo: float(score)/total for emo, score in emotion_scores.items() if score > 0}

def detect_emotions_multi_label(user_input: str) -> Tuple[dict, Optional[str], Optional[str]]:
    """
    Detects multiple emotions using both TextBlob and VADER, with pre-processing and translation.
    Returns (emotion_weights, clarification_message, translation_error_message)
    """
    processed_text, translation_error = preprocess_user_input(user_input)
    if not processed_text:
        return {"neutral": 1.0}, "Por favor, descreva como você está se sentindo para receber recomendações.", translation_error
    # VADER + keyword (existing logic)
    vader_weights = detect_emotions_with_weights(processed_text)
    # TextBlob polarity/subjectivity
    tb = TextBlob(processed_text)
    polarity = getattr(tb.sentiment, 'polarity', 0)
    subjectivity = getattr(tb.sentiment, 'subjectivity', 0)
    # Map TextBlob to emotion
    tb_emotion = None
    if subjectivity < 0.3:
        tb_emotion = "neutral"
    elif polarity > 0.5 and subjectivity > 0.5:
        tb_emotion = "excited"
    elif polarity > 0.2:
        tb_emotion = "happy"
    elif polarity < -0.5 and subjectivity > 0.5:
        tb_emotion = "angry"
    elif polarity < -0.2:
        tb_emotion = "sad"
    elif subjectivity > 0.7:
        tb_emotion = "surprise"
    else:
        tb_emotion = "neutral"
    # Combine: average weights, boost if both agree
    combined = vader_weights.copy()
    if tb_emotion in combined:
        combined[tb_emotion] += 0.5
    else:
        combined[tb_emotion] = 0.5
    # Normalize
    total = sum(combined.values())
    if total == 0:
        combined = {"neutral": 1.0}
    else:
        combined = {k: float(v)/total for k, v in combined.items() if v > 0}
    # Ambiguity/clarification logic
    if len(combined) == 1 and list(combined.keys())[0] == "neutral":
        clarification = "Não consegui identificar claramente suas emoções. Poderia descrever melhor como você está se sentindo?"
    elif max(combined.values()) < 0.5:
        clarification = "Sua resposta está um pouco ambígua. Poderia detalhar mais como você se sente?"
    else:
        clarification = None
    return combined, clarification, translation_error

def explain_emotion_recommendation(emotion_weights: dict) -> str:
    """
    Gera uma explicação em PT-BR para o usuário sobre como as emoções detectadas influenciaram as recomendações.
    """
    if not emotion_weights or (len(emotion_weights) == 1 and 'neutral' in emotion_weights):
        return "Não foi possível identificar claramente uma emoção. Por favor, descreva melhor como você está se sentindo para recomendações mais personalizadas."
    top_emotions = sorted(emotion_weights.items(), key=lambda x: x[1], reverse=True)[:2]
    emotion_map_pt = {
        'happy': 'feliz', 'sad': 'triste', 'excited': 'animado', 'calm': 'calmo',
        'fear': 'com medo', 'angry': 'com raiva', 'surprise': 'surpreso', 'bored': 'entediado', 'neutral': 'neutro'
    }
    genres = []
    for emo, w in top_emotions:
        genres += EMOTION_GENRE_MAP.get(emo, [])
    genres = list(set(genres))
    emostr = ' e '.join([f"{emotion_map_pt.get(emo, emo)} ({w:.0%})" for emo, w in top_emotions])
    genstr = ', '.join(genres)
    return f"Detectamos que você está {emostr}. Por isso, priorizamos filmes dos gêneros: {genstr}."

def recommend_movies_based_on_emotion(movies_df: pd.DataFrame, emotion: str, n: int = 10) -> pd.DataFrame:
    genres = EMOTION_GENRE_MAP.get(emotion, EMOTION_GENRE_MAP["neutral"])
    def score_row(row):
        for g in genres:
            if g in row["Genres"]:
                return 2
        return 0
    movies_df = movies_df.copy()
    movies_df["EmotionScore"] = movies_df.apply(score_row, axis=1)
    recommendations = movies_df.sort_values(
        by=["EmotionScore", "Title"], ascending=[False, True]
    ).head(n)
    return recommendations

def load_emotion_genre_map(filepath=None):
    """
    Loads a many:many emotion-to-genre mapping (with weights) from a text file.
    Returns a dict: tuple(emotions) -> list of (genre, weight)
    """
    if filepath is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, 'emotion_genre_map.txt')
    mapping = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            emo_part, genres_part = line.split(':', 1)
            emotions = tuple(e.strip() for e in emo_part.split(','))
            genres = []
            for g in genres_part.split(','):
                if ':' in g:
                    genre, weight = g.split(':')
                    genres.append((genre.strip(), float(weight)))
                else:
                    genres.append((g.strip(), 1.0))
            mapping[emotions] = genres
    return mapping

EMOTION_GENRE_MANY_MAP = load_emotion_genre_map()

def parse_movie_genres(movies_df):
    """
    Adds a 'GenreSet' column to movies_df with set of genres for each movie.
    """
    movies_df = movies_df.copy()
    movies_df['GenreSet'] = movies_df['Genres'].apply(lambda g: set(g.split('|')))
    return movies_df

def get_genre_weights_for_emotions(emotions: list) -> dict:
    """
    Given a list of detected emotions, returns a dict of genre -> weight using the many:many mapping.
    """
    genre_weights = {}
    # Try exact match first
    key = tuple(sorted(emotions))
    if key in EMOTION_GENRE_MANY_MAP:
        for genre, weight in EMOTION_GENRE_MANY_MAP[key]:
            genre_weights[genre] = genre_weights.get(genre, 0) + weight
    else:
        # Fallback: sum all single-emotion mappings
        for emo in emotions:
            for k in EMOTION_GENRE_MANY_MAP:
                if len(k) == 1 and k[0] == emo:
                    for genre, weight in EMOTION_GENRE_MANY_MAP[k]:
                        genre_weights[genre] = genre_weights.get(genre, 0) + weight
    return genre_weights

def recommend_movies_multi_emotion(movies_df: pd.DataFrame, emotion_weights: dict, n: int = 10) -> pd.DataFrame:
    """
    Recommend movies by boosting genres according to weighted emotions and many:many mapping.
    """
    movies_df = parse_movie_genres(movies_df)
    # Get top emotions (by weight)
    sorted_emotions = sorted(emotion_weights.items(), key=lambda x: x[1], reverse=True)
    top_emotions = [e for e, w in sorted_emotions if w > 0][:2]
    genre_weights = get_genre_weights_for_emotions(top_emotions)
    def score_row(row):
        score = 0
        for genre, weight in genre_weights.items():
            # Exact match or partial match
            if genre in row['GenreSet'] or any(genre in g for g in row['GenreSet']):
                score += weight
            # Composite genre match (e.g., Comedy|Romance)
            if '|' in genre and set(genre.split('|')).issubset(row['GenreSet']):
                score += weight * 1.2
        return score
    movies_df['EmotionScore'] = movies_df.apply(score_row, axis=1)
    recommendations = movies_df.sort_values(
        by=['EmotionScore', 'Title'], ascending=[False, True]
    ).head(n)
    return recommendations

def ask_user_to_adjust_emotion_genre(emotion_weights, genre_weights):
    """
    Interactively let the user adjust detected emotions and genres before final recommendation.
    """
    print("\nDetectamos as seguintes emoções e pesos:")
    for emo, w in emotion_weights.items():
        print(f"  - {emo}: {w:.0%}")
    print("Gêneros sugeridos (com pesos):")
    for genre, w in genre_weights.items():
        print(f"  - {genre}: {w:.2f}")
    adjust = input("Deseja ajustar as emoções ou gêneros detectados? (s/n): ").strip().lower()
    if adjust == 's':
        new_emos = input("Digite as emoções desejadas separadas por vírgula (ou pressione Enter para manter): ").strip()
        if new_emos:
            emos = [e.strip() for e in new_emos.split(',') if e.strip()]
            # Recompute genre_weights
            from sentiment_recommendation import get_genre_weights_for_emotions
            genre_weights = get_genre_weights_for_emotions(emos)
            emotion_weights = {e: 1/len(emos) for e in emos} if emos else emotion_weights
        new_genres = input("Digite os gêneros desejados separados por vírgula (ou pressione Enter para manter): ").strip()
        if new_genres:
            genres = [g.strip() for g in new_genres.split(',') if g.strip()]
            # Set all selected genres to max weight
            genre_weights = {g: 1.0 for g in genres}
    return emotion_weights, genre_weights
