from textblob import TextBlob

from googletrans import Translator
from textblob import Word


def analyze_sentiment(text="I love this movie! It's fantastic and inspiring."):
    """
    Analyze the sentiment of a given text.
    """

    blob = TextBlob(text)

    # Sentiment analysis
    print("Polarity:", blob.sentiment.polarity)  # Range: [-1.0, 1.0]
    print("Subjectivity:", blob.sentiment.subjectivity)  # Range: [0.0, 1.0]


def translate_text(text="Je suis très heureux aujourd'hui.", src="fr", dest="en"):
    translator = Translator()

    # Detect language
    detected_language = translator.detect(text)
    print("Detected Language:", detected_language.lang)

    # Translate to English
    translation = translator.translate(text, src, dest)
    print("Translated Text:", translation.text)


def tokenize_text(text="The quick brown fox jumps over the lazy dog."):
    blob = TextBlob(text)

    # Tokenization
    print("Words:", blob.words)
    print("Sentences:", blob.sentences)

    # Part-of-speech tagging
    print("POS Tags:", blob.tags)


def correct_spelling(text="I am hapy with the servce I recived."):

    blob = TextBlob(text)

    # Correct spelling
    corrected_text = blob.correct()
    print("Corrected Text:", corrected_text)


def word_inflection():
    word = Word("running")

    # Lemmatization
    print("Lemmatized:", word.lemmatize("v"))  # Verb form

    # Pluralization
    print("Plural:", Word("cat").pluralize())

    # Singularization
    print("Singular:", Word("dogs").singularize())


def sentence_analysis():
    text = """
    I love this movie. The acting was fantastic. However, the plot was a bit predictable.
    """
    blob = TextBlob(text)

    # Analyze sentiment for each sentence
    for sentence in blob.sentences:
        print(
            f"Sentence: {sentence.strip()} | Sentiment Polarity: {sentence.sentiment.polarity}"
        )


analyze_sentiment()
translate_text()
tokenize_text()
correct_spelling()
word_inflection()
sentence_analysis()
