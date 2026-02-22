"""
text_cleaner.py
---------------
Generic text cleaning functions reusable across all datasets.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def remove_brackets(text: str) -> str:
    """
    Remove ['...'] artifacts found in CNH-PSX headlines.
    e.g. "['KSE index plunges by 83 points']" -> "KSE index plunges by 83 points"
    """
    text = re.sub(r"^\[\'", "", text)
    text = re.sub(r"\'\]$", "", text)
    return text.strip()


def remove_punctuation(text: str) -> str:
    """Remove all punctuation from text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_extra_whitespace(text: str) -> str:
    """Remove extra whitespace and strip."""
    return re.sub(r'\s+', ' ', text).strip()


def remove_stopwords(text: str) -> str:
    """Remove English stopwords from text."""
    tokens = text.split()
    return " ".join([w for w in tokens if w.lower() not in STOP_WORDS])


def lemmatize(text: str) -> str:
    """Lemmatize each word in text."""
    tokens = text.split()
    return " ".join([LEMMATIZER.lemmatize(w) for w in tokens])


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def filter_short_texts(text: str, min_words: int = 3) -> bool:
    """Return True if text has at least min_words words."""
    return len(text.split()) >= min_words


def clean_text(text: str, remove_stops: bool = True, do_lemmatize: bool = True) -> str:
    """
    Full cleaning pipeline for a single text string.
    Steps: brackets -> lowercase -> punctuation -> whitespace -> stopwords -> lemmatize
    
    Args:
        text: Raw input text
        remove_stops: Whether to remove stopwords (default True)
        do_lemmatize: Whether to lemmatize (default True)
    
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = remove_brackets(text)
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_extra_whitespace(text)
    
    if remove_stops:
        text = remove_stopwords(text)
    
    if do_lemmatize:
        text = lemmatize(text)
    
    return text
