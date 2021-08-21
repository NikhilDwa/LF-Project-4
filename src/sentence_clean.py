import re
import nltk
from nltk.corpus import stopwords


def clean_text(text):

    text = text.lower()
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.lower().split()
    stop_word_set = set(stopwords.words("english"))
    cleaned_words = list(set([w for w in words if w not in stop_word_set]))
    cleaned_words = [' '.join(cleaned_words[::])]

    return cleaned_words
