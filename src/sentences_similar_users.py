import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def main():

    df = pd.read_csv('F:/Leapfrog Technology/Project 4/data/profile.csv', header=0)
    clean_text = []
    for index in range(len(df['description'])):
        text = df['description'][index]
        text = text.lower()
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\d', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        words = text.lower().split()
        stop_word_set = set(stopwords.words("english"))
        cleaned_words = list(set([w for w in words if w not in stop_word_set]))
        cleaned_words = [' '.join(cleaned_words[::])]
        clean_text.append(cleaned_words[0])
    df.insert(3, "cleaned_words", clean_text, True)

    return df


def clean_input(input_text):

    process_text = input_text.lower()
    process_text = re.sub(r'\[[0-9]*\]', ' ', process_text)
    process_text = re.sub(r'\s+', ' ', process_text)
    process_text = re.sub(r'\d', ' ', process_text)
    process_text = re.sub(r'\s+', ' ', process_text)
    process_words = process_text.lower().split()
    stop_words = set(stopwords.words("english"))
    cleaned_input = list(set([w for w in process_words if w not in stop_words]))
    cleaned_input = [' '.join(cleaned_input[::])]

    return cleaned_input


def similarities(df, new_des):

    scores = {}
    for i in range(len(df['cleaned_words'])):
        new_des_embeddings = model.encode(new_des, convert_to_numpy=True)
        old_embeddings_another = model.encode(df['cleaned_words'][i], convert_to_numpy=True)
        cosine_scores = util.cos_sim(new_des_embeddings, old_embeddings_another)
        scores[i] = cosine_scores

    return scores


def rank_score(scores):

    sorted_score = sorted(scores, key=scores.get)
    rank = sorted_score[::-1]

    return rank


def top_similar(df, user_id):

    top_user = []
    for user in user_id:
        top_user.append(df['Name'][user])

    return top_user
