import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
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
