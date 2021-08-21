from sentence_transformers import SentenceTransformer, utis


def cosine_similarities(text_1, text_2):
    cosine_scores = util.cos_sim(text_1, text_2)
    return cosine_scores
