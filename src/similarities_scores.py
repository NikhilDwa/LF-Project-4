from sentence_transformers import SentenceTransformer, utis


def cosine_similarities(text_1, text_2):

    text_1_embedding = model.encode(text_1, convert_to_numpy=True)
    text_2_embedding = model.encode(text_2, convert_to_numpy=True)
    cosine_scores = util.cos_sim(text_1_embedding, text_2_embedding)

    return cosine_scores
