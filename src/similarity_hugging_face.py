from sentence_transformers import util


def similar_check(model, sentence_1, sentence_2):
    """Function that checks the similarity of sentences.

    Parameters
    ------------------------
    sentences : [type: string]
        [description: a string of sentences separated by full stop.]
        :param model: sentence transformer model
        :param sentence_1: sentence one with no stopwords
        :param sentence_2: sentence two with no stopwords
    """

    embeddings_1 = model.encode(sentence_1, convert_to_numpy=True)
    embeddings_2 = model.encode(sentence_2, convert_to_numpy=True)

    cosine_scores = util.cos_sim(embeddings_1, embeddings_2)
    round_off_cosine_scores = "{:0.2f}".format(cosine_scores.item())

    return round_off_cosine_scores
