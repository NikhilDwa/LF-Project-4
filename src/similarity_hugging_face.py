from sentence_transformers import SentenceTransformer, util


def similar_check(sentence_1, sentence_2):
    """Function that checks the similarity of sentences.

    Parameters
    ----------
    sentences : [type: string]
        [description: a string of sentences separated by full stop.]
        :param sentence_1:
        :param sentence_2:
    """

    model = SentenceTransformer("paraphrase-MiniLM-L12-v2")
    embeddings_1 = model.encode(sentence_1, convert_to_numpy=True)
    embeddings_2 = model.encode(sentence_2, convert_to_numpy=True)

    cosine_scores = util.cos_sim(embeddings_1, embeddings_2)

    return print(f"{sentence_1} \t {sentence_2} \t Score: {cosine_scores.item()}")
