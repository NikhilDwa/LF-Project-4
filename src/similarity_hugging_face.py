from sentence_transformers import SentenceTransformer, util


def similar_check(sentence1, sentence2):
    """Function that checks the similarity of sentences.

    Parameters
    ----------
    sentences : [type: string]
        [description: a string of sentences seperated by full stop.]
    """

    model = SentenceTransformer("paraphrase-MiniLM-L12-v2")
    embeddings1 = model.encode(sentence1, convert_to_numpy=True)
    embeddings2 = model.encode(sentence2, convert_to_numpy=True)

    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    return print(f"{sentence1} \t\n {sentence2} \t Score: {cosine_scores.item()}")
