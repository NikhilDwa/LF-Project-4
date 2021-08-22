from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_vector(sentence_1, sentence_2):
    """Vectorization of sentences in tfidf way.

    Parameters
    ------------------------
    sentence_1 : [list]
        [a list of cleaned sentence]
    sentence_2 : [list]
        [a list of cleaned sentence]

    Returns
    ------------------------
    [sparse matrix]
        [Returns the pairwise similarity.]
    """
    sent_1 = " ".join([str(element) for element in sentence_1])
    sent_2 = " ".join([str(element) for element in sentence_2])
    document = [sent_1, sent_2]
    try:
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(document)
        pairwise_similarity = (tfidf * tfidf.T)[0]
    except:
        pairwise_similarity = "No pairwise similarity."
    return pairwise_similarity
