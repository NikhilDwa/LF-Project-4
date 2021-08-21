from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize


def tfidf_vector(sentence_1, sentence_2):
    """Vectorization of sentences in tfidf way.

    Parameters
    ----------
    sentence_1 : [list]
        [a list of cleaned sentence]
    sentence_2 : [list]
        [a list of cleaned sentence]

    Returns
    -------
    [sparse matrix]
        [Returns the pairwise similarity.]
    """

    sent_1 = " ".join([str(elem) for elem in sentence_1])
    sent_2 = " ".join([str(elem) for elem in sentence_2])
    document = [sent_1, sent_2]
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(document)
    pairwise_similarity = tfidf * tfidf.T

    return pairwise_similarity[0]
