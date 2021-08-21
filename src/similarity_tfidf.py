from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize


def tfidf_vect(sentence1, sentence2):
    """Vectorizing sentences tfidf way.

    Parameters
    ----------
    sentence1 : [list]
        [a list of cleaned sentence]
    sentence2 : [list]
        [a list of cleaned sentence]

    Returns
    -------
    [sparce matrix]
        [Returns the pairwise similarity.]
    """
    sent1 = " ".join([str(elem) for elem in sentence1])
    sent2 = " ".join([str(elem) for elem in sentence2])
    document = [sent1, sent2]
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(document)

    pairwise_similarity = tfidf * tfidf.T
    return pairwise_similarity[0]
