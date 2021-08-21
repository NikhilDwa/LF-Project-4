from sentence_transformers import SentenceTransformer, util


def jaccard_similarity(list1, list2):
    """Calculates the jaccard similarity of two lists.

    Parameters
    ----------
    list1 : [list]
        [a list containing a sentence]
    list2 : [list]
        [a list containing a sentence]

    Returns
    -------
    [float]
        [returns the similarity score nin float]
    """
    intersection = len(set(list1).intersection(list2))
    union = len(set(list1)) + len(set(list2)) - intersection

    return intersection / union
