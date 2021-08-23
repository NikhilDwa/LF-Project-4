def jaccard_similarity(list1, list2):
    """Calculates the jaccard similarity of two lists.

    Parameters
    ------------------------
    list1 : [list]
        [a list containing a sentence]
    list2 : [list]
        [a list containing a sentence]

    Returns
    ------------------------
    [float]
        [returns the similarity score nin float]
    """

    intersection = len(set(list1).intersection(list2))
    union = len(set(list1)) + len(set(list2)) - intersection
    try:
        similarity_score = intersection / union
    except:
        similarity_score = "Similarity score can't be calculated."

    return similarity_score
