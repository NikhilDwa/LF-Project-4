from sentence_transformers import SentenceTransformer, util
def similar_check(sentences):
    """Function that checks the similarity of sentences.

    Parameters
    ----------
    sentences : [type: string]
        [description: a string of sentences seperated by full stop.]
    """
    sentence = sentences.split('.')

    model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
    embeddings = model.encode(
        sentence,convert_to_numpy = True)

    cosine_scores = util.cos_sim(embeddings,embeddings)
    pairs = []
    for i in range(len(cosine_scores)-1):
        for j in range(i+1,len(cosine_scores)):
            pairs.append(
                {'index' : [i,j], 'score' : cosine_scores[i][j] })

    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)


    for pair in pairs[0:10]:
        i, j = pair['index']
        return print("{} \t\t {} \t\t Score: {:.4f}".format(
            sentence[i], sentence[j], pair['score']))
    

    