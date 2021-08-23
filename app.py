from sentence_transformers import SentenceTransformer, util

from src import clean_text, tfidf_vector, similar_check, jaccard_similarity


def model():
    sentence_transformer_model = SentenceTransformer("paraphrase-MiniLM-L12-v2")
    return sentence_transformer_model


# Input two text to calculate similarity scores.
first_text = input("Enter your first text: ")
second_text = input("Enter your second text: ")

# Cleaning text
clean_first_text = clean_text(first_text)
clean_second_text = clean_text(second_text)

# Similarity using hugging face, tfidf vectorization and jaccard.
hugging_similarity = similar_check(model(), clean_first_text, clean_second_text)
tfidf_way = tfidf_vector(clean_first_text, clean_second_text)
jaccard = jaccard_similarity(clean_first_text, clean_second_text)

# Output all the calculated scores.
print(f"\n Hugging face similarity: {hugging_similarity}")
print(f"\n Pairwise similarity by tfidf :\n {tfidf_way}")
print(f"\n Jaccard similarity: {jaccard}")
