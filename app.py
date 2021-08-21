from src.sentence_clean import *
from src.similarity_tfidf import *
from src.similarities_scores import *
from src.similarity_hugging_face import *


first_text = input("Enter your first text: ")
second_text = input("Enter your second text: ")

# Cleaning text
clean_first_text = clean_text(first_text)
clean_second_text = clean_text(second_text)


# Similarity using hugging face.
hugging_similarity = similar_check(clean_first_text, clean_second_text)


# Pairwise similarity with tfidf vectorization.
tfidf_way = tfidf_vect(clean_first_text, clean_second_text)
print(f"Pairwise similarity by tfidf: {tfidf_way}")


# Jaccard Similarity.
jaccar = jaccard_similarity(clean_first_text, clean_second_text)
print(f"jaccard similarity: {jaccar}")
