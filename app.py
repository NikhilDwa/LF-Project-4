from src.sentence_clean import *
from src.similarity_tfidf import *
from src.similarity_jaccard import *
from src.similarity_hugging_face import *


first_text = input("Enter your first text: ")
second_text = input("Enter your second text: ")

# Cleaning text
clean_first_text = clean_text(first_text)
clean_second_text = clean_text(second_text)


# Similarity using hugging face.
print("\nUsing hugging similarity, the similarity score between the text:")
hugging_similarity = similar_check(clean_first_text, clean_second_text)


# Pairwise similarity with tfidf vectorization.
tfidf_way = tfidf_vector(clean_first_text, clean_second_text)
print(f"\nPairwise similarity by tfidf:\n {tfidf_way}")


# Jaccard Similarity.
jaccard = jaccard_similarity(clean_first_text, clean_second_text)
print(f"\nJaccard similarity: {jaccard}")
