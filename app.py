from src.sentences_similar_users import *

df = main()
user_description = input("Enter your description to find the matching: ")
clean_input = clean_input(user_description)
score = similarities(df, clean_input)
id_rank = rank_score(score)

user_list = top_similar(df, id_rank)

print('\nThe top similar users are:')
for user in user_list:
    print(user)

