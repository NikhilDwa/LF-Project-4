# Sentence Similarity

This is program to check out the similarity between two set of sentences. 

## How to use?

Run the app.py file and it will ask you to input two sentences.

```bash
python run app.py
```

## Usage

Enter the first and the second text.
```bash
Enter your first text. it is a sunny day today
Enter your second text. it is a sunny day today
```

### Analyzing the result
```bash
 Using hugging similarity, the similarity score between the text:
['day today sunny']      ['day today sunny']     Score: 1.000000238418579
```
The words inside the [] brackets are the words used to calculate the score. 

Score is the similarity score with 1 being the highest similarity.

```bash 
Pairwise similarity by tfidf:
  (0, 1)       1.0000000000000002
  (0, 0)        1.0000000000000002

jaccard similarity: 1.0
```
The second half shows the pairwise similarity score of the first(0) and the second(1) sentence.
(0,0) is the similarity score with the same sentence.

Finally, the jaccard similarity is also calculated.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors
[Yugant] & [Nikhil]
