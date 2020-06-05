from sklearn.feature_extraction.text import CountVectorizer  # freq based

corpus = [
    "This is the first document",
    "This is the second document",
    "This is the third document",
    "Number four. To repeat, number four",
]

# create and initialize the vectorizer
vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(corpus)
# display
# print(bag_of_words)

# display full vocabulary
# print(vectorizer.vocabulary_)

# different bag same words have same integer representation.
# most frequently used words are assigned lower ids

import pandas as pd

df = pd.DataFrame(bag_of_words.toarray(), columns=vectorizer.get_feature_names())
# rows are docs , columns are individual words
print(df)

# using a Tfidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()
bow = vec.fit_transform(corpus)
# every word gets a score, unique id

df = pd.DataFrame(bow.toarray(), columns=vec.get_feature_names())
# rows are docs , columns are individual words, cells are Tfidf scores
print(df)

# complete vocabulary
print(vec.vocabulary_)

# For large datasets we use HashingVectorizer

from sklearn.feature_extraction.text import HashingVectorizer

v = HashingVectorizer(n_features=8)  # no of hash buckets
feature_vector = v.fit_transform(corpus)
print(feature_vector)
