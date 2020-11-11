from __future__ import print_function, division
from future.utils import iteritems
from builtins import range

import nltk
import random
import numpy as np

from bs4 import BeautifulSoup

# Use BeautifulSoup to load the reviews
# (only load the positive reviews, since words in negative reviews have different meanings)
positive_reviews = BeautifulSoup(open('Data/domain_sentiment_data/sorted_data_acl/electronics/positive.review').read(), features="html.parser")
# Find all 'review_text' in positive_reviews
positive_reviews = positive_reviews.find_all("review_text")
# Extract trigrams and insert into dictionary
# Let the two surrounding words be the key of the dictionary
# and have all the possible middle words be the values
# Each middle word needs to be associated with a probability and so that itself is a nested dictionary
# Create an empty trigrams dictionary
trigrams = {}
# Loop through all reviews in positive_reviews
for review in positive_reviews:
    # Down case the review texts
    review = review.text.lower()
    # Get the tokens using word_tokenize
    tokens = nltk.word_tokenize(review)

    for i in range(3, len(tokens)): # I use the three-element trigram
        # Create a key tuple consisted of token[i] and token[i + 2]
        k = (tokens[i - 3], tokens[i - 2], tokens[i - 1])
        # Create an empty list for value at key k in trigram dictionary
        if k not in trigrams:
            trigrams[k] = []
        # Append the middle token to the trigram value list at key k
        trigrams[k].append(tokens[i])

# Turn each array of middle-words into a probability vector
# Create a trigram_probabilities dictionary
trigram_probabilities = {}
# Loop through all key and values in iteritems of trigrams
for k, words in iteritems(trigrams):
    # Create a dictionary of word -> count
    if len(set(words)) > 1:
        # Only do this when there are different possibilities for middle word
        # Create an empty dictionary called d
        d = {}
        # Initialize n to 0
        n = 0
        # Loop through all word in words
        for w in words:
            # Pass 0 to d[w] if w is not in d
            d[w] = d.get(w, 0) + 1
            # Increase d[w] by 1

            # Increase n by 1
            n = 1
        # Loop through all keys and values in iteritems of d
        for w, c in iteritems(d):
            # Calculate the probability of value at d[w]
            d[w] = float(c)/n
        # Add the k, d pair to the trigram_probabilities dictionary
        trigram_probabilities[k] = d
def random_sample(d):
    # Choose a random sample from dictionary where values are the probabilities
    r = random.random()
    # Initialize a variable called cumulative to 0
    cumulative = 0
    # Loop through all keys and values in iteritem of d
    for w, p in iteritems(d):
        # Add p to cumulative
        cumulative += p
        # Return w if r is smaller than cumulative
        if r < cumulative:
            return w
def test_spinner():
    # Create a random chosen review variable
    review = random.choice(positive_reviews)
    # Down case the text of the review
    s = review.text.lower()
    print(f"Original{s}")
    # Tokenize the words
    tokens = nltk.word_tokenize(s)
    for i in range(3, len(tokens)):
        if random.random() < 0.2: # 20% chance of replacement
            # Create a tuple called k which consisted of the first and the last word of a trigram
            k = (tokens[i - 3], tokens[i - 2], tokens[i - 1])
            # Pass a random_sample of trigram_probabilities values at key k to the middle word
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                tokens[i] = w
    print("Spun:")
    print(" ".join(tokens).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!"))

if __name__ == '__main__':
    test_spinner()