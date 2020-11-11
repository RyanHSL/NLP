from __future__ import print_function, division
from future.utils import iteritems
from builtins import range

import nltk
import numpy as np
from sklearn.utils import shuffle

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

# Instantiate the WordNetLemmatizer object
wordnet_lemmatizer = WordNetLemmatizer()
# Create a stopwords set which uses stopwords.words('english')
# Or download from http://www.lextek.com/manuals/onix/stopwords1.html
stop_words = stopwords.words('english')
# Load the reviews using BeautifulSoup()
positive_reviews = BeautifulSoup(open('Data/domain_sentiment_data/sorted_data_acl/electronics/positive.review').read(), features="html.parser")
positive_reviews = positive_reviews.find_all('review_text')
negative_reviews = BeautifulSoup(open('Data/domain_sentiment_data/sorted_data_acl/electronics/negative.review').read(), features="html.parser")
negative_reviews = negative_reviews.find_all('review_text')
# First let me try to tokenize the text using nltk's tokenizer
# Let me take the first review for example:
# nltk.tokenize.word_tokenize(t.text)
#
# Notice how it doesn't down-case, so It != it
# Not only that, but do we really want to include the word "it" anyway?
# You can imagine it wouldn't be any more common in a positive review than a negative review
# So it might only add noise to the model.
# So let me create a function that does all this pre-processing for us

def my_tokenizer(s):
    # Lower-case the sentences
    s = s.lower()
    # Split string into words (tokens) using nltk.tokenize.word_tokenize()
    tokens = nltk.tokenize.word_tokenize(s)
    # Remove words that is shorter than or equal to 2, they are probably not useful
    tokens = [t for t in tokens if len(t) > 2]
    # Put words into base form
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]

    return tokens

# Create a word-to-index map so that I can create my word-frequency vectors later
# Let me also save the tokenized versions so we don't have to tokenizer again later
# Create a word_index_map dictionary
word_index_map = {}
# Set the current_index to 0
current_index = 0
# Create a positive_tokenized list
positive_tokenized = []
# Create a negative_tokenized list
negative_tokenized = []
# Create an orig_reviews list
orig_reviews = []
# Loop through all the positive_reviews
for review in positive_reviews:
    # Append the review text to the orig_reviews list
    orig_reviews.append(review.text)
    # Use my_tokenizer to get the tokens
    tokens = my_tokenizer(review.text)
    # Append the tokens to the positive_tokenized list
    positive_tokenized.append(tokens)
    # Loop through all token in the tokens
    for token in tokens:
        # Add the token to the word_index_map if the token is not in the map
        if token not in word_index_map:
            word_index_map[token] = current_index
        # Update the current_index
            current_index += 1
# do the same thing for negative_reviews
for review in negative_reviews:
    orig_reviews.append(review.text)
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

print("len(word_index_map):", len(word_index_map))

# Now let me create my input matrices
def tokens_to_vector(tokens, label):
    # Create a (D, 1) shape zeros-matrix
    x = np.zeros(len(word_index_map) + 1)
    # Loop through all token in tokens
    for t in tokens:
        # Get the index i of token t
        i = word_index_map[t]
        # Add 1 to the ith index of input matrix
        x[i] += 1

    x = x/x.sum()
    x[-1] = label

    return x

# Sum up the length of positive_tokenized and the negative_tokenized lists
N = len(positive_tokenized) + len(negative_tokenized)
# (N x D + 1 matrix - keeping them together for now so I can shuffle more easily later)
data = np.zeros((N, len(word_index_map) + 1))
# Create a variable i = 0
i = 0
# Loop through all tokens in the positive_tokenized
for tokens in positive_tokenized:
    # Use tokens_to_vector to create the variable xy for positive reviews
    xy = tokens_to_vector(tokens, 1)
    # Assign xy to data[i:]
    data[i:] = xy
    # Increase i by 1
    i += 1

# Do the same thing to negative_tokenized, but remember to change the label to 0
for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i:] = xy
    i += 1

# Shuffle the data and create train/test splits
# Try it multiple times
orig_reviews, data = shuffle(orig_reviews, data)

# Create x and y. y are the last element of each row in data
x, y = data[:, :-1], data[:, -1]
# Create x_train, x_test, y_train, y_test. Test data is the last 100 elements in x and y
x_train, x_test = x[:-100], x[-100:]
y_train, y_test = y[:-100], y[-100:]
# Create a model object then fit the training data and print the Train accuracy and the Test accuracy
model = LogisticRegression()
model.fit(x_train, y_train)
print("train score:", model.score(x_train, y_train))
print("test score:", model.score(x_test, y_test))
# Let me look at the weights for each word
# Try it with different threshold values
threshold = 0.5
# Loop through all word and index in the word_index_map
for word, index in iteritems(word_index_map):
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)

# Check the misclassified examples
preds = model.predict(x)
P = model.predict_proba(x)[:,1] # p(y = 1 | x)

# Since there are many, just print the most wrong samples
minP_whenYis1 = 1
maxP_whenYis0 = 0
wrong_positive_review = None
wrong_negative_review = None
wrong_positive_prediction = None
wrong_negative_prediction = None
for i in range(N):
    p = P[i]
    yt = y[i]
    if yt == 1 and p < 0.5:
        if p < minP_whenYis1:
            wrong_positive_review = orig_reviews[i]
            wrong_positive_prediction = preds[i]
            minP_whenYis1 = p
    elif yt ==0 and p > 0.5:
        if p > maxP_whenYis0:
            wrong_negative_review = orig_reviews[i]
            wrong_negative_prediction = preds[i]
            maxP_whenYis0 = p

print(f"Most wrong positive review (prob = {minP_whenYis1, wrong_positive_prediction})")
print(wrong_positive_review)
print(f"Most wrong negative review (prob = {maxP_whenYis0, wrong_negative_prediction})")
print(wrong_negative_review)