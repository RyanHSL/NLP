from __future__ import print_function, division
from builtins import range

import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD

# Instantiate a WordNetLemmatizer object
wordnet_lemmatizer = WordNetLemmatizer()
# Get all book titles
titles = [line.rstrip() for line in open('Data/all_book_titles.txt')]

# Get the stop words
stop_words = set(stopwords.words('english'))
# Add more stopwords specific to this problem
stop_words = stop_words.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth',
})

# Create a tokenizer function
def my_tokenizer(s):
    # Downcase the string
    s = s.lower()
    # Use nltk.tokenize.word_tokenize to split string into words (tokens)
    tokens = nltk.word_tokenize(s)
    # Remove the short words, they are probably not useful
    tokens = [t for t in tokens if len(t) > 2]
    # Put words into base form
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    # Remove stop words
    tokens = [t for t in tokens if t not in stop_words]
    # Remove any digits using c.isdigit(), i.e. "3rd edition"
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]

    return tokens

# Create a word-to-index map so that we can create our word-frequency vectors later
# Let me also save the tokenized versions so I do not have to tokenize again later
# Create an empty word_index_map
word_index_map = {}
# Initialize the current_index to 0
current_index = 0
# Create two empty lists all_tokens and all_titles
all_tokens, all_titles = [], []
# Create an empty index_word_map list
index_word_map = []
# Initialize the error_count to 0
error_count = 0
# Loop through all titles
for title in titles:
    # Since some of the titles have invalid Ascii, I am trying to encode them in Ascii
    # Skip if there is an exception
    try:
        # Encode the title in Ascii. This will throw exception if contains bad characters
        title = title.encode('ascii', 'ignore').decode('utf-8')
        # Append the encoded title to all_titles
        all_titles.append(title)
        # Use my_tokenizer to tokenize the title
        tokens = my_tokenizer(title)
        # Append the tokenized title to all_tokens
        all_tokens.append(tokens)
        # Loop through all token in tokens
        for token in tokens:
            # Add the token to the word_index_map if it is not in the word_index_map
            if token not in word_index_map:
                word_index_map[token] = current_index
                # Increase the current_index
                current_index += 1
            # Append the token to the index_word_map
            index_word_map.append(token)
    except Exception as e:
        print(e)
        print(title)
        error_count += 1

print(f"Number of errors parsing file: {error_count}, number of lines in file: {len(titles)}")
# Print "There is no data to do anything with! Quitting..." if the error_count equals the total length of titles
if error_count == len(titles):
    print("There is no data to do anything with! Quitting...")
    quit()
# Now let me create my input matrices - just indicator variables for this example - works better than proportions
def tokens_to_vectors(tokens):
    # Create an all-zero list x that has length of word_index_map. It is unsupervised learning so no labels.
    x = np.zeros(len(word_index_map))
    # Loop through all token in the tokens
    for t in tokens:
        # Get the index of t in the word_index_map
        i = word_index_map[t]
        # Pass 1 to the ith element in x
        x[i] = 1

    return x

N = len(all_tokens)
D = len(word_index_map)
# Create an all-zero D x N matrix called X. Terms will go along rows, documents along columns
X = np.zeros((D, N))
i = 0
# Loop through all tokens in all_tokens
for tokens in all_tokens:
    # Use tokens_to_vector to vectorize the tokens then assign it to the ith column of X
    X[:, i] = tokens_to_vectors(tokens)
    # Increase i by 1
    i += 1

def main():
    # Instantiate a svd object
    svd = TruncatedSVD()
    # Model fit_transform X to get Z
    Z = svd.fit_transform(X)
    # Scatter plot the first and the second column of Z
    plt.scatter(Z[:, 0], Z[:, 1])
    # Plot the LSA cloud
    for i in range(D):
        plt.annotate(text=index_word_map[i], xy=(Z[i,0], Z[i,1]))
    plt.show()

if __name__ == '__main__':
    main()