import numpy as np
import matplotlib.pyplot as plt

import string
import random
import re
import requests
import os
import textwrap

np.random.seed(22)
random.seed(22)
### Create a substitution cipher
# One will act as the key, other as the value
# Create two lists of acii_lowercase string so that every element in the list is a lower case character
letters1 = list(string.ascii_lowercase)
letters2 = list(string.ascii_lowercase)
# Create a empty dictionary called true_mapping
true_mapping = {}
# Shuffle the second list of letters
random.shuffle(letters2)
# populate the map
for k, v in zip(letters1, letters2):
    true_mapping[k] = v
### The language model
# Initialize Markov matrix (a 26x26 matrix which full of 1)
M = np.ones((26, 26))
# Initial state distribution (a 26x1 matrix which full of 0)
pi = np.zeros(26)
# A function to update the Markov matrix. ch1: starting character, ch2: ending character
def update_transition(ch1, ch2):
    # ord('a') = 97, ord('b') = 98 so minus 97 when doing the operation
    i = ord(ch1) - 97
    j = ord(ch2) - 97

    M[i, j] += 1

# A function to update the initial state distribution
def update_pi(ch):
    i = ord(ch) - 97

    pi[i] += 1

# Get the log-probability of a word / token
def get_word_prob(word):
    # print('word:', word)
    i = ord(word[0]) - 97
    logp = np.log(pi[i]) # log probability of the first unigram

    for ch in word[1:]:
        j = ord(ch) - 97
        logp += np.log(M[i, j]) # log probability of following bigrams
        i = j

    return logp

# Get the probability of a sequence of words
def get_sequence_prob(words):
    # If input is a string, split into an array of tokens
    if type(words) == str:
        words = words.split()
    # Initialize the logp to 0
    logp = 0
    # Add the log probability of each word to logp then return logp
    for token in words:
        logp += get_word_prob(token)

    return logp

### Create a markov model based on an English dataset
# https://www.gutenberg.org/ebooks/2701
# Download the file
if not os.path.exists('Data/moby_dick.txt'):
    print("Downloading moby dick...")
    r = requests.get('https://lazyprogrammer.me/course_files/moby_dick.txt')
    if r.status_code == 200:
        with open('Data/moby_dick.txt', 'w', encoding="utf-8") as f:
            f.write(r.content.decode())

# For replacing non-alpha characters
regex = re.compile('[^a-zA-Z]')

# Load in words
with open('Data/moby_dick.txt', encoding="utf-8") as f:
    for line in f:
        line = line.rstrip() # Strip out any whitespace

        # There are blank lines in the file
        if line:
            line = regex.sub(' ', line) # Replace all non-alpha characters with space. If the line is empty it will fail.

            # Split the tokens in the line and lowercase
            tokens = line.lower().split()
            # Loop through all token in the tokens
            for token in tokens:
                # Update the model using update_pi and update_transition
                # First letter
                ch0 = token[0]
                update_pi(ch0)
                # Other letters
                for ch1 in token[1:]:
                    update_transition(ch0, ch1)
                    ch0 = ch1
    # Normalize the probabilities
    pi /= pi.sum()
    M /= M.sum(axis=1, keepdims=True)

### Encode a message
# this is a random excerpt from Project Gutenberg's
# The Adventures of Sherlock Holmes, by Arthur Conan Doyle
# https://www.gutenberg.org/ebooks/1661

original_message = '''I then lounged down the street and found,
as I expected, that there was a mews in a lane which runs down
by one wall of the garden. I lent the ostlers a hand in rubbing
down their horses, and received in exchange twopence, a glass of
half-and-half, two fills of shag tobacco, and as much information
as I could desire about Miss Adler, to say nothing of half a dozen
other people in the neighbourhood in whom I was not in the least
interested, but whose biographies I was compelled to listen to.'''

# Away they went, and I was just wondering whether I should not do well
# to follow them when up the lane came a neat little landau, the coachman
# with his coat only half-buttoned, and his tie under his ear, while all
# the tags of his harness were sticking out of the buckles. It hadn't
# pulled up before she shot out of the hall door and into it. I only
# caught a glimpse of her at the moment, but she was a lovely woman, with
# a face that a man might die for.

# My cabby drove fast. I don't think I ever drove faster, but the others
# were there before us. The cab and the landau with their steaming horses
# were in front of the door when I arrived. I paid the man and hurried
# into the church. There was not a soul there save the two whom I had
# followed and a surpliced clergyman, who seemed to be expostulating with
# them. They were all three standing in a knot in front of the altar. I
# lounged up the side aisle like any other idler who has dropped into a
# church. Suddenly, to my surprise, the three at the altar faced round to
# me, and Godfrey Norton came running as hard as he could towards me.

# A function to encode a message
def encode_message(msg):
    # Message to lower case
    msg = msg.lower()
    # Replace non-alpha characters
    msg = regex.sub(' ', msg)
    # Make the encoded message
    # Create an empty list called coded_msg
    coded_msg = []
    # Loop through the message
    for ch in msg:
        coded_ch = ch # Could just be a space
        # Assign the true_mapping value to the coded_ch if the ch is in true_mapping
        if ch in true_mapping:
            coded_ch = true_mapping[ch]
        # Append the coded_ch to the coded_msg
        coded_msg.append(coded_ch)

    return ''.join(coded_msg)

# Use encode_message function to get the encoded_message of original_message
encoded_message = encode_message(original_message)
# A function to decode a message
def decode_message(msg, word_map):
    # Create an empty list called decode_msg
    decode_msg = []
    # Loop through the msg
    for ch in msg:
        # Could just be a space
        decode_ch = ch
        # Assign the word_map value to the decode_ch if the ch is in word_map
        if ch in word_map:
            decode_ch = word_map[ch]
        # Append the decoded_ch to the decoded_msg
        decode_msg.append(decode_ch)

    return ''.join(decode_msg)

### Run an evolutionary algorithm to decode the message
# Initialization point
dna_pool = []
for _ in range(20):
    dna = list(string.ascii_lowercase)
    random.shuffle(dna)
    dna_pool.append(dna)

def evolve_offspring(dna_pool, n_children):
    # Make n_children per offspring. Create an empty list called offspring.
    offspring = []
    # Loop through all dna in the dna pool
    for dna in dna_pool:
        for _ in range(n_children):
            copy = dna.copy()
            # Pick two random indexes in range of length copy
            i, j = 0, 0
            while i == j:
                i = np.random.randint(len(copy))
                j = np.random.randint(len(copy))
            # Switch the dna sequences in these two indexes
            temp = copy[i]
            copy[i] = copy[j]
            copy[j] = temp
            # Append the copy to the offspring list
            offspring.append(copy)

    # Return offspring + dna_pool
    return offspring + dna_pool

num_iters = 1000
scores = np.zeros(num_iters)
best_dna = None
best_map = None
best_score = float('-inf')
for i in range(num_iters):
    # If i is greater than 0 then get offspring from the current dna pool through evolve_offspring function
    if i > 0:
        dna_pool = evolve_offspring(dna_pool, 3)
    # Calculate score for each dna
    # Create an empty dictionary called dna2score
    dna2score = {}
    for dna in dna_pool:
        # Create an empty dictionary called current_map
        current_map = {}
        # Add all elements of letters1 and dna to the current_map dictionary as keys and values
        for k, v in zip(letters1, dna):
            current_map[k] = v
        # Decode the message using decode_message function and pass the result to the variable decoded_message
        decoded_message = decode_message(encoded_message, current_map)
        # Use get_sequence_prob to get the score of the decoded_message
        score = get_sequence_prob(decoded_message)
        # Store it to the dna2score
        # The dict key needs to be a string so join the dna to an empty string
        dna2score["".join(dna)] = score
        # Record the best so far
        # If the current score is better than the best score then update best_dna, best_map and best_score
        if score > best_score:
            best_score = score
            best_dna = dna
            best_map = current_map
    # Calculate the average score for the ith generation
    scores[i] = np.mean(list(dna2score.values()))
    # Sort the dna by their scores in reversed order and keep the best 5 dna also turn them back into list of single chars
    sorted_dna = sorted(dna2score.items(), key=lambda x:x[1], reverse=True)
    dna_pool = [list(k) for k, v in sorted_dna[:5]]

    if i % 200 == 0:
        print(f"iter:{i} score:{scores[i]} best so far:{best_score}")

# Use the best map to decode the encoded_message
decoded_message = decode_message(encoded_message, best_map)
print("LL of decoded message:", get_sequence_prob(decoded_message))
print("LL of true message:", get_sequence_prob(regex.sub(' ', original_message.lower())))

# Print the wrong letters
for true, v in true_mapping.items():
    pred = best_map[v]
    if true != pred:
        print(f"true: {true}, pred: {pred}")

# Print the final decoded message
print("Decoded message:\n", textwrap.fill(decoded_message))
print("\nTrue message:\n", original_message)

plt.plot(scores)
plt.show()