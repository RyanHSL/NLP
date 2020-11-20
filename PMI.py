from __future__ import print_function, division
from future.utils import iteritems
from builtins import range

import os
import string
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances
from glob import glob
from datetime import datetime


# Input files
files = glob('Data/enwiki-preprocessed/enwiki*.txt')

def remove_punctuation(s: str):
    return s.translate(str.maketrans('', '', string.punctuation))

# Max vocab size
V = 2000

# Context size
context_size = 10

# Word counts
all_word_counts = {}

# Get the top V words
num_lines = 0
num_tokens = 0
for f in files:
    for line in open(f, encoding='utf-8'):
        if line and line[0] not in ('[', '*', '-', '|', '=', '{', '}'):
            num_lines += 1
            for word in remove_punctuation(line).lower().split():
                num_tokens += 1
                all_word_counts[word] = all_word_counts.get(word, 0) + 1

print("number of lines: ", num_lines)
print("number of tokens: ", num_tokens)

# The keep words
keep_words = [
    'king', 'man', 'queen', 'woman',
    'heir', 'heiress', 'prince', 'princess',
    'nephew', 'niece', 'uncle', 'aunt',
    'husband', 'wife', 'brother', 'sister',
    'tokyo', 'beijing', 'dallas', 'texas',
    'january', 'february', 'march',
    'april', 'may', 'june',
    'july', 'august', 'september',
    'october', 'november', 'december',
    'actor', 'actress',
    'rice', 'bread', 'miami', 'florida',
    'walk', 'walking', 'swim', 'swimming',
]
for w in keep_words:
    all_word_counts[w] = float('inf')

# Get appropriate vocab size
V = min(len(all_word_counts), V)

# Sort in descending order
all_word_counts = sorted(all_word_counts.items(), key=lambda x:x[1], reverse=True)

# Keep the top V words
# Save a slot for <UNK>
top_words = [k for k, v in all_word_counts[:V - 1]] + ['<UNK>']

# Reverse the array to get word2idx mapping
word2idx = {w:i for i, w in enumerate(top_words)}
unk = word2idx['<UNK>']

if not os.path.exists("pmi_counts_%s.npz" %V):
    # Init counts
    wc_counts = lil_matrix((V, V))

    ### Make PMI matrix
    # Add counts
    k = 0
    # Create a line_as_idx list
    for f in files:
        for line in open(f, encoding='utf-8'):
            if line and line[0] not in ('[', '*', '-', '|', '=', '{', '}'):
                line_as_idx = []
                for word in remove_punctuation(line).lower().split():
                    idx = word2idx.get(word, unk)
                    line_as_idx.append(idx)

                for i, w in enumerate(line_as_idx):
                    # Keep count
                    k += 1
                    if k % 10000 == 0:
                        print(f"{k}/{num_tokens}")

                    start = max(0, i - context_size)
                    end = min(len(line_as_idx), i + context_size)
                    for c in line_as_idx[start:i]:
                        wc_counts[w, c] += 1
                    for c in line_as_idx[i+1:end]:
                        wc_counts[w, c] += 1
    print("Finished counting")

    save_npz('pmi_counts_%s.npz' %V, csr_matrix(wc_counts))
else:
    wc_counts = load_npz('pmi_counts_%s.npz' %V)

# Smooth the context counts
c_counts = wc_counts.sum(axis=0).A.flatten()**0.75
c_probs = c_counts / c_counts.sum()
c_probs = c_probs.reshape(1, V)

pmi = wc_counts.multiply(1.0 / wc_counts.sum(axis=1) / c_probs).tocsr()
# This operation changes it to a coo_matrix which doesn't have functions I need, e.g log1p()
# So convert it back to a csr
print("type(pmi): ", type(pmi))
logX = pmi.log1p() # Would be logX = np.log(pmi.A + 1) in numpy
print("type(logX): ", type(logX))
logX[logX < 0] = 0

### Do alternating least squares

# Latent dimension
D = 100
reg = 0.1 # Regularizer

# Initialize weights
W = np.random.randn(V, D) / np.sqrt(V + D)
b = np.zeros(V)
U = np.random.randn(V, D) / np.sqrt(V + D)
c = np.zeros(V)
mu = logX.mean()

costs = []
t0 = datetime.now()
for epoch in range(10):
    print("epoch: ", epoch)
    delta = W.dot(U.T) + b.reshape(V, 1) + c.reshape(1, V) + mu - logX
    cost = np.multiply(delta, delta).sum()
    costs.append(cost)

    ### Partially vectorized updates ###
    # # Update W
    # matrix = reg * np.eye(D) + U.T.dot(U)
    # for i in range(V):
    #     vector = (logX[i, :] - b[i] - c - mu).dot(U)
    #     W[i] = np.linalg.solve(matrix, vector)
    #
    # # Update b
    # for i in range(V):
    #     numerator = (logX[i, :] - W[i].dot(U.T) - c - mu).sum()
    #     b[i] = numerator / V
    #
    # # Update U
    # matrix = reg * np.eye(D) + W.T.dot(W)
    # for j in range(V):
    #     vector = (logX[:, j] - b - c[j] - mu).dot(W)
    #     U[j] = np.linalg.solve(matrix, vector)
    #
    # # Update c
    # for j in range(V):
    #     numerator = (logX[:, j] - W.dot(U[j]) - b - mu).sum()
    #     c[j] = numerator / V

    ### Vectorized updates ###
    # Vectorized update W
    matrix = reg * np.eye(D) + U.T.dot(U)
    vector = (logX - b.reshape(V, 1) - c.reshape(1, V) - mu).dot(U).T
    W = np.linalg.solve(matrix, vector).T

    # Vectorized update b
    b = (logX - W.dot(U.T) - c.reshape(1, V) - mu).sum(axis=1) / V

    # Vectorized update U
    matrix = reg * np.eye(D) + W.T.dot(W)
    vector = (logX - b.reshape(V, 1) - c.reshape(1, V) - mu).T.dot(W).T
    U = np.linalg.solve(matrix, vector).T

    # Vectorized update c
    c = (logX - W.dot(U.T) - b.reshape(V, 1) - mu).sum(axis=0) / V

print("train duration: ", (datetime.now() - t0))

plt.plot(costs)
plt.show()

# Test it
king  = W[word2idx['king']]
man   = W[word2idx['man']]
queen = W[word2idx['queen']]
woman = W[word2idx['woman']]

vec = king - man + woman

# find closest
# closest = None
# min_dist = float('inf')
# for i in range(len(W)):
#   dist = cos_dist(W[i], vec)
#   if dist < min_dist:
#     closest = i
#     min_dist = dist

# set word embedding matrix
# W = (W + U) / 2

distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
idx = distances.argsort()[:10]

print("closest 10:")
for i in idx:
  print(top_words[i], distances[i])

print("dist to queen:", cos_dist(W[word2idx['queen']], vec))



def analogy(pos1, neg1, pos2, neg2):
  # don't actually use pos2 in calculation, just print what's expected
  print("testing: %s - %s = %s - %s" % (pos1, neg1, pos2, neg2))
  for w in (pos1, neg1, pos2, neg2):
    if w not in word2idx:
      print("Sorry, %s not in word2idx" % w)
      return

  p1 = W[word2idx[pos1]]
  n1 = W[word2idx[neg1]]
  p2 = W[word2idx[pos2]]
  n2 = W[word2idx[neg2]]

  vec = p1 - n1 + n2

  distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
  idx = distances.argsort()[:10]

  # pick the best that's not p1, n1, or n2
  best_idx = -1
  keep_out = [word2idx[w] for w in (pos1, neg1, neg2)]
  for i in idx:
    if i not in keep_out:
      best_idx = i
      break

  print("got: %s - %s = %s - %s" % (pos1, neg1, top_words[best_idx], neg2))
  print("closest 10:")
  for i in idx:
    print(top_words[i], distances[i])

  print("dist to %s:" % pos2, cos_dist(p2, vec))


analogy('king', 'man', 'queen', 'woman')
analogy('miami', 'florida', 'dallas', 'texas')
# analogy('einstein', 'scientist', 'picasso', 'painter')
analogy('china', 'rice', 'england', 'bread')
analogy('man', 'woman', 'he', 'she')
analogy('man', 'woman', 'uncle', 'aunt')
analogy('man', 'woman', 'brother', 'sister')
analogy('man', 'woman', 'husband', 'wife')
analogy('man', 'woman', 'actor', 'actress')
analogy('man', 'woman', 'father', 'mother')
analogy('heir', 'heiress', 'prince', 'princess')
analogy('nephew', 'niece', 'uncle', 'aunt')
analogy('france', 'paris', 'japan', 'tokyo')
analogy('france', 'paris', 'china', 'beijing')
analogy('february', 'january', 'december', 'november')
analogy('france', 'paris', 'italy', 'rome')
analogy('paris', 'france', 'rome', 'italy')
analogy('france', 'french', 'england', 'english')
analogy('japan', 'japanese', 'china', 'chinese')
analogy('japan', 'japanese', 'italy', 'italian')
analogy('japan', 'japanese', 'australia', 'australian')
analogy('walk', 'walking', 'swim', 'swimming')