from __future__ import print_function, division
from builtins import range

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

from datetime import datetime
from sklearn.utils import shuffle
from utils import find_analogies
from rnn_class.utils import get_wikipedia_data
from rnn_class.brown import get_sentences_with_word2indx_limit_vocab, get_sentences_with_word2idx

class Glove_SVD:
    def __init__(self, D, V, context_sz):
        self.V = V
        self.D = D
        self.context_sz = context_sz

    def fit(self, sentences, cc_matrix=None):
        # Build the co-occurence matrix
        # Paper calls it X, so I will call it X instead of calling the training data X
        t0 = datetime.now()
        V = self.V
        D = self.D

        if not os.path.exists(cc_matrix):
            X = np.zeros((V, V))
            N = len(sentences)
            print("number of sentences to process: ", N)
            it = 0
            for sentence in sentences:
                it += 1
                if it % 10000 == 0:
                    print("processed", it, "/", N)
                n = len(sentence)
                for i in range(n):
                    wi = sentence[i]

                    start = max(0, i - self.context_sz)
                    end = min(n, i + self.context_sz)

                    if i - self.context_sz < 0:
                        point = 1.0 / (i + 1)
                        X[wi, 0] += point
                        X[0, wi] += point
                    if i + self.context_sz > n:
                        point += 1.0 / (n - i)
                        X[wi, 1] += point
                        X[1, wi] += point

                    # Left side
                    for j in range(start, i):
                        wj = sentence[j]
                        point = 1.0 / (i - j)
                        X[wi, wj] += point
                        X[wj, wi] += point
                    # Right side
                    for j in range(i + 1, end):
                        wj = sentence[j]
                        point = 1.0 / (j - i)
                        X[wi, wj] += point
                        X[wj, wi] += point

            # Save the cc matrix because it takes forever to create
            np.save(cc_matrix, X)
        else:
            X = np.load(cc_matrix)

        print("max in X: ", X.max())

        # Target
        logX = np.log(X + 1)

        print("max in log(X): ", logX.max())

        print("time to build co-occurence matrix: ", (datetime.now() - t0))

        # Substract global mean
        mu = logX.mean()

        model = TruncatedSVD(n_components=D)
        Z = model.fit_transform(logX - mu)
        S = np.diag(model.explained_variance_)
        Sinv = np.linalg.inv(S)
        self.W = Z.dot(Sinv)
        self.U = model.components_.T

        # Calculate cost once
        delta = self.W.dot(S).dot(self.U.T) + mu - logX
        cost = (delta * delta).sum()
        print("svd cost: ", cost)

    def save(self, fn):
        # Function word_analogies expects a (V, D) matrix and a (D, V) matrix
        arrays = [self.W, self.U.T]
        np.savez(fn, *arrays)

def main(we_file, w2i_file, use_brown=True, n_files=100):
    if use_brown:
        cc_matrix = "cc_matrix_brown.npy"
    else:
        cc_matrix = "cc_matrix_%s.npy" % n_files

    # hacky way of checking if we need to re-load the raw data or not
    # remember, only the co-occurrence matrix is needed for training
    if os.path.exists(cc_matrix):
        with open(w2i_file) as f:
            word2idx = json.load(f)
        sentences = []
    else:
        if use_brown:
            keep_words = set([
                'king', 'man', 'woman',
                'france', 'paris', 'london', 'rome', 'italy', 'britain', 'england',
                'french', 'english', 'japan', 'japanese', 'chinese', 'italian',
                'australia', 'australian', 'december', 'november', 'june',
                'january', 'february', 'march', 'april', 'may', 'july', 'august',
                'september', 'october',
            ])
            sentences, word2idx = get_sentences_with_word2indx_limit_vocab(n_vocab=5000, keep_words=keep_words)
        else:
            sentences, word2idx = get_wikipedia_data(n_files=n_files, n_vocab=2000)

        with open(w2i_file, 'w') as f:
            json.dump(word2idx, f)

    V = len(word2idx)
    model = Glove_SVD(100, V, 10)

    # Alternating least squares method
    model.fit(sentences, cc_matrix=cc_matrix)
    model.save(we_file)

if __name__ == "__main__":
    we = "glove_svd_50.npz"
    w2i = "glove_word2idx_50.npz"
    main(we, w2i, use_brown=False)

    # Load back embeddings
    npz = np.load(we)
    W1 = npz['arr_0']
    W2 = npz['arr_1']

    with open(w2i) as f:
        word2idx = json.load(f)
        idx2word = {i:w for w, i in word2idx.items()}

    for concat in (True, False):
        print("** concat: ", concat)

        if concat:
            We = np.hstack([W1, W2.T])
        else:
            We = (W1 + W2.T) / 2

        find_analogies('king', 'man', 'woman', We, word2idx, idx2word)
        find_analogies('france', 'paris', 'london', We, word2idx, idx2word)
        find_analogies('france', 'paris', 'rome', We, word2idx, idx2word)
        find_analogies('paris', 'france', 'italy', We, word2idx, idx2word)
        find_analogies('france', 'french', 'english', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'chinese', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'italian', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'australian', We, word2idx, idx2word)
        find_analogies('december', 'november', 'june', We, word2idx, idx2word)
