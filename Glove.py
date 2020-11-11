import numpy as np
import json
import os
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.utils import shuffle
from Word2Vec import get_wiki, analogy
from rnn_class.brown import get_sentences_with_word2indx_limit_vocab, get_sentences_with_word2idx
from rnn_class.utils import get_wikipedia_data
from utils import find_analogies

class Glove:
    def __init__(self, D, V, context_sz):
        self.D = D
        self.V = V
        self.context_sz = context_sz

    def fit(self, sentences, cc_matrix=None, learning_rate=1e-4, reg=0.1, xmax=100, alpha=0.75, epochs=10, gd=False):
        # Build the co-occurence matrix
        # Paper calls it X, so I will call it X, instead of call the training data X
        # TODO: would it be better to use a sparse matrix?
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
                    print("processed ", it, "/", N)
                n = len(sentence)
                for i in range(n):
                    # i is the index of the word index
                    wi = sentence[i]
                    start = max(0, i - self.context_sz)
                    end = min(n, i + self.context_sz)

                    # I can either choose only one side as context, or both
                    # Here I am doing both

                    # Make sure "start" and "end" tokens are part of some context
                    # Otherwise their f(X) will be 0 (denominator in bias update)
                    if i - self.context_sz < 0:
                        point = 1.0/(1 + i)
                        X[wi, 0] += point
                        X[0, wi] += point # 0 means start
                    if i + self.context_sz > n:
                        point = 1.0/(n - i)
                        X[wi, 1] += point
                        X[i, wi] += point # 1 mean end

                    # Left side
                    for j in range(start, i):
                        wj = sentence[j]
                        point = 1.0/(i - j)
                        X[wi, wj] += point
                        X[wj, wi] += point
                    # Right side
                    for j in range(i + 1, end):
                        wj = sentence[j]
                        point = 1.0/(j - i)
                        X[wi, wj] += point
                        X[wj, wi] += point

            # save the cc matrix because it takes forever to create
            np.save(cc_matrix, X)
        else:
            X = np.load(cc_matrix)

        print("max in X: ", X.max())
        # Weighting
        fX = np.zeros((V, V))
        fX[X < xmax] = (X[X < xmax] / float(xmax))**alpha
        fX[X >= xmax] = 1

        print("max in f(X): ", fX.max())

        # Target
        logX = np.log(X + 1) # Avoid NaN
        print("max in log(X): ", logX.max())
        print("time to build co-occurence matrix: ", (datetime.now() - t0))

        # Initalize weights
        W = np.random.randn(V, D) / np.sqrt(V + D)
        b = np.zeros(V)
        U = np.random.randn(V, D) / np.sqrt(V + D)
        c = np.zeros(V)
        mu = logX.mean()

        costs = []
        sentence_indexes = range(len(sentences))
        for epoch in range(epochs):
            delta = W.dot(U.T) + b.reshape(V, 1) + c.reshape(1, V) + mu - logX
            cost = (fX * delta * delta).sum()
            costs.append(cost)
            print("epoch: ", epoch, "cost: ", cost)

            if gd:
                # gradient descent method
                # update W
                oldW = W.copy()
                for i in range(V):
                    # for j in range(V):
                    #     W[i] -= learning_rate * fX[i,j] * (W[i].dot(U[j]) + b[i] + c[j] + mu - logX[i,j]) * U[j]
                    W[i] -= learning_rate * (fX[i, :] * delta[i, :]).dot(U)
                W -= learning_rate * reg * W

                # Update b
                for i in range(V):
                    # for j in range(V):
                    #     b[i] -= learning_rate * fX[i, j] * (W[i].dot(U[j]) + b[j] + c[j] + mu - logX[i, j])
                    b[i] -= learning_rate * fX[i, :].dot(delta[i, :])
                b -= learning_rate * reg * b

                # Update U
                for j in range(V):
                    U[j] -= learning_rate * (fX[:, j] * delta[:, j]).dot(oldW)
                U -= learning_rate * reg * U

                # Update c
                for j in range(V):
                    c[j] -= learning_rate * fX[:, j].dot(delta[:, j])
                c -= learning_rate * reg * c
            else:
                # Alternating Least Square (ALS) method
                # Update W
                for i in range(V):
                    # matrix = reg * np.eye(D) + np.sum((fX[i, j] * np.outer(U[j], U[i]) for j in range(V)), axis=0)
                    matrix = reg * np.eye(D) + (fX[i, :] * U.T).dot(U)
                    vector = (fX[i, :] * (logX[i, :] - b[i] - c - mu)).dot(U)
                    W[i] = np.linalg.solve(matrix, vector)

                # Update b
                for i in range(V):
                    denominator = fX[i, :].sum() + reg
                    assert(denominator > 0)
                    numerator = fX[i, :].dot(logX[i, :] - W[i].dot(U.T) - c - mu)
                    # for j in range(V):
                    #     numerator += fX[i, j] * (logX[i, j] - W[i].dot(U[j] - c[j]))
                    b[i] = numerator / denominator

                # Update U
                for j in range(V):
                    matrix = reg * np.eye(D) + (fX[:, j] * W.T).dot(W)
                    vector = (fX[:, j] * (logX[:, j] - b - c[j] - mu)).dot(W)
                    U[j] = np.linalg.solve(matrix, vector)

                # Update c
                for j in range(V):
                    denominator = fX[:, j].sum() + reg
                    assert(denominator > 0)
                    numerator = fX[:, j].dot(logX[:, j] - W.dot(U[j]) - b - mu)
                    c[j] = numerator / denominator

        self.W = W
        self.U = U

        plt.plot(costs)
        plt.show()

    def save(self, fn):
        # function word_analogies expects a (V, D) matrix and a (D, V) matrix
        arrays = [self.W, self.U.T]
        np.savez(fn, *arrays)

def main(we_file, w2i_file, use_brown=True, n_files=100):
    if use_brown:
        cc_matrix = "cc_matrix_brown.npy"
    else:
        cc_matrix = "cc_matrix%s.npy" % n_files

    # hacky way of checking if I need to re-load the raw data or not
    # remember, only the co-occurrence matrix is needed for training
    if os.path.exists(cc_matrix):
        with open(w2i_file) as f:
            word2idx = json.load(f)
        sentences = [] # dummy - I won't actually use it
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
    model = Glove(100, V, 10)

    # Alternating least squares method
    model.fit(sentences, cc_matrix, epochs=20)

    # Gradient descent method
    # model.fit(
    #     sentences,
    #     cc_matrix=cc_matrix,
    #     learning_rate=1e-4,
    #     reg=0.1,
    #     epochs=500,
    #     gd=True
    # )

    model.save(we_file)

if __name__ == "__main__":
    we = "glove_model_50.npz"
    w2i = "glove_word2idx_50.json"
    main(we, w2i, use_brown=False)

    # Load back embeddings
    npz = np.load(we)
    W1 = npz["arr_0"]
    W2 = npz["arr_1"]

    with open(w2i) as f:
        word2idx = json.load(f)
        idx2word = {i:w for w,i in word2idx.items()}

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