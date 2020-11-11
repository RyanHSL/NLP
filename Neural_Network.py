from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input

import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

import os
import sys
from rnn_class.utils import get_wikipedia_data
from rnn_class.brown import get_sentences_with_word2indx_limit_vocab, get_sentences_with_word2idx
from Markov import get_bigram_probs

if __name__ == "__main__":
    # load in the data
    # note: sentences are already converted to sequences of word indexes
    # note: you can limit the vocab size if you run out of memory
    sentences, word2idx = get_sentences_with_word2indx_limit_vocab(2000)

    V = len(word2idx)
    print("Vocab size:", V)
    start_idx = word2idx['START']
    end_idx = word2idx['END']

    # a matrix where: row = last word and col = current word
    # value at [row, col] = p(current word | last word)
    bigram_probs = get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=0.1)

    D = 100
    W1 = np.random.randn(V, D) / np.sqrt(V)
    W2 = np.random.randn(D, V) / np.sqrt(D)

    losses = []
    epochs = 1
    lr = 1e-2

    def softmax(a):
        a = a - a.max()
        exp_a = np.exp(a)
        return exp_a / exp_a.sum(axis=1, keepdims=True)

    # what is the loss if we set W = log(bigrams_probs)?
    W_bigram = np.log(bigram_probs)
    bigram_losses = []

    t0 = datetime.now()
    for epoch in range(epochs):
        random.shuffle(sentences)

    j = 0 # keep track of iterations
    for sentence in sentences:
        # convert sentence into one-hot encoding inputs and targets
        sentence = [start_idx] + sentence + [end_idx]
        n = len(sentence)
        inputs = np.zeros((n - 1, V))
        targets = np.zeros((n - 1, V))
        inputs[np.arange(n - 1), sentence[:n - 1]] = 1
        targets[np.arange(n - 1), sentence[1:]] = 1

        # get output predictions
        hidden = np.tanh(inputs.dot(W1))
        predictions = softmax(hidden.dot(W2))

        # do a gradient descent step
        W2 = W2 - lr * hidden.T.dot(predictions - targets)
        dhidden = (predictions - targets).dot(W2.T) * (1 - hidden * hidden)
        W1 = W1 - lr * inputs.T.dot(dhidden)

        # keep track of the loss
        loss = -np.sum(targets * np.log(predictions)) / (n - 1)
        losses.append(loss)

        # keep track of the bigram loss
        # only do it for the first epoch to avoid redundancy
        if epoch == 0:
            bigram_predictions = softmax(inputs.dot(W_bigram))
            bigram_loss = -np.sum(targets * np.log(bigram_predictions)) / (n - 1)
            bigram_losses.append(bigram_loss)

        if j % 10 == 0:
            print(f"epoch:{epoch} sentence:{j}/{len(sentence)} loss:{loss}")
        j += 1

    print("Elapsed time training:", datetime.now() - t0)
    plt.plot(losses)

    # plot a horizontal line for the bigram loss
    avg_bigram_loss = np.mean(bigram_losses)
    print("avg_bigram_loss:", avg_bigram_loss)
    plt.axhline(y=avg_bigram_loss, color='r', linestyle='-')

    # plot smoothed losses to reduce variablilty
    def smooth_loss(x, decay=0.99):
        y = np.zeros(len(x))
        last = 0
        for t in range(len(x)):
            z = decay * last + (1 - decay) * x[t]
            y[t] = z / (1 - decay ** (t + 1))
            last = z
        return y

    plt.plot(smooth_loss(losses))
    plt.show()

    # plot W and bigram probs side-by-side for the most common 200 words
    plt.subplot(1, 2, 1)
    plt.title("Neural Network Model")
    plt.imshow(np.tanh(W1).dot(W2))
    plt.subplot(1, 2, 2)
    plt.title("Bigram Probs")
    plt.imshow(W_bigram)
    plt.show()