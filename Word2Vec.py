from __future__ import print_function, division
from builtins import range

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import string
from scipy.special import expit as sigmoid
from sklearn.utils import shuffle
from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances
from datetime import datetime
from glob import glob
from rnn_class.brown import get_sentences_with_word2indx_limit_vocab as get_brown

def remove_punctuation_2(s: str):
    return s.translate(None, string)

def remove_punctuation_3(s: str):
    return s.translate(str.maketrans("", "", string.punctuation))

if sys.version.startswith("2"):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3

def get_wiki():
    V = 20000
    files = glob("Data/enwiki-preprocessed/enwiki*.txt")
    all_words_count = {}
    for f in files:
        for line in open(f, encoding="utf8"):
            if line and line[0] not in "[*-|=\{\}":
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    for word in s:
                        all_words_count[word] = all_words_count.get(word, 0) + 1
    print("Finish counting.")
    all_words_count = sorted(all_words_count.items(), key=lambda x:x[1], reverse=True)
    V = min(V, len(all_words_count))
    top_words = [k for k, v in all_words_count[:V-1]] + ["<UNK>"]
    word2idx = {w:i for i, w in enumerate(top_words)}
    unk = word2idx["<UNK>"]
    sentences = []
    for f in files:
        for line in open(f, encoding="utf8"):
            if line and line[0] not in "[*-|=\{\}":
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    sentence = [word2idx.get(word, unk) for word in s]
                    sentences.append(sentence)

    return sentences, word2idx

def train_model(savedir):
    # Get data
    sentences, word2idx = get_wiki()
    # Number of unique words
    vocab_size = len(word2idx)
    # Config
    window_size = 5
    learning_rate = 0.025
    final_learning_rate = 1e-4
    num_negatives = 5 # Number of negative samples to draw per input word
    epochs = 20
    D = 50 # Word embedding size
    # Learning rate decay
    learning_rate_delta = (learning_rate - final_learning_rate) / epochs

    # Params
    W = np.random.randn(vocab_size, D) # Input-to-hidden
    V = np.random.randn(D, vocab_size) # Hidden-to-output

    # Distribution for drawing negative samples
    p_neg = get_negative_sampling_distribution(sentences, vocab_size)
    # Save the costs to plot them per iteration
    costs = []
    # Number of total words in corpus
    total_words = sum(len(sentence) for sentence in sentences)
    # For subsampling each sentence
    # p_drop(w) = 1 - sqrt(threshold / p(w))
    threshold = 1e-5
    p_drop = 1 - np.sqrt(threshold / p_neg)

    # Train the model
    for epoch in range(epochs):
        # Randomly order sentences so I do not always see sentences in the same order
        np.random.shuffle(sentences)
        # Accumulate the cost
        cost = 0
        counter = 0
        t0 = datetime.now()
        for sentence in sentences:
            # Keep only certain words based on p_neg
            sentence = [w for w in sentence if np.random.random() < (1 - p_drop[w])]
            if len(sentence) < 2:
                continue

            # Randomly order sentences so I do not always see sentences in the same order
            randomly_ordered_positions = np.random.choice(len(sentence), size=len(sentence), replace=False)

            for pos in randomly_ordered_positions:
                # The middle word
                word = sentence[pos]
                # Get the positive context words/negative samples
                context = get_context(pos, sentence, window_size)
                neg_word = np.random.choice(vocab_size, p=p_neg)
                targets = np.array(context)
                # Do one iteration of stochastic gradient descent
                c = sgd(word, targets, 1, learning_rate, W, V)
                cost += c
                c = sgd(word, targets, 0, learning_rate, W, V)
                cost += c

            counter += 1
            if counter % 100 == 0:
                print(f"processed {counter} / {len(sentence)}")
                break

            dt = datetime.now() - t0
            print(f"epoch complete: {epoch} cost: {cost} dt: {dt}")
            # Save the cost
            costs.append(cost)
            # Update the learning rate
            learning_rate -= learning_rate_delta
        # Plot the cost per iteration
        plt.plot(costs)
        plt.show()

        # Save the model
        if not os.path.exists(savedir):
            os.mkdir(savedir)

        with open(f"{savedir}/word2idx.json", 'w') as f:
            json.dump(word2idx, f)

        np.savez(f"{savedir}/weights.npz", W, V)

        return word2idx, W, V

def get_negative_sampling_distribution(sentences, vocab_size):
    # Pn(w) = prob of word occuring
    # I would like to sample the negative samples such that
    # words occur more often should be sampled more often
    word_freq = np.zeros(vocab_size)
    word_count = sum(len(s) for s in sentences)
    for sentence in sentences:
        for word in sentence:
            word_freq[word] += 1
    # Smooth the p_neg
    p_neg = word_freq**0.75
    # Normalize the p_neg. Negative sampling distribution function: p(w) = count(w)^0.75/sum(count^0.75)
    p_neg = p_neg / p_neg.sum()

    assert(np.all(p_neg>0))
    return p_neg

def get_context(pos, sentence, window_size):
    # Input:
    # A sentence of form: x x x x c c c pos c c c x x x x
    # Output:
    # the context word indices: c c c c c c
    # Check if the start or end indices are out of bounce
    start = max(0, pos - window_size - 1)
    end = min(len(sentence), pos + window_size)
    context = []
    for ctx_pos, ctx_word_index in enumerate(sentence[start:end], start=start):
        if ctx_pos != pos:
            context.append(ctx_word_index)

    return context

def sgd(input_, targets, label, learning_rate, W, V):
    # W[input_] shape: D
    # V[:, targets] shape: D x N
    # activation shape: N
    # print("input_:", input_, "targets:", targets)
    # Linear activation so prob = sigmoid(W1[word], W2[:, context])
    activation = W[input_].dot(V[:, targets])
    prob = sigmoid(activation)
    # Gradients
    # gW1 = outer(W1[word], prob - label)
    # gW2 = W2[:, context] . (prob - label)
    gV = np.outer(W[input_], prob - label) # D x N
    gW = np.sum((prob - label) * V[:, targets], axis=1) # D
    # W2[:, context] -= lr * gW2
    # W1[word] -= lr * gW1
    V[:, targets] -= learning_rate * gV # D x N
    W[input_] -= learning_rate * gW # D
    # Return -(label*log(prob) + (1-label)*log(1-prob)).sum()
    cost = label * np.log(prob + 1e-10) + (1 - label) * np.log(1 - prob + 1e-10)
    return cost.sum()

def load_model(savedir):
    with open(f"{savedir}/word2idx.json") as f:
        word2idx = json.load(f)
    npz = np.load(f"{savedir}/weights.npz")
    W = npz["arr_0"]
    V = npz["arr_1"]
    return word2idx, W, V

def analogy(pos1, neg1, pos2, neg2, word2inx, idx2word, W):
    V, D = W.shape

    # Do not actually use pos2 in calculation, just print what is expected
    print(f"testing: {pos1} - {neg1} = {pos2} - {neg2}")
    for w in (pos1, neg1, pos2, neg2):
        if w not in word2inx:
            print(f"Sorry, {w} not in word2idx")
            return

    p1 = W[word2inx[pos1]]
    n1 = W[word2inx[neg1]]
    p2 = W[word2inx[pos2]]
    n2 = W[word2inx[neg2]]

    vec = p1 - n1 + n2

    distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
    idx = distances.argsort()[:10]

    # Pick one that is not p1, n1, or n2
    best_idx = -1
    keep_out = [word2inx[w] for w in (pos1, neg1, neg2)]
    for i in idx:
        print(idx2word[i], distances[i])

    print(f"dist to {pos2} {cos_dist(p2, vec)}")

def test_model(word2idx, W, V):
    idx2word = {v:k for k, v in word2idx.items()}

    for We in (W, (W + V.T) / 2):
        print("******************")

        analogy('king', 'man', 'queen', 'woman', word2idx, idx2word, We)
        analogy('king', 'prince', 'queen', 'princess', word2idx, idx2word, We)
        analogy('miami', 'florida', 'dallas', 'texas', word2idx, idx2word, We)
        analogy('einstein', 'scientist', 'picasso', 'painter', word2idx, idx2word, We)
        analogy('japan', 'sushi', 'germany', 'bratwurst', word2idx, idx2word, We)
        analogy('man', 'woman', 'he', 'she', word2idx, idx2word, We)
        analogy('man', 'woman', 'uncle', 'aunt', word2idx, idx2word, We)
        analogy('man', 'woman', 'brother', 'sister', word2idx, idx2word, We)
        analogy('man', 'woman', 'husband', 'wife', word2idx, idx2word, We)
        analogy('man', 'woman', 'actor', 'actress', word2idx, idx2word, We)
        analogy('man', 'woman', 'father', 'mother', word2idx, idx2word, We)
        analogy('heir', 'heiress', 'prince', 'princess', word2idx, idx2word, We)
        analogy('nephew', 'niece', 'uncle', 'aunt', word2idx, idx2word, We)
        analogy('france', 'paris', 'japan', 'tokyo', word2idx, idx2word, We)
        analogy('france', 'paris', 'china', 'beijing', word2idx, idx2word, We)
        analogy('february', 'january', 'december', 'november', word2idx, idx2word, We)
        analogy('france', 'paris', 'germany', 'berlin', word2idx, idx2word, We)
        analogy('week', 'day', 'year', 'month', word2idx, idx2word, We)
        analogy('week', 'day', 'hour', 'minute', word2idx, idx2word, We)
        analogy('france', 'paris', 'italy', 'rome', word2idx, idx2word, We)
        analogy('paris', 'france', 'rome', 'italy', word2idx, idx2word, We)
        analogy('france', 'french', 'england', 'english', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'china', 'chinese', word2idx, idx2word, We)
        analogy('china', 'chinese', 'america', 'american', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'italy', 'italian', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'australia', 'australian', word2idx, idx2word, We)
        analogy('walk', 'walking', 'swim', 'swimming', word2idx, idx2word, We)

if __name__ == "__main__":
    word2idx, W, V = train_model("w2v_model")
    test_model(word2idx, W, V)