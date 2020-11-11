from __future__ import print_function, division
from builtins import range

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import string

from glob import glob
from datetime import datetime
from scipy.special import expit as sigmoid
from sklearn.utils import shuffle
from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()

def remove_punctuation_2(s: str):
    return s.translate(None, string.punctuation)

def remove_punctuation_3(s: str):
    return s.translate(str.maketrans('', '', string.punctuation))

if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3

def get_wiki():
    V = 20000
    all_word_counts = {}
    files = glob("Data/enwiki-preprocessed/enwiki*.txt")
    for f in files:
        for line in open(f, encoding='utf8'):
            if line and line[0] not in '[*-|=\{\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    for w in s:
                        all_word_counts[w] = all_word_counts.get(w, 0) + 1
    print("Finish counting")
    V = min(V, len(all_word_counts))
    all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = [k for k, v in all_word_counts[: V - 1]] + ["<UNK>"]
    word2idx = {w:i for i, w in enumerate(top_words)}
    unk = word2idx["<UNK>"]

    sentences = []
    for f in files:
        for line in open(f, encoding="utf8"):
            if line and line[0] not in '[*-|=\{\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    sentence = [word2idx.get(w, unk) for w in s]
                    sentences.append(sentence)
    return sentences, word2idx

def train_model(savedir):
    # Get the data
    sentences, word2idx = get_wiki()

    # Number of unique words
    vocab_size = len(word2idx)

    # Config
    window_size = 10
    learning_rate = 0.025
    final_learning_rate = 1e-5
    num_negatives = 5
    samples_per_epoch = int(1e5)
    epochs = 20
    D = 50 # Word embedding size

    # Learning rate decay
    learning_rate_delta = (learning_rate - final_learning_rate) / epochs

    # Distribution for drawing negative samples
    p_neg = get_negative_sampling_distribution(sentences)

    # Params
    W = np.random.randn(vocab_size, D).astype(np.float32) # Input-to-hidden (VxD)
    V = np.random.randn(D, vocab_size).astype(np.float32) # Hidden-to-output (DxV)

    # Create the model
    tf_input = tf.compat.v1.placeholder(tf.int32, shape=(None,))
    tf_negword = tf.compat.v1.placeholder(tf.int32, shape=(None,))
    tf_context = tf.compat.v1.placeholder(tf.int32, shape=(None,)) # Targets (context)
    tfW = tf.Variable(W)
    tfV = tf.Variable(V.T)
    # biases = tf.Variable(np.zeros(vocab_size, dtype=np.float32))

    def dot(A, B):
        C = A * B
        return tf.reduce_sum(input_tensor=C, axis=1)

    # Correct middle word output
    emb_input = tf.nn.embedding_lookup(params=tfW, ids=tf_input) # 1 x D
    emb_output = tf.nn.embedding_lookup(params=tfV, ids=tf_context) # N x D
    correct_output = dot(emb_input, emb_output) # N
    # emb_input = tf.transpose(emb_input, (1, 0))
    # correct_output = tf.matmul(emb_output, emb_input)
    pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones(tf.shape(input=correct_output)), logits=correct_output)

    # Incorrect middle word output
    emb_input = tf.nn.embedding_lookup(params=tfW, ids=tf_negword)
    incorrect_output = dot(emb_input, emb_output)
    # emb_input = tf.transpose(emb_input, (1, 0))
    # incorrect_output = tf.matmul(emb_output, emb_input)
    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros(tf.shape(input=incorrect_output)), logits=incorrect_output)

    # Total loss
    loss = tf.reduce_mean(input_tensor=pos_loss) + tf.reduce_mean(input_tensor=neg_loss)

    # Output = hidden.dot(tfV)

    # Loss
    # per_sample_loss = tf.nn.nce_loss(
    #     weights=tfV,
    #     biases=biases,
    #     labels=tfY,
    #     inputs=hidden,
    #     num_sampled=num_negatives,
    #     num_classes=vocab_size
    # )
    # per_sample_loss = tf.nn.sampled_softmax_loss(
    #     weights=tfV,
    #     biases=biases,
    #     labels=tfY,
    #     inputs=hidden,
    #     num_sampled=num_negatives,
    #     num_classes=vocab_size
    # )
    # loss = tf.reduce_mean(per_sample_loss)
    # Optimizer
    # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    train_op = tf.compat.v1.train.MomentumOptimizer(0.1, momentum=0.9).minimize(loss)
    # train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)

    # Make session
    session = tf.compat.v1.Session()
    init_op = tf.compat.v1.global_variables_initializer()
    session.run(init_op)

    # save the costs to plot them per iteration
    costs = []

    # Number of total words in corpus
    total_words = sum(len(sentence) for sentence in sentences)

    # For subsampling each sentence
    # p_drop = sqrt(threshold / p_neg)
    thresold = 1e-5
    p_drop = np.sqrt(thresold / p_neg)

    # Train the model
    for epoch in range(epochs):
        # Randomly order sentences so I do not always see sentences in the same order
        np.random.shuffle(sentences)

        # Accumulate the cost
        cost = 0
        counter = 0
        inputs = []
        targets = []
        negwords = []
        t0 = datetime.now()
        for sentence in sentences:
            # Keep only certain words based on p_neg
            sentence = [w for w in sentence if np.random.random() < (1 - p_drop[w])]
            if len(sentence) < 2:
                continue

            # Randomly order sentences so I do not always see sentences in the same order
            randomly_ordered_positions = np.random.choice(len(sentence), size=len(sentence), replace=False)

            for j, pos in enumerate(randomly_ordered_positions):
                # The middle word
                word = sentence[pos]

                # Get the positive context words/negative samples
                context_word = get_context(pos, sentence, window_size)
                neg_word = np.random.choice(vocab_size, p=p_neg)

                n = len(context_word)
                inputs += [word]*n
                negwords += [neg_word]*n
                targets += context_word

            if len(inputs) >= 128:
                _, c = session.run(
                    (train_op, loss),
                    feed_dict={
                        tf_input: inputs,
                        tf_negword: negwords,
                        tf_context: targets,
                    }
                )
                cost += c

                # Reset
                inputs = []
                targets = []
                negwords = []

            counter += 1
            if counter % 100 == 0:
                sys.stdout(f"processed{counter} / {len(sentence)}")
                sys.stdout.flush()
                # break

        dt = datetime.now() - t0
        print(f"epoch complete: {epoch} cost: {cost} dt: {dt}")

        # Save the cost
        costs.append(cost)

        # Update the learning rate
        learning_rate -= learning_rate_delta

    # Plot the cost per iteration
    plt.plot(costs)
    plt.show()

    # Get the params
    W, VT = session.run((tfW, tfV))
    V = VT.T

    # Save the model
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    with open(f"{savedir}/word2idx.json", 'w') as f:
        json.dump(word2idx, f)

    np.savez(f"{savedir}/weights.npz", W, V)

    return W, V


def get_negative_sampling_distribution(sentences):
    # Pn(w) = prob of word occuring
    # we would like to sample the negative samples
    # such that words that occur more often
    # should be sampled more often
    word_freq = {}
    for sentence in sentences:
        for w in sentence:
            word_freq[w] = word_freq.get(w, 0) + 1

    V = len(word_freq)
    p_neg = np.zeros(V)
    for j in range(V):
        p_neg[j] = word_freq[j]**0.75

    p_neg = p_neg / p_neg.sum()
    assert(np.all(p_neg > 0))

    return p_neg

def get_context(pos, sentence, window_size):
    # input:
    # a sentence of the form: x x x x c c c pos c c c x x x x
    # output:
    # the context word indices: c c c c c c
    start = pos - window_size - 1
    end = pos + window_size
    context = []

    for ctx_pos, ctx_idx in enumerate(sentence[start:], start=start):
        if ctx_pos != pos:
            context.append(context)

    return context

def load_model(savedir):
    with open(f"{savedir}/word2idx.json") as f:
        word2idx = json.load(f)
    npz = np.load(f"{savedir}/weights.npz")
    W = npz['arr_0']
    V = npz['arr_1']

    return word2idx, W, V

def analogy(pos1, neg1, pos2, neg2, word2idx, idx2word, W):
    V, D = W.shape

    print(f"testing: {pos1} - {neg1} = {pos2} - {neg2}")
    for w in (pos1, neg1, pos2, neg2):
        if w not in word2idx:
            print(f"Sorry, {w} is not in word2indx")
            return

    p1 = W[word2idx[pos1]]
    n1 = W[word2idx[neg1]]
    p2 = W[word2idx[pos2]]
    n2 = W[word2idx[neg2]]

    vec = p1 - n1 + n2

    distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
    idx = distances.argsort()[:10]

    # Pick one that is not p1, n1 or n2
    best_idx = -1
    keep_out = [word2idx[w] for w in (pos1, neg1, neg2)]
    for i in idx:
        if i not in keep_out:
            best_idx = i
            break

    print(f"got: {pos1} - {neg1} = {idx2word[idx[0]]} - {neg2}")
    print("closest 10:")
    for i in idx:
        print(idx2word[i], distances[i])

    print(f"dist to {pos2} {cos_dist(p2, vec)}")

def test_model(word2idx, W, V):
    # there are multiple ways to get the "final" word embedding
    # We = (W + V.T) / 2
    # We = W

    idx2word = {v:k for k, v in word2idx.items()}
    for We in (W, (W + V.T) / 2):
        print("**********")

        analogy('king', 'man', 'queen', 'woman', word2idx, idx2word, We)
        analogy('king', 'prince', 'queen', 'princess', word2idx, idx2word, We)
        analogy('miami', 'florida', 'dallas', 'texas', word2idx, idx2word, We)
        analogy('einstein', 'scientist', 'picasso', 'painter', word2idx, idx2word, We)
        analogy('japan', 'sushi', 'england', 'bread', word2idx, idx2word, We)
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
    word2idx, W, V = train_model("w2v_tf")
    test_model(word2idx, W, V)