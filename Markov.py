from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input

import numpy as np
import os
import sys
# sys.path.append(os.path.abspath('..'))
from rnn_class.brown import get_sentences_with_word2indx_limit_vocab, get_sentences_with_word2idx

def get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=1):
    bigram_probs = np.ones((V, V)) * smoothing
    for sentence in sentences:
        for i in range(len(sentence)):
            if i == 0:
                # beginning word
                bigram_probs[start_idx, sentence[i]] += 1
            else:
                # middle word
                bigram_probs[sentence[i-1], sentence[i]] += 1

            # if we are at the final word
            # we update the bigram for last -> current
            # AND current -> END token
            if i == len(sentence) - 1:
                # final word
                bigram_probs[sentence[i], end_idx] += 1

    # normalize the counts along the rows to get probabilities
    bigram_probs /= bigram_probs.sum(axis=1, keepdims=True)
    return bigram_probs

if __name__ == "__main__":
    # load in the data
    # note: sentences are already converted to sequences of word indexes
    # note: you can limit the vocab size if you run out of memory
    sentences, word2idx = get_sentences_with_word2indx_limit_vocab(10000)
    V = len(word2idx)
    print("Vocab size:", V)
    # we will also treat beginning of sentence and end of sentence as bigrams
    # START -> first word
    # last word -> END
    start_idx = word2idx['START']
    end_idx = word2idx['END']

    # a matrix where:
    # row = last word
    # value at [row, col] = p(current word | last word)
    bigram_probs = get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=0.1)

    # a function to calculate normalized log prob score
    # for a sentence
    def get_score(sentence):
        score = 0
        for i in range(len(sentence)):
            if i == 0:
                # beginning word
                score += np.log(bigram_probs[start_idx, sentence[i]])
            else:
                # middle word
                score += np.log(bigram_probs[sentence[i-1], sentence[i]])
        # final word
        score += np.log(bigram_probs[sentence[-1], end_idx])

        # normalize the score
        return score / (len(sentence) + 1)

    # a function to map word indexes back real word
    idx2word = dict((v, k) for k, v in iteritems(word2idx))
    def get_words(sentence):
        return ' '.join(idx2word[i] for i in sentence)

    # when we sample a fake sentence, we want to ensure not to sample start token or end token
    sample_probs = np.ones(V)
    sample_probs[start_idx] = 0
    sample_probs[end_idx] = 0
    sample_probs /= sample_probs.sum()

    # test our model on real and fake sentences
    while True:
        # real sentence
        real_idx = np.random.choice(len(sentences))
        real = sentences[real_idx]

        # fake sentence
        fake = np.random.choice(V, size=len(real), p=sample_probs)
        print("Real:", get_words(real), "Score:", get_score(real))
        print("Fake:", get_words(fake), "Score:", get_score(fake))
        custom = input("Enter your own sentence:\n")
        custom = custom.lower().split()

        bad_sentence = False
        for token in custom:
            if token not in word2idx:
                bad_sentence = True

        if bad_sentence:
            print("Sorry, you entered words that are not in the vocabulary")
        else:
            custom = [word2idx[token] for token in custom]
            print("Score:", get_score(custom))

        cont = input("Continue? [y/n]")
        if cont and cont.lower() in ('N', 'n'):
            break