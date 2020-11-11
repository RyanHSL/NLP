from __future__ import print_function, division
from builtins import range

import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from gensim.models import KeyedVectors

train = pd.read_csv("Data/r8-train-all-terms.txt", header=None, sep='\t')
test = pd.read_csv("Data/r8-test-all-terms.txt", header=None, sep='\t')
train.columns = ['label', 'content']
test.columns = ['label', 'content']

class GloveVector:
    def __init__(self):
        print("Loading word vectors...")
        word2vec = {}
        embedding = []
        idx2word = []
        with open("Data/glove.6B/glove.6B.50d.txt", encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec
                embedding.append(vec)
                idx2word.append(word)
        print("Found %s word vectors." % len(word2vec))

        self.word2vec = word2vec
        self.embedding = np.array(embedding)
        self.word2idx = {v:k for k, v in enumerate(idx2word)}
        self.V, self.D = self.embedding.shape

    def fit(self, data):
        pass

    def transform(self, data):
        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.lower().split()
            vecs = []
            for word in tokens:
                if word in self.word2vec:
                    vec = self.word2vec[word]
                    vecs.append(vec)
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print("Number of samples with no words found: %s / %s" % (emptycount, len(data)))
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

class Word2VecVectorizer:
    def __init__(self):
        print("Loading in word vectors...")
        self.word_vectors = KeyedVectors.load_word2vec_format(
            "Data/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin",
            binary=True
        )
        print("Finished loading in word vectors")

    def fit(self, data):
        pass

    def transform(self, data):
        v = self.word_vectors.get_vector('king')
        self.D = v.shape[0]

        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.split() # Do not use lower because the Google vector is case sensitive
            vecs = []
            m = 0
            for word in tokens:
                try:
                    # Throws KeyError if word not found
                    vec = self.word_vectors.get_vector(word)
                    vecs.append(vec)
                    m += 1
                except KeyError:
                    pass
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print("Number of samples with no words found: %s / %s" % (emptycount, len(data)))
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

vectorizer = GloveVector()
# vectorizer = Word2VecVectorizer()
Xtrain = vectorizer.fit_transform(train.content)
Ytrain = train.label
Xtest = vectorizer.fit_transform(test.content)
Ytest = test.label

model = RandomForestClassifier(n_estimators=200)
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print(("test score:", model.score(Xtest, Ytest)))