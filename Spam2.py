from __future__ import print_function, division
from future.utils import iteritems
from builtins import range

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from wordcloud import WordCloud

# Get data using encoding='ISO-8859-1'
df = pd.read_csv("Data/spam/spam.csv", encoding='ISO-8859-1')
# Drop unecessary columns ("Unnamed: 2", "Unnamed: 3", "Unnamed: 4")
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
# Rename the columns to something better ("labels", "data")
df.columns = ['labels', 'data']
# Create binary labels (b_labels)
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
# Split up the data
df_train, df_test, y_train, y_test = train_test_split(df['data'], df['b_labels'].values, train_size=0.33, random_state=42, shuffle=True)
# Try multiple ways of calculating features(TfidfVectorizer, CountVectorizer)
# tfidf = TfidfVectorizer(decode_error='ignore') # train score: 0.95, test score: 0.93
# x_train = tfidf.fit_transform(df_train)
# x_test = tfidf.transform(df_test)
count = CountVectorizer(decode_error='ignore') # train score: 0.99, test score: 0.98
x_train = count.fit_transform(df_train)
x_test = count.transform(df_test)
# Create the model(MultinomialNB), train it, print scores
model = MultinomialNB()
model.fit(x_train, y_train)
print("training scores:", model.score(x_train, y_train))
print("test scores", model.score(x_test, y_test))
# Visualize the data
def visualize(label):
    words = ''
    for msg in df[df['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

# See what I have got wrong
# x = tfidf.transform(df['data'])
x = count.transform(df['data'])
df['predictions'] = model.predict(x)
# Things that should be spam
sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
for msg in sneaky_spam:
    print(msg)
# Things that should not be spam
not_actually_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg in not_actually_spam:
    print(msg)