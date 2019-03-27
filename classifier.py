
from flask import Flask,render_template, request,flash,make_response, url_for
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import csv
import tweepy
import matplotlib.pyplot as plt

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
# from wordcloud import WordCloud,STOPWORDS
from numpy import nan
# from bs4 import BeautifulSoup    
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from matplotlib.figure import Figure
import StringIO
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
from matplotlib.ticker import MaxNLocator
import pickle
from sklearn import datasets

file1='Sentiment_Analysis_of_50_Tweets_About_'+user_input+'.csv'
train(file1)

def train(filename):
    train_data = pd.read_csv('imdbtest.csv')
    print('read data')
    Sentiment_words=[]
    for row in train_data['polarity']:
        if row ==0:
            Sentiment_words.append('negative')
        elif row == 1:
            Sentiment_words.append('positive')
    train_data['Sentiment'] = Sentiment_words
    print('converted polarity to sentiment')
    lengthtrain = len(train_data['Sentiment'])
    corpus= []
    for i in range(0, lengthtrain):
        if i%1000==0:
            print('reviewing %d of the text' % i)
        corpus.append(cleanTweets(train_data['text'][i]))
    print('cleaned data')
    train_data['new_Phrase']=corpus
    cv = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
    messages_bow=cv.fit_transform(train_data['new_Phrase'])

    print('Shape of Sparse Matrix: ', messages_bow.shape)
    print('Amount of Non-Zero occurences: ', messages_bow.nnz)

    mnb = MultinomialNB()

    spam_detect_model = mnb.fit(messages_bow,train_data['Sentiment'])

    print('training done')
    print(spam_detect_model)

    f = open('my_classifier.pickle', 'wb')
    pickle.dump(spam_detect_model, f)
    f.close()