import pandas as pd
import re
from nltk import *
nltk.download()
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet

def train():
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

    g = open('cv.pickle', 'wb')
    pickle.dump(cv, g, protocol=pickle.HIGHEST_PROTOCOL)
    g.close()

    f = open('my_classifier.pickle', 'wb')
    pickle.dump(spam_detect_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def cleanTweets(raw_review):
    review =raw_review
    # review = re.sub('[^a-zA-Z]', ' ',review)
    result = re.sub(r"http\S+", "", review) #removing URLs
    result = re.sub(r'@[A-Za-z0-9]+','',result) #removing @
    result = re.sub("[^a-zA-Z]", " ", result) #removing hashtags
    review = result.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    return (' '.join(review))

train()