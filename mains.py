from flask import Flask,render_template, request,flash,make_response, url_for
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import csv
# import tweepy
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



app = Flask(__name__)
@app.route("/")
def start():
    return render_template('homepage.html')

@app.route("/test",methods = ['GET','POST'])
def test():
    if request.method == 'POST':
        print('in post')
        try:
            print('in try')
            user_input=request.form['mname']
            print(user_input)
            getTweets(user_input)
            file1='Sentiment_Analysis_of_50_Tweets_About_'+user_input+'.csv'
            print(file1)
            values=train(file1)
            user_input=user_input.capitalize()
            pic=plot(values,user_input)
            #print(obj)
            #testData(file1,obj)
            print('testing done')

            # print('Sentiment_Analysis_of_50_Tweets_About_'+user_input+'.csv')
            # pred='Sentiment_Analysis_of_50_Tweets_About_'+user_input+'.csv'
            # query = input("What subject do you want to analyze for this example? \n")

            print('test function')
            print(values)
            # print(pic)
            d={'Positive':values[0],'Negative':values[1],'result':pic}
            print(d)
            return render_template('firstpage.html',p=values,result=pic)
        except:
            print('in except block')
            error="You are already registered.Please log in."
            return render_template('homepage.html',error=error)
    else:
        print('in get')

        return render_template('homepage.html')

# @app.route("/",methods = ['GET','POST'])
# def check():
#     if request.method == 'POST':
#         try:
#             print('in try')
#             user_input=request.form['question']
#             print(user_input)
#             if(user_input == 'y'):
#                 obj=train()
#             else:
#                 return
#         except:

    # query=input('Do you want to train the data? Enter y for yes and n for no')
    # if(query=='y'):
    #     obj=train()
    #     return obj
    # else:
    #     return




def getTweets(user_input):
    consumer_key = "2ZoeWWY19CAURNll9RIzTWaOx"
    consumer_secret = "JYhvMNRqaXn2Z7f6gv1qVAyNHppiHYBZOjOIThazKAOFCetV2I"
    access_token = "78381104-htXdK7XHZEoF0PMAmMw6DxEKeKsVQwDjvLPY2VPhz"
    access_token_secret = "NtITCCkcrRbM6QNoSPuUjaNN3IxUPFUzDsAjo6zHRpzti"

    ## set up an instance of Tweepy
    print('1')
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # ## set up an instance of the AYLIEN Text API
    # client = textapi.Client(application_id, application_key)

    ## search Twitter for something that interests you
# query = input("What subject do you want to analyze for this example? \n")
# number = input("How many Tweets do you want to analyze? \n")
    query=user_input
    number=100
    print(query)
    print(number)
    print('2')
    results = api.search(
   lang="en",
   q=query + " -rt",
   count=number,
   result_type="recent")

    print("--- Gathered Tweets \n")

    ## open a csv file to store the Tweets and their sentiment 
    file_name = 'Sentiment_Analysis_of_{}_Tweets_About_{}.csv'.format(number, query)

    with open(file_name, 'w') as csvfile:
        csv_writer = csv.DictWriter(
            f=csvfile,
            fieldnames=["Tweet", "Sentiment"])
        csv_writer.writeheader()

        print("--- Opened a CSV file to store the results of your sentiment analysis... \n")

## tidy up the Tweets and send each to the AYLIEN Text API
        for c, result in enumerate(results, start=1):
            tweet = result.text
            tidy_tweet = tweet.strip().encode('ascii', 'ignore')
            cleanedtweets = cleanTweets(tidy_tweet)

            if len(tweet) == 0:
                print('Empty Tweet')
                continue

       # response = client.Sentiment({'text': tidy_tweet})
            csv_writer.writerow({
                'Tweet': cleanedtweets
           # 'Sentiment': response['polarity']
            })

            print("Analyzed Tweet {}".format(c))

    


def cleanTweets(raw_review):
    review =raw_review
    # review = re.sub('[^a-zA-Z]', ' ',review)
    result = re.sub(r"http\S+", "", review) #removing URLs
    result = re.sub(r'@[A-Za-z0-9]+','',result) #removing @
    # result = re.sub("[^a-zA-Z]", " ", result) #removing hashtags
    review = result.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    return (' '.join(review))

def train(filename):
    # train_data = pd.read_csv('imdbtest.csv')
    # print('read data')
    # Sentiment_words=[]
    # for row in train_data['polarity']:
    #     if row ==0:
    #         Sentiment_words.append('negative')
    #     elif row == 1:
    #         Sentiment_words.append('positive')
    # train_data['Sentiment'] = Sentiment_words
    # print('converted polarity to sentiment')
    # lengthtrain = len(train_data['Sentiment'])
    # corpus= []
    # for i in range(0, lengthtrain):
    #     if i%1000==0:
    #         print('reviewing %d of the text' % i)
    #     corpus.append(cleanTweets(train_data['text'][i]))
    # print('cleaned data')
    # train_data['new_Phrase']=corpus
    # cv = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
    # messages_bow=cv.fit_transform(train_data['new_Phrase'])

    # print('Shape of Sparse Matrix: ', messages_bow.shape)
    # print('Amount of Non-Zero occurences: ', messages_bow.nnz)

    # mnb = MultinomialNB()

    # spam_detect_model = mnb.fit(messages_bow,train_data['Sentiment'])

    # print('training done')
    # print(spam_detect_model)

    # f = open('my_classifier.pickle', 'wb')
    # pickle.dump(spam_detect_model, f)
    # f.close()

    # g = open('cv.pickle', 'wb')
    # pickle.dump(cv, g)
    # g.close()

    print('created classifier')

    f = open('my_classifier.pickle', 'rb')
    classifier = pickle.load(f)

    g = open('cv.pickle', 'rb')
    cv = pickle.load(g)
    

    print('opened classifier')
   

    print('in test data function')
    test_data = pd.read_csv(filename)
    clean_test_reviews = []
    data_set_length=len(test_data['Tweet'])
    for i in xrange(0,data_set_length):
        if((i+1)%10==0):
            print ('Review %d of %d\n'%(i+1,data_set_length))
        clean_reviews = cleanTweets(test_data['Tweet'][i])
        clean_test_reviews.append(clean_reviews)
    print(clean_test_reviews)
    print('after reading')
    # cv = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
    print('vectorizer done')
    test_data_features = cv.transform(clean_test_reviews)
    print('cv.transform done')
    result = classifier.predict(test_data_features)
    f.close()
    print(result)
    test_data['New_Sentiment']= result
    lentest = len(test_data)
    pos = len(test_data[test_data['New_Sentiment']=='positive'])
    neg = len(test_data[test_data['New_Sentiment']=='negative'])
    pos_percent = (pos*100)/lentest
    neg_percent = (neg*100)/lentest
    print('positive percentage is: %d' %pos_percent)
    print('negative percentage is: %d' %neg_percent)
    # counts = test_data["New_Sentiment"].value_counts()
    # plt.bar(range(len(counts)), counts)
    # plt.show()
    l=[]
    l=[pos_percent,neg_percent]
    print (l)
    return l
    # return spam_detect_model

# def testData(filename,obj):
#     print('in test data function')
#     test_data = pd.read_csv(filename)
#     clean_test_reviews = []
#     data_set_length=len(test_data['Tweet'])
#     for i in xrange(0,data_set_length):
#         if((i+1)%10==0):
#             print ('Review %d of %d\n'%(i+1,data_set_length))
#         clean_reviews = cleanTweets(test_data['Tweet'][i])
#         clean_test_reviews.append(clean_reviews)
#     print(clean_test_reviews)
#     print('after reading')
#     cv = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
#     print('vectorizer done')
#     test_data_features = cv.transform(clean_test_reviews)
#     print('cv.transform done')
#     result = obj.predict(test_data_features)
#     print(result)
#     test_data['New_Sentiment']= result
#     lentest = len(test_data)
#     pos = len(test_data[test_data['New_Sentiment']=='positive'])
#     neg = len(test_data[test_data['New_Sentiment']=='negative'])
#     pos_percent = (pos*100)/lentest
#     neg_percent = (neg*100)/lentest
#     print('positive percentage is: %d' %pos_percent)
#     print('negative percentage is: %d' %neg_percent)



    # train_data = pd.read_csv('train.tsv',sep='\t')

    # corpus= []
    # for i in range(0, 156060):
    #     corpus.append(cleanTweets(train_data['Phrase'][i]))

    # train_data['new_Phrase']=corpus

    # train_data.drop(['Phrase'],axis=1,inplace=True)
    # msg_train, msg_test, label_train, label_test = \
    # train_test_split(train_data['new_Phrase'], train_data['Sentiment'], test_size=0.3,random_state=101)
    
    # pipeline = Pipeline([
    # ('bow', CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)),  # strings to token integer counts
    # ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    # ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    # ])
    # print('after pipeline')

    # pipeline.fit(test_data['Tweets'])

    # predictions = pipeline.predict(test_data['Tweets'])

    # # a = classification_report(predictions,label_test)

    # return predictions

# def testData(file_name):
#     testtweet = file_name
#     print(testtweet)
#     test_data = pd.read_csv(testtweet)

# colors = ['green', 'red', 'grey']
# sizes = [positive, negative, neutral]
# labels = 'Positive', 'Negative', 'Neutral'

# ## use matplotlib to plot the chart
# plt.pie(
#    x=sizes,
#    shadow=True,
#    colors=colors,
#    labels=labels,
#    startangle=90
# )

# plt.title("Sentiment of {} Tweets about {}".format(number, query))
# plt.show()

# @app.route('/<int:pos_percent>/<int:neg_percent>')
def plot(values,user_input):
    print('in plot get')
    xs = range(1,3)
    # newlist=[]
    # for i in values:
    #     newlist.append(values[i])

    ys = [values[0],values[1]]
    plt.bar(xs,ys)
    plt.xlabel('Sentiment')
    plt.ylabel('Percentage')
    yaxis=['Positive','Negative']
    plt.xticks(xs,yaxis)
    #plt.color('r')
    #plt.color('g')
    plt.savefig('E:/pendrive/practicum 3/practicum 3/practicum/static/bar.png')
    plt.title('Twitter Sentiment Analysis for %s'% user_input)

    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue()).decode('ascii')
    result = figdata_png
    print('generated result')
    print(result)
    return result
    #     except:
    #         print('in image except block')
    #         return render_template('test2.html',result=result)
    # else:
    #     print('in plot post')

    # return render_template('test2.html',result=result)

    # axis.plot(xs, ys)



    # canvas = FigureCanvas(fig)
    # output = StringIO.StringIO()
    # canvas.print_png(output)
    # response = make_response(output.getvalue())
    # response.mimetype = 'image/png'
    # return response


    # test_data = pd.read_csv(filename)
    # counts = t["candidate"].value_counts()
    # plt.bar(range(len(counts)), counts)
    # plt.show()

    # fig = Figure()
    # axis = fig.add_subplot(1, 1, 1)





app.secret_key='super secret key'
if __name__=='__main__':
    app.debug=True
    app.run(host='0.0.0.0', port=5000)

