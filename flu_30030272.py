import numpy as np
from pytz import timezone
import pytz
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import nltk
import re
import math
import datetime
from datetime import datetime, timedelta, tzinfo
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from nltk import *
import cls_flu_tweets
import pickle

# Defining Modules---------------------------------------------------------------------
# Cleaning Docs(Tweets)-----------------------------------------------------------------

def clean_data():
    porter = PorterStemmer()
    f_text=[] 
    for te in text:
    # Remove all the special characters
        doc = re.sub(r'\W', ' ', te)

    #Remove all punctuations
        doc=re.sub(r'[^\w\s]',' ',doc)
    
    # Substituting multiple spaces with single space
        doc = re.sub(r'\s+', ' ', doc, flags=re.I)
        
    # Remove numbers from the start
        doc = ''.join([w for w in doc if not w.isdigit()])

    # Converting to Lowercase
        doc = doc.lower()

    # Lemmatization
        doc = doc.split()
    
    #Stemmer
        doc = [porter.stem(word) for word in doc]
        doc = ' '.join(doc)
    
        f_text.append(doc)
    return f_text

# Analyzing All Tweets with onegram, Bigram and Trigram----------------------------------------------------
def vectorization(clean_tweets):
    vectorizer = CountVectorizer(ngram_range=(1,3),max_features=300,stop_words=stopwords.words('english'))
    numeric_features = vectorizer.fit_transform(clean_tweets).toarray()
    tfidfconverter = TfidfTransformer()
    numeric_features = tfidfconverter.fit_transform(numeric_features).toarray()
    return numeric_features

#Saving Results in a Text File--------------------------------------------------------------
def save(filename):
    f = open('labels_' + filename + '.txt', 'w')
    for label in fl.labels:
        f.write(label)
        f.write('\n')
    f.close()

# Creat a Plot for Predictions--------------------------------------------------------------------------------
def plot():
    datetimepercent=[]
    for date in date_time_set:
        date_time=[]
        label_date=[]
        date_time=[i for i, e in enumerate(dt) if e == date]
        label_date=[e for i, e in enumerate(y_pred_RF) if i in date_time]
        datetimepercent.append(label_date.count(1)*100/(float(len(date_time))))
    datetimepercent =  [round(x,2) for x in datetimepercent]
    timeseries= pd.DataFrame({'pos%':datetimepercent, 'date':date_time_set})
    timeseries=timeseries.sort_values(by=['date'])
    timeseries.index = timeseries['date']
    del timeseries['date']
    ax=timeseries.plot(rot=90)
    ax.set_xlabel("Time Sequence for Positive Tweets", fontsize='larger')
    ax.set_ylabel("Percentage of Positive Tweets", fontsize='larger')
    ax.set_title("Surveillance Plot for Positive Tweets", fontsize='x-large')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

    
#Main Codes Using Modules--------------------------------------------------------------------------------------
#Reading Tweets from Text File -----------------------------------------------------------------


fl=cls_flu_tweets.flu_tweets()
fl.load("tweets.txt")
text= []
dt=[]
for tweet in fl.tweets:
    tw= json.loads(tweet)['text'].encode('ascii',errors='ignore')
    d=datetime.strptime(json.loads(tweet)['created_at'],
                        '%a %b %d %H:%M:%S +0000 %Y').replace(tzinfo=pytz.utc).astimezone(timezone('Canada/Mountain')).strftime("%b %d %Y")
    dt.append(d)
    text.append(tw)

date_time_set=list(dict.fromkeys(dt))


# Cleaning Docs(Tweets)-----------------------------------------------------------------
clean_tweets=clean_data()

# Analyzing All Tweets with onegram, Bigram and Trigram----------------------------------------------------
numeric_features=vectorization(clean_tweets)

# Load Trained Random Forest Model from pkl File-----------------------------------------------------------
pf=open("flu_classifier_30030272.pkl","rb")
Randomforest_model=pickle.load(pf)
pf.close()

# Prediction and Saving Results in a Text File--------------------------------------------------------------
y_pred_RF=Randomforest_model.predict(numeric_features)
for i in range(0,len(y_pred_RF)):
    if y_pred_RF[i]==1:
        fl.labels.append('pos')
    if y_pred_RF[i]==0:
        fl.labels.append('neg')
save("30030272")

# Creat a Plot for Predictions--------------------------------------------------------------------------------
plot()






