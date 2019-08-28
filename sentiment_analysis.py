# -*- coding: utf-8 -*-
"""
Created by Asaf Gazit
github address https://github.com/AsafGazit/sentiment-analysis

The following details the training of a NLTK classifiers for
tweets sentiment analysis. 

The classifiers are then used to predict a second dataset,
which contains the twitter timestamp, to examine if sentiment 
events may be derived using such classifiers.
"""

from nltk.corpus import stopwords 
from nltk.classify import NaiveBayesClassifier
from nltk.classify import MaxentClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import mean_squared_error
import pattern.en
import pandas as pd
import numpy as np
import re
import itertools
from pyampd.ampd import find_peaks
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# load data
print(" ** load ")

data_filename='training.1600000.processed.noemoticon.csv'
column_labels=['sentiment','id','timestamp','query','user','tweet']
data_df=pd.read_csv(data_filename, header=None, 
                    encoding = 'ISO-8859-1', names=column_labels) 

data_df.loc[data_df['sentiment']==0,'sentiment']='negative'
data_df.loc[data_df['sentiment']==4,'sentiment']='positive'

## preprocess
print(" ** preprocess ")

def preprocess_tweets(tweets_list):
    split_tweets = [re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet) for tweet in tweets_list]
    split_tweets = [re.sub('@[^\s]+',' ',tweet) for tweet in split_tweets]
    split_tweets = [re.sub('[^a-zA-Z0-9#]',' ',tweet) for tweet in split_tweets]
    sw_list=[re.sub("'","",stopword) for stopword in stopwords.words('english')]
    clean_tweets=[' '.join(np.array(tweet.lower().split(' '))[~np.isin(tweet.lower().split(' '),sw_list)]) for tweet in split_tweets]
    result=[' '.join([word for word in tweet.split(' ') if len(word)>2]) for tweet in clean_tweets]
    return result
#
preprocessed_data=preprocess_tweets(data_df['tweet'].values)
data_df['preprocessed']=preprocessed_data

# vectorize
print(" ** vectorize ")
# {feature:True} for classifiers

def word_feats(words):
        return dict([(word, True) for word in words])

data_df['dict_features']=[word_feats(preprocessed_tweet.split()) for preprocessed_tweet in preprocessed_data]

# split
print(" ** split ")

def pd_train_test_split(Xdf, test_size=0.2, randomstate=123):
    array_size=Xdf.shape[0]
    df = Xdf.copy().sample(frac=1,random_state=randomstate).reset_index(drop=True)
    idx_test=int(array_size*test_size)
    # returns test set df and training set df
    return df.iloc[:idx_test,:],\
           df.iloc[idx_test:,:]

test_df, training_df=\
    pd_train_test_split(data_df, test_size=0.2, randomstate=123)

train_Xy=[(wfeatures,sentiment) for wfeatures,sentiment in \
           zip(training_df['dict_features'].tolist(),training_df['sentiment'].tolist())]

test_Xy=[(wfeatures,sentiment) for wfeatures,sentiment in \
          zip(test_df['dict_features'].tolist(),test_df['sentiment'].tolist())]


# training
print(" ** training ")
    
NB_nltk_clf = NaiveBayesClassifier.train(train_Xy)

MaxEnt_nltk_clf=classifier = MaxentClassifier.train(train_Xy, max_iter = 10)

NB_nltk_clf.show_most_informative_features(10)
MaxEnt_nltk_clf.show_most_informative_features(10)

# test and report

# support functions
def classifier_predict(clf, testXy):
    test_predictions, test_labels=[],[]
    for sampleid in range(len(testXy)):
        test_predictions.append(clf.classify(testXy[sampleid][0]))
        test_labels.append(testXy[sampleid][1])
    return np.array(test_predictions), np.array(test_labels)

'''
plot_confusion_matrix function from scikit-learn.org

Citiation
---------
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

'''
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          savename=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()
    if(savename!=False):
        plt.savefig(savename)
    else:
        plt.show()

def classifier_report(labels, predictions , CMsavename=False):
    results = confusion_matrix(labels, predictions) 
    print('Confusion Matrix :')
    print(results) 
    accuracy=accuracy_score(labels, predictions)
    print('Accuracy Score :',accuracy) 
    print('Report : ')
    print(classification_report(labels, predictions))
    plot_confusion_matrix(results,target_names = ['positive', 'negative'],
                          savename=CMsavename)
    return accuracy, results

def baseline_prediction(sentence_list, pos_threshold=0):
    test_predictions = []
    for sampleid in range(len(sentence_list)):
#        pen_sentiment=pattern.en.sentiment()
        if(pattern.en.positive(sentence_list[sampleid], threshold=pos_threshold)):
            test_predictions.append('positive')
        else:
            test_predictions.append('negative')
    return test_predictions

bl_predictions = baseline_prediction(test_df.preprocessed.tolist())

NBtest_predictions, NBtest_labels = \
    classifier_predict(NB_nltk_clf, test_Xy)
MEtest_predictions, MEtest_labels = \
    classifier_predict(MaxEnt_nltk_clf, test_Xy)

blaccuracy, blresults = \
    classifier_report(NBtest_labels, bl_predictions, CMsavename='BL_clf_CM.png')
NBaccuracy, NBresults = \
    classifier_report(NBtest_labels, NBtest_predictions, CMsavename='NB_clf_CM.png')
MEaccuracy, MEresults = \
    classifier_report(MEtest_labels, MEtest_predictions, CMsavename='ME_clf_CM.png')

'''
Applying the sentiment classifier on tweets time series:
Exploring the US Airline Sentiment dataset
'''

#load
USairlines_filename='US_AirlineTweets.csv'
USairlines_df=pd.read_csv(USairlines_filename, encoding ='ISO-8859-1')

USairlines_df=USairlines_df[(USairlines_df['airline_sentiment']!='neutral') &
                            (USairlines_df['airline_sentiment_confidence'] > 0.9) ]
#preprocess
USairlines_preprocessed_data=preprocess_tweets(USairlines_df['text'].values)
#vectorize
USairlines_dict_features=[word_feats(preprocessed_tweet.split()) for \
                          preprocessed_tweet in USairlines_preprocessed_data]

USairlines_Xy=[(wfeatures,sentiment) for wfeatures,sentiment in \
          zip(USairlines_dict_features, USairlines_df['airline_sentiment'].tolist())]
#predict
NBustest_predictions, NBustest_labels = \
    classifier_predict(NB_nltk_clf, USairlines_Xy)
MEustest_predictions, MEustest_labels = \
    classifier_predict(MaxEnt_nltk_clf, USairlines_Xy)
#examine results
NBaccuracy, NBresults = \
    classifier_report(NBustest_predictions, NBustest_labels, CMsavename='NBusa_clf_CM.png')
MEaccuracy, MEresults = \
    classifier_report(MEustest_predictions, MEustest_labels, CMsavename='MEusa_clf_CM.png')

#create sentiment time series
USairlines_predictions_int_positive=np.zeros(len(USairlines_Xy))
USairlines_predictions_int_positive[np.where(NBustest_predictions=='positive')[0]]=1
USairlines_predictions_int_negative=np.zeros(len(USairlines_Xy))
USairlines_predictions_int_negative[np.where(NBustest_predictions=='negative')[0]]=1

USairlines_labels_int_positive=np.zeros(len(USairlines_Xy))
USairlines_labels_int_positive[np.where(NBustest_labels=='positive')[0]]=1
USairlines_labels_int_negative=np.zeros(len(USairlines_Xy))
USairlines_labels_int_negative[np.where(NBustest_labels=='negative')[0]]=1


USairlines_df['label_positive']=USairlines_labels_int_positive
USairlines_df['label_negative']=USairlines_labels_int_negative
USairlines_df['NBpredictions_int_positive']=USairlines_predictions_int_positive
USairlines_df['NBpredictions_int_negative']=USairlines_predictions_int_negative
USairlines_df['NBpredictions']=NBustest_predictions

#aggregate time series per hour
USairlines_df['timestamp']=pd.to_datetime(USairlines_df['tweet_created'])
USairlines_df=USairlines_df.sort_values(by='timestamp')
USairlines_df=USairlines_df.set_index(USairlines_df.timestamp, drop=True)

resampled_USairlines_df=USairlines_df.resample("60T").sum()
resampled_USairlines_df['ts']=resampled_USairlines_df.index

#locate peaks of negative sentiment on the time series 
pr_rolling_troughs=find_peaks(resampled_USairlines_df['NBpredictions_int_negative'].tolist())

timestamps=resampled_USairlines_df.iloc[pr_rolling_troughs]

#peak nm 5 : event
eventid=5
event_tweet_ids= np.where(np.logical_and(USairlines_df['timestamp'].values>
                                         timestamps.index.values[eventid],
                                         USairlines_df['timestamp'].values<
                                         timestamps.index.values[eventid]+
                                         np.timedelta64(1,'h')))

eventdf = USairlines_df.iloc[event_tweet_ids]

# event sentiment ratio
sentiment_df = eventdf['NBpredictions'].value_counts()/eventdf['NBpredictions'].count()
sentiment_argmax = sentiment_df.index[sentiment_df.values.argmax()]
sentiment_argmax = 'negative'
event_tweet_ids= np.where(np.logical_and(\
                          np.logical_and(\
                          USairlines_df['timestamp'].values>\
                                timestamps.index.values[eventid],
                          USairlines_df['timestamp'].values<\
                                timestamps.index.values[eventid]+np.timedelta64(1,'h')),
                         (USairlines_df['NBpredictions']==\
                                sentiment_argmax).values))[0]

print(sentiment_df)

event_words_list=np.array(USairlines_dict_features)[event_tweet_ids]
event_words=[list(event_words_list[x].keys()) for x in range(len(event_words_list))]
event_words=np.array([word for sublist in event_words for word in sublist])

# word clouds

# get tags '#'
hashsignmask = np.array([event_words[x].startswith('#')==
                         True for x in range(len(event_words))])
hash_tags = event_words[hashsignmask]

if(len(hash_tags)>0):
    wc = WordCloud(width=800,height=800, background_color='white',
                   min_font_size=10).generate(text=' '.join(hash_tags))
    wc.to_file("hashtags_cloud.jpg")

# wordcount 
words_clean = event_words[~hashsignmask]
wc = WordCloud(width=800,height=800, background_color='white',
               min_font_size=10).generate(text=' '.join(words_clean))
wc.to_file("words_clean_cloud.jpg")

# sentiment over time: predicted vs actual

fig, ax = plt.subplots(figsize=(14,6))
ax.plot(resampled_USairlines_df.index.tolist(),resampled_USairlines_df['label_positive'],\
        'g',alpha=0.8, label='Positive sentiment labels')
ax.plot(resampled_USairlines_df.index.tolist(),resampled_USairlines_df['NBpredictions_int_positive'],\
        'g--',alpha=0.8, label='Positive sentiment predictions')
ax.plot(resampled_USairlines_df.index.tolist(),resampled_USairlines_df['label_negative'],\
        'r',alpha=0.8, label='Negative sentiment labels')
ax.plot(resampled_USairlines_df.index.tolist(),resampled_USairlines_df['NBpredictions_int_negative'],\
        'r--',alpha=0.8, label='Negative sentiment predictions')
ax.scatter(timestamps.index.values[eventid],
           resampled_USairlines_df['NBpredictions_int_negative'][pr_rolling_troughs[eventid]],
           s=80,label='"Negative event" example')
plt.title('Positive and negative sentiments over time: predictions and ground truth')
plt.legend()
plt.tight_layout()
plt.savefig('sentiment_over_time.jpg')

# rmse
rms_neg = np.sqrt(mean_squared_error(resampled_USairlines_df['label_negative'].values,
                                     resampled_USairlines_df['NBpredictions_int_negative'].values))

print("rms_neg",rms_neg)

#eof