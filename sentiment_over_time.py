# -*- coding: utf-8 -*-
"""
Created by Asaf Gazit
github address https://github.com/AsafGazit/sentiment-analysis/blob/master/sentiment_over_time.py

Sentiment analysis over time
"""

import pandas as pd
import numpy as np
import pickle
import re
from pyampd.ampd import find_peaks
from wordcloud import WordCloud
import matplotlib.pyplot as plt


from nltk.classify import NaiveBayesClassifier
from nltk.stem import PorterStemmer 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


''' functions and definitions '''

ps = PorterStemmer() # stemmer to be used

def preprocess_tweets(filtered_df, labels=True):
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt  
    
    def word_feats(words):
        return dict([(word, True) for word in words])
    
    filtered_df.loc[:,'tidy_tweet'] = np.vectorize(remove_pattern)(filtered_df['tweet'], '@[\w]*')
    filtered_df.loc[:,'tidy_tweet'] = filtered_df['tidy_tweet'].str.replace('[^a-zA-Z0-9#]', ' ')
    filtered_df.loc[:,'tidy_tweet'] = filtered_df['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    tokenized_tweet = filtered_df['tidy_tweet'].apply(lambda x: x.split())
    tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x]) # stemming  
    tokenized_tweet = tokenized_tweet.apply(lambda x: word_feats(x)) # word features  
    filtered_df.loc[:,'tidy_tweet'] = tokenized_tweet
    if(labels==True):
        return [tuple(x) for x in filtered_df.loc[:,['tidy_tweet','sentiment']].values]
    else:
        return filtered_df.loc[:,'tidy_tweet'].values

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          savename=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import itertools

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


# prepare train data
traindata_filename='training.1600000.processed.noemoticon.csv'
column_labels=['polarity','id','timestamp','query','user','text']
traindata_df=pd.read_csv(traindata_filename, header=None, 
                         encoding = 'ISO-8859-1', names=column_labels) 

filtered_traindata_df=traindata_df.loc[:,['text','polarity']]
filtered_traindata_df.loc[filtered_traindata_df['polarity']==0,'polarity']='negative'
filtered_traindata_df.loc[filtered_traindata_df['polarity']==4,'polarity']='positive'
filtered_traindata_df.columns=['tweet','sentiment']
print(filtered_traindata_df.sentiment.value_counts())
traindata_processed_features = preprocess_tweets(filtered_traindata_df)

# train classifier
classifier = NaiveBayesClassifier.train(traindata_processed_features)

print(classifier.show_most_informative_features())

# test data
testdata_filename='testdata.manual.2009.06.14.csv'
testdata_df=pd.read_csv(testdata_filename, header=None,
                        encoding = 'ISO-8859-1', names=column_labels) 
testdata_df=testdata_df.loc[(testdata_df['polarity']!=2).values]
filtered_testdata_df=testdata_df.loc[(testdata_df['polarity']!=2).values,['text','polarity']]
filtered_testdata_df.loc[filtered_testdata_df['polarity']==0,'polarity']='negative'
filtered_testdata_df.loc[filtered_testdata_df['polarity']==4,'polarity']='positive'
filtered_testdata_df.columns=['tweet','sentiment']

testdata_processed_features = preprocess_tweets(filtered_testdata_df, labels=False)
# predict
test_predictions=[]
for sampleid in range(len(testdata_processed_features)):
    test_predictions.append(classifier.classify(testdata_processed_features[sampleid]))
test_predictions=np.array(test_predictions)

actual = filtered_testdata_df['sentiment'].values
predicted = test_predictions
results = confusion_matrix(actual, predicted) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(actual, predicted)) 
print('Report : ')
print(classification_report(actual, predicted))
plot_confusion_matrix(results,target_names = ['positive', 'negative'],
                      savename='test_CM.jpg')

# airline sentiment
USairlines_filename='US_AirlineTweets.csv'
USairlines_df=pd.read_csv(USairlines_filename, encoding ='ISO-8859-1')

USairlinescondition=USairlines_df[(USairlines_df['airline_sentiment']!='neutral') &
                                 (USairlines_df['airline_sentiment_confidence'] > 0.9) ]

filteredUSairlines=USairlinescondition.loc[:,['text','airline_sentiment']]
filteredUSairlines.columns=['tweet','sentiment']

USairlines_processed_features = preprocess_tweets(filteredUSairlines, labels=False)

USairlines_predictions=[]
for sampleid in range(len(USairlines_processed_features)):
    USairlines_predictions.append(classifier.classify(USairlines_processed_features[sampleid]))

actual = filteredUSairlines['sentiment'].values
predicted = np.array(USairlines_predictions)
results = confusion_matrix(actual, predicted) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(actual, predicted)) 
print('Report : ')
print(classification_report(actual, predicted))
plot_confusion_matrix(results,target_names = ['positive', 'negative'],
                      savename='USairlines_CF.jpg')

# prepare results for time plot (count)
USairlines_predictions=np.array(USairlines_predictions)
USairlines_predictions_int_positive=np.zeros(len(USairlines_predictions))
USairlines_predictions_int_positive[np.where(USairlines_predictions=='positive')[0]]=1
USairlines_predictions_int_negative=np.zeros(len(USairlines_predictions))
USairlines_predictions_int_negative[np.where(USairlines_predictions=='negative')[0]]=1

USairlinescondition.loc[:,'predictions']=USairlines_predictions
USairlinescondition.loc[:,'predictions_int_positive']=USairlines_predictions_int_positive
USairlinescondition.loc[:,'predictions_int_negative']=USairlines_predictions_int_negative

airline_sentiment_positive_int = np.zeros(len(USairlinescondition['airline_sentiment']))
airline_sentiment_positive_int[(USairlinescondition['airline_sentiment']=='positive').values]=1
airline_sentiment_negative_int = np.zeros(len(USairlinescondition['airline_sentiment']))
airline_sentiment_negative_int[(USairlinescondition['airline_sentiment']=='negative').values]=1

USairlinescondition.loc[:,'airline_sentiment_positive_int']=airline_sentiment_positive_int
USairlinescondition.loc[:,'airline_sentiment_negative_int']=airline_sentiment_negative_int

# timestamp handling
USairlinescondition.loc[:,'ts']=pd.to_datetime(USairlinescondition['tweet_created'])
sortedUSairlines_df=USairlinescondition.sort_values(by='ts')
sortedUSairlines_df = sortedUSairlines_df.set_index(sortedUSairlines_df['ts'])

sortedUSairlines_df=sortedUSairlines_df.resample("60T").sum()
sortedUSairlines_df['ts']=sortedUSairlines_df.index

# sentiment event 
prediction_troughs=find_peaks(sortedUSairlines_df['predictions_int_negative'].values)
timestamps=sortedUSairlines_df.iloc[prediction_troughs]
eventid=5 # fifth trough

# sentiment over time figure
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(sortedUSairlines_df['ts'],
        sortedUSairlines_df['airline_sentiment_positive_int'],'g',alpha=0.8
        , label='Positive sentiment labels')
ax.plot(sortedUSairlines_df['ts'],
        sortedUSairlines_df['predictions_int_positive'],'g--',alpha=0.8
        , label='Positive sentiment predictions')
ax.plot(sortedUSairlines_df['ts'],
        sortedUSairlines_df['airline_sentiment_negative_int'],'r',alpha=0.8
        , label='Negative sentiment labels')
ax.plot(sortedUSairlines_df['ts'],
        sortedUSairlines_df['predictions_int_negative'],'r--',alpha=0.8
        , label='Negative sentiment predictions')
ax.scatter(timestamps.index.values[eventid],
           sortedUSairlines_df['predictions_int_negative'][prediction_troughs[eventid]],
           s=100,label='"Negative event" example')
plt.title('Positive and negative sentiments over time: predictions and ground truth')
plt.legend()
plt.tight_layout()
plt.show()

# event tweets
event_tweet_ids= np.where(np.logical_and(USairlinescondition['ts'].values>
                                         timestamps.index.values[eventid],
                                         USairlinescondition['ts'].values<
                                         timestamps.index.values[eventid]+
                                         np.timedelta64(1,'h')))

eventdf = USairlinescondition.iloc[event_tweet_ids]

# event sentiment ratio
sentiment_df = eventdf['predictions'].value_counts()/eventdf['predictions'].count()
print('event sentiment predicted:')
print(sentiment_df)

sentiment_opposite = 'positive' # to be excluded
event_tweet_ids= np.where(np.logical_and(np.logical_and(USairlinescondition['ts'].values>timestamps.index.values[eventid],
                                         USairlinescondition['ts'].values<timestamps.index.values[eventid]+np.timedelta64(1,'h')),
                                         (USairlinescondition['predictions']!=sentiment_opposite).values))

# event words
stemmed_words_list=USairlines_processed_features[event_tweet_ids]
stemmed_words=[list(stemmed_words_list[x].keys()) for x in range(len(stemmed_words_list))]
stemmed_words=np.array([word for sublist in stemmed_words for word in sublist])

# hashtags wordcloud visualisation
hashsignmask = np.array([stemmed_words[x].startswith('#')==
                         True for x in range(len(stemmed_words))])
hash_tags = stemmed_words[hashsignmask]
wc = WordCloud(width=800,height=800, background_color='white',
               min_font_size=10).generate(text=' '.join(hash_tags))
wc.to_file("hashtags_cloud.jpg")

# tweet words wordcloud visualisation 
words_clean = stemmed_words[~hashsignmask]
wc = WordCloud(width=800,height=800, background_color='white',
               min_font_size=10).generate(text=' '.join(words_clean))
wc.to_file("words_clean_cloud.jpg")

#eof
