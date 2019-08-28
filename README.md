## Tweets sentiment analysis and “sentiment events” over time

### Overview
The following compares a two classifiers from NLTK package for tweets sentiment analysis. The classifiers are then applied to predict a second tweets dataset. The second dataset contains the tweet’s timestamp which is used to examine if sentiment “events” may be derived using such classifier.

### Datasets
1.The Sentiment140 dataset A collection of tweets and their associated sentiment labels ('positive'/'negative'). The training dataset includes 1,600,000 tweets and their assosiated sentiment label. http://help.sentiment140.com/
This dataset is used for training, validation and testing of classifiers.

2.Kaggle’s Twitter US Airline Sentiment dataset Analyze how travelers in February 2015 expressed their feelings on Twitter: A sentiment analysis job about the problems of each major U.S. airline. https://www.kaggle.com/crowdflower/twitter-airline-sentiment
This dataset will be used to examine sentiment over time.

### Classes
The Sentiment140 used contains 1,600,000 tweets labeled into two classes: 
positive and negative (sentiment). Each class contains 800,000 tweets.

Of the US airline dataset, only tweets with confidence higher than 0.9 and 'positive' or 'negative' classes (excluding 'neutral') are used. Confirming with those requirements are 8,908 tweets, of which 7,391 labeled negative and 1,517 positive.

### Preprocessing
This part aims to generalise features and reduce the dimentionality of the model. 
The following steps has been applied as part of the preprocessing:
1. Removing URLs
2. Removing Usernames
3. Removing characters that are not letters or numbers
4. Removing stopwords and lowercase words
5. Removing words that are less than 3 characters long

The following python function takes a twitter dataframe and applies the described preprocessing:

```python
def preprocess_tweets(tweets_list):
    split_tweets = [re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet) for tweet in tweets_list]
    split_tweets = [re.sub('@[^\s]+',' ',tweet) for tweet in split_tweets]
    split_tweets = [re.sub('[^a-zA-Z0-9#]',' ',tweet) for tweet in split_tweets]
    sw_list=[re.sub("'","",stopword) for stopword in stopwords.words('english')]
    clean_tweets=[' '.join(np.array(tweet.lower().split(' '))[~np.isin(tweet.lower().split(' '),sw_list)]) for tweet in split_tweets]
    result=[' '.join([word for word in tweet.split(' ') if len(word)>2]) for tweet in clean_tweets]
    return result
```

### Processing
For NLTK classifiers, each observation should be formalised as a tuple with features {'feature_word':True} and the sentiment label.

```python
def word_feats(words):
        return dict([(word, True) for word in words])
```
And
```python
[word_feats(preprocessed_tweet.split()) for preprocessed_tweet in preprocessed_data]
```

NLTK classifiers and baseline

Two classifiers from the NLTK package are trained and tested:
-Naive Bayes
-Maximum Entropy

The baseline reference will be a sentiment classifier for text (not tweets) from the Pattern.en package.

### Splitting
0.2 : test set = 320,000 tweets.
0.8 : train set = 1,280,000‬ tweets.
1 : total of 1,600,000 tweets

### NLTK classifiers and baseline

Two classifiers from the NLTK package are trained and tested:
-Naive Bayes
-Maximum Entropy

The baseline reference is the sentiment classifier for text (not tweets) from the Pattern.en package.

### Results

<img src="https://github.com/AsafGazit/sentiment-analysis/blob/master/img/resultsbar.PNG" alt="Bar chart accuracy" width="45%" height="30%">

The trained classifiers perform similarly (76%-77% accuracy) and well above the baseline classifier.

### Confusion matrix

Naive Bayes on test set

<img src="https://github.com/AsafGazit/sentiment-analysis/blob/master/img/NB_clf_CM.png" alt="Bar chart accuracy" width="60%" height="60%">

```
              precision    recall  f1-score   support

    negative       0.74      0.81      0.77    160046
    positive       0.79      0.72      0.75    159954

   micro avg       0.76      0.76      0.76    320000
   macro avg       0.76      0.76      0.76    320000
weighted avg       0.76      0.76      0.76    320000
```

Most informative features:
```
                 sadface = True           negati : positi =     69.7 : 1.0
                     447 = True           negati : positi =     51.8 : 1.0
                  farrah = True           negati : positi =     44.7 : 1.0
                     os3 = True           negati : positi =     41.7 : 1.0
                    owie = True           negati : positi =     41.7 : 1.0
             shareholder = True           positi : negati =     41.0 : 1.0
                 unloved = True           negati : positi =     37.8 : 1.0
              recommends = True           positi : negati =     35.0 : 1.0
                 saddens = True           negati : positi =     32.6 : 1.0
                fuzzball = True           positi : negati =     32.6 : 1.0
```

Maximum Entropy on test set

<img src="https://github.com/AsafGazit/sentiment-analysis/blob/master/img/ME_clf_CM.png" alt="Bar chart accuracy" width="60%" height="60%">

```
              precision    recall  f1-score   support

    negative       0.77      0.77      0.77    160046
    positive       0.77      0.77      0.77    159954

   micro avg       0.77      0.77      0.77    320000
   macro avg       0.77      0.77      0.77    320000
weighted avg       0.77      0.77      0.77    320000
```

Most informative features:
```
  -5.265 sadface==True and label is 'positive'
  -4.839 triste==True and label is 'positive'
   4.526 #sickfriday==True and label is 'negative'
  -4.404 fuzzball==True and label is 'negative'
  -4.368 whyyyyy==True and label is 'positive'
   4.329 prowd==True and label is 'positive'
  -4.193 fome==True and label is 'positive'
  -4.191 sadd==True and label is 'positive'
  -4.138 backache==True and label is 'positive'
  -4.118 tummyache==True and label is 'positive'
```

Baseline (Pattern.en)

<img src="https://github.com/AsafGazit/sentiment-analysis/blob/master/img/BL_clf_CM.png" alt="Bar chart accuracy" width="60%" height="60%">

```
              precision    recall  f1-score   support

    negative       0.76      0.30      0.43    160046
    positive       0.56      0.91      0.70    159954

   micro avg       0.60      0.60      0.60    320000
   macro avg       0.66      0.60      0.56    320000
weighted avg       0.66      0.60      0.56    320000
```

The baseline classifier seem to have a tendency to predict a negative sentiment. The overall accuracy is 60.4%. Not much better than the 50% of random choice.


### Applying the sentiment classifier on tweets time series : exploring the US Airline Sentiment dataset

The US Airlines Sentiment dataset is also labeled for sentiment. 
The Naive Bayes and Maximum Entropy trained classifiers prediction accuracy on the dataset are 87.91% and 83.6%, respectively. Those scores are higher than found testing the test set. This may be as people tend to be expressive when tweeting in regards to airlines, especially when complaining or when something goes wrong.  

Naive Bayes on US Airline Sentiment

<img src="https://github.com/AsafGazit/sentiment-analysis/blob/master/img/NBusa_clf_CM.png" alt="confusion matrix" width="60%" height="60%">

Maximum Entropy on US Airline Sentiment

<img src="https://github.com/AsafGazit/sentiment-analysis/blob/master/img/MEusa_clf_CM.png" alt="confusion matrix" width="60%" height="60%">

Now, lets plot the sentiment over time using the timestamp.
The following plot shows the twitter sentiment over the time (tweet timestamp). It details the actual and predicted labels counts per hour.

<img src="https://github.com/AsafGazit/sentiment-analysis/blob/master/img/sentiment_over_time.jpg" alt="confusion matrix" width="80%" height="80%">

The labels and the classifier predictions trends over time looks very similar, which is somewhat expected at a 87.53% accuracy rate (RMSE=4.808 rooted error tweets/hour). 
The predicted and actual sentiment count over time show the daily seasonality of tweets. Those seem to be correlated with daytime, when most people tweet and travel. 
The plot also shows a spike in the negative sentiment between the 22nd and the 23rd of February (marked in the previous figure). During this spike, the sentiments ratio is 0.833333 (negative) to 0.166667 (positive) or, in other words, 5:1 negative to positive tweets ratio.

To explore the sentiment spike related tweets without reading multiple tweets, I extract the tweets related and produce two word clouds: one for the hashtags and one for the tweets' content.

<img src="https://github.com/AsafGazit/sentiment-analysis/blob/master/img/hashtags_cloud.jpg" alt="confusion matrix" width="30%" height="30%">

This hashtags cloud gives indication that two airlines are associated with this negative tweet surge: Jet Blue and United Airlines. 

<img src="https://github.com/AsafGazit/sentiment-analysis/blob/master/img/words_clean_cloud.jpg" alt="confusion matrix" width="30%" height="30%">

This tweet words cloud gives indication that flights were delayed and/or canceled. It gives an indication to a temporal disturbance of some sort.

### Summary: Sentiment over time

The application of a sentiment classifier seem to be applicable to highlight sentiment-involved social media events. Such an application may be useful to monitor social media and to improve any reaction to an event as it is happening. This may also be useful to recognise opportunities when such sentiment driven events are assosiated with a business competitor.

The code for the described above can be found [here.](https://github.com/AsafGazit/sentiment-analysis/blob/master/sentiment_analysis.py)
