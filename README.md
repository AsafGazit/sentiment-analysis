## Tweets sentiment analysis and “sentiment events” over time

### Overview
The following details the training of a Naive Bayes classifier for tweets sentiment analysis. The classifier is then used on a second dataset, which contains the twitter timestamp, to examine if sentiment events may be derived using such classifier.

### Datasets
1. The Sentiment140 dataset
A collection of tweets and their associated sentiment labels ('positive'/'negative'). 
http://help.sentiment140.com/
The training dataset includes 1600000 tweets and their assosiated sentiment label.

2. Kaggle’s Twitter US Airline Sentiment dataset
Analyze how travelers in February 2015 expressed their feelings on Twitter: A sentiment analysis job about the problems of each major U.S. airline.
https://www.kaggle.com/crowdflower/twitter-airline-sentiment
The tweets used for testing have confidence (of the label assosiated) higher than 0.9 and the classifier recognised classes ('positive' or 'negative', excluding 'neutral').
The dataset includes 8908 tweets (that confirm with the requirements), of which 7391 labeled negative and 1517 positive.

### Preprocessing
This part aims to generalise features and reduce the dimentionality of the model. 
The following steps has been applied as part of the preprocessing:
1. Removal of Twitter handles (usernames).
2. Removal of non-alphanumeric characters.
3. Removal of short words (less than 4 characters).
4. Stemming.
5. Casting to a feature vector (for NB classifier training: tuple with features {'feature_word':True} and the sentiment label).

The following python function takes a twitter dataframe and applies the described method:

```python
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
```

### Training and testing the classifier

The classifier trained is NLTK's Naive Bayes classifier.
```python
from nltk.classify import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(traindata_processed_features)
```

The test set contains 359 samples, of which:
182 have positive sentiment labels and 177 have negative sentiment labels.

The prediction accuracy of the classifier on test set is 78.55% (score: 0.7855153203342619).

<img src="https://github.com/AsafGazit/sentiment-analysis/blob/master/img/confusion_matrix_plot.jpg" alt="confusion matrix" width="60%" height="60%">

For the test dataset, the classifier tend to predicts positive sentiments. Overall, not a bad score result.

Most Informative Features:
```
              tweeteradd = True           positi : negati =    553.7 : 1.0
                dividend = True           positi : negati =     57.0 : 1.0
                  sadfac = True           negati : positi =     40.1 : 1.0
                 fuzzbal = True           positi : negati =     39.8 : 1.0
                  farrah = True           negati : positi =     39.0 : 1.0
                 whyyyyi = True           negati : positi =     38.3 : 1.0
                     owi = True           negati : positi =     34.6 : 1.0
               sharehold = True           positi : negati =     33.4 : 1.0
                    #tag = True           negati : positi =     32.3 : 1.0
                  boohoo = True           negati : positi =     30.5 : 1.0
```

Perhaps not so surprising to find that the most positive informative indication is assosiated with someone adding you ('tweeteradd'), with a positive to negative ratio of 553.7 to 1.0. On the other side, 'sadfac'(sadface) is the most informative negative feature with a ratio of 40 to 1.

### Exploring the US Airline Sentiment dataset

The prediction accuracy of the classifier on US Airline Sentiment dataset is 87.53% (score: 0.8753929052537045). A higher accuracy score than the test set accuracy. This may be as people tend to be expressive when tweeting in regards to airlines, especially when complaining or when something goes wrong.  

<img src="https://github.com/AsafGazit/sentiment-analysis/blob/master/img/confusion_matrix_plot_usa.jpg" alt="confusion matrix" width="60%" height="60%">

Now, lets plot the sentiment over time using the timestamp available in this dataset.
The following plot shows the twitter sentiment over the time (tweet timestamp). It details the actual and predicted labels counts per hour.

<imgsrc="https://github.com/AsafGazit/sentiment-analysis/blob/master/img/sentiment_over_time.jpg" alt="tweets over time" width="60%" height="60%">

The labels and the classifier predictions trends over time looks very similar, which is somewhat expected at a 87.53% accuracy rate (RMSE=5.098). 
The predicted and actual sentiment count over time show the daily seasonality of tweets. Those seem to be correlated with daytime, on which most people tweet and travel. 
The plot also shows a spike in the negative sentiment between the 22nd and the 23rd of February (marked in the previous figure).

To explore the sentiment spike related tweets without reading multiple tweets, I extract the tweets related and produce two word clouds: one for the hashtags and one for the tweets' content.

<imgsrc="https://github.com/AsafGazit/HDTW/blob/master/img/hashtags_cloud.jpg" alt="hashtag word could" width="60%" height="60%">

This hashtags cloud gives indication that two airlines are associated with this negative tweet surge: Jet Blue and United Airlines. 

<imgsrc="https://github.com/AsafGazit/HDTW/blob/master/img/words_clean_cloud.jpg" alt="tweets word could" width="60%" height="60%">

This tweet words cloud gives indication that flights were delayed and/or canceled. It gives an indication to a temporal disturbance of some sort.

### Summary: Sentiment over time

The application of a sentiment classifier seem to be applicable to highlight sentiment-involved social media events. Such an application may be useful to monitor social media and to improve operations reaction to an event as it is happening. This may also be useful to recognise opportunities when such sentiment driven events are assosiated with a business competitor.

