import nltk
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
import re
import string

print(len(twitter_samples.fileids()))
print(twitter_samples.fileids())
from collections import Counter

counts = Counter(twitter_samples.positive_tweets())
	
pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')
tweets = twitter_samples.strings('tweets.20150430-223406.json')


for tweet in pos_tweets[:10]:
	print (tweet)
	
	
def data_cleaning (tweet):
  tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
	tweet = re.sub(r'#', '', tweet)
	tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
	tweet_tkn = tokenizer.tokenize(tweet)
	clean_tweets = []
	for word in tweet_tkn:
	  clean_tweets.append(word)
	  
	return clean_tweets

test = "Hello this is #cool"

print(data_cleaning(test))
	  
