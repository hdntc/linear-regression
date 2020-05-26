import time
import tweepy
import read_data
from numpy import asfarray, array


class TwitterScraper:

    def __init__(self):
        self.consumer_key = "WO4lMtYzIqVtDBQQJ77HL5ph2"
        self.consumer_secret = "VKgees6GpkXaC5b1AEEYmLE9DUjq0QMFeQlmeyuZ5mrQbStU4w"
        self.access_token = "1261117480539312128-ZpWiOO5m77SZbAIXg7892uIX29bRmH"
        self.acess_token_secret = "pEHitI5oguo0TsA1kXiUU3atjp7Sb2lG7BtS87CgwisVX"

        auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        auth.set_access_token(self.access_token, self.acess_token_secret)
        self.api = tweepy.API(auth, wait_on_rate_limit=False)

    def scrape_user(self, user, count):
        tweets = []
        try:
            for tweet in self.api.user_timeline(id=user, count=count):
                tweets.append(tweet)

        except BaseException as e:
            print('failed on_status,', str(e))
            time.sleep(3)

        return tweets


def remove_retweets(tweets):
    new_tweets = []
    for tweet in tweets:
        if not tweet.retweeted:
            new_tweets.append(tweet)
    return new_tweets

def format_tweets_as_training_data(tweets):
    """take list of tweets and form design matrix and responses based on these critera:
    responses: # of likes
    predictors: retweet count, media(boolean), reply(boolean), quote tweet(boolean)
    """
    responses = ([])
    training_data = ([])

    for tweet in tweets:
        responses.append(tweet.favorite_count)
        training_data.append([tweet.retweet_count, int('media' in tweet.entities),
                              int(tweet.in_reply_to_screen_name!=None), int(tweet.is_quote_status)])


    training_data = read_data.add_1s_column(training_data)

    data = asfarray(training_data, float), asfarray(asfarray(responses, float).reshape(-1,1),float)
    return data