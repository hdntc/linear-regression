import time
import tweepy


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
            for tweet in self.api.user_timeline(user=user, count=count):
                print(tweet)
                tweets.append((tweet.text))

        except BaseException as e:
            print('failed on_status,', str(e))
            time.sleep(3)

        return tweets
