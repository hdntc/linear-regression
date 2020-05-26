from read_data import read_data
import linear_regression as linear_regression
from numpy import array
import twitter_scraper


def data_test(training_data=None, read_from_file=True):
    model = linear_regression.LinearRegression()
    if read_from_file or training_data is None:
        training_data = read_data("data.csv")

    model.fit(training_data[0], training_data[1])
    print(model.coefficients)
    if read_from_file or training_data is None:
        print("Predict X1 = 10, X2 = 3: ", end=" ")
        print(model.predict(array([[10, 3]])))
    else:
        print("Predict X1 = 0, X2 = 0, X3 = 0, X4 = 1: ", end=" ")
        print(model.predict(array([[0, 0, 0, 1]])))
    print("Minimized RSS: ", end=" ")
    print(model.calculate_rss(training_data[0], training_data[1], True))
    print("Squared RSE: ", end=" ")
    print(model.calculate_squared_rse())
    print("99% Confidence Interval: ", end=" ")
    print(model.calculate_coefficient_ci(0.99))
    print("R^2: ", end=" ")
    print(model.calculate_r2(training_data[0], training_data[1], True))
    print("F-Statistic: ", end=" ")
    print(model.calculate_f_statistic(True))
    print(model.calculate_tss(training_data[1]))
    print("T-statics for the coefficients: ", end=" ")
    print(model.calculate_t_statistic())
    print("P-Values for the coefficients: ", end=" ")
    print(model.calculate_p_values(model.calculate_t_statistic()))
    print("Leverage Statistics for the training data: ")
    print(model.calculate_leverage_statistic())


if __name__ == "__main__":
    my_bot = twitter_scraper.TwitterScraper()
    tweets = my_bot.scrape_user('BarackObama', 300)
    tweets = twitter_scraper.remove_retweets(tweets)
    data = twitter_scraper.format_tweets_as_training_data(tweets)

    data_test(data, True)
    print("X0 = intercept, X1 = # of retweets, X2 = has_media, X3 = is_reply, X4 = is_quote_tweet")
