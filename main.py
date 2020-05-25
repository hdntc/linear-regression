from read_data import read_data
import linear_regression as linear_regression
from numpy import array
import twitter_scraper

def data_test():
    data = read_data("data.csv")
    model = linear_regression.LinearRegression()
    model.fit(data[0], data[1])
    print("Predict X1 = 10, X2 = 3: ", end=" ")
    print(model.predict(array([[10, 3]])))
    print("Minimized RSS: ", end=" ")
    print(model.calculate_rss(data[0], data[1], True))
    print("Squared RSE: ", end=" ")
    print(model.calculate_squared_rse())
    print("99% Confidence Interval: ", end=" ")
    print(model.calculate_coefficient_ci(0.99))
    print("R^2: ", end=" ")
    print(model.calculate_r2(data[0], data[1], True))
    print("F-Statistic: ", end=" ")
    print(model.calculate_f_statistic(True))
    print(model.calculate_tss(data[1]))
    print("T-statics for the coefficients: ", end=" ")
    print(model.calculate_t_statistic())
    print("P-Values for the coefficients: ", end=" ")
    print(model.calculate_p_values(model.calculate_t_statistic()))
    print("Leverage Statistics for the training data: ")
    print(model.calculate_leverage_statistic())
    print(model.coefficients)

if __name__ == "__main__":
    my_bot = twitter_scraper.TwitterScraper()
    tweets = my_bot.scrape_user("TouhouHomophobe",1)
    print(tweets)