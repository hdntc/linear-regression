from read_data import read_data, add_1s_column
from numpy.linalg import inv
from numpy import dot, array, diag
from scipy.stats import t, ttest_1samp


class LinearRegression:
    def __init__(self):
        self.coefficients = array([])
        self.training_design = array([])
        self.training_response = array([])
        self.features = 0
        self.training_n = 0

    def fit(self, design: array, response: array):
        """Fit the linear regression model to the training data."""
        """The least-squares normal equation is used to estimate the coefficients."""

        self.coefficients = dot(inv(dot(design.T, design)), dot(design.T, response))
        self.training_design = design
        self.training_response = response
        self.features = len(design.T)
        self.training_n = len(design)

    def predict(self, predictors: array, design=False):
        """Use the coefficients to predict the response(s) for given values for the predictors"""
        """If predictors is in design-matrix format then design should be True"""
        if self.coefficients.size == 0:
            raise Exception("Cannot predict before fitting model")
        if design:
            return dot(predictors, self.coefficients)[0][0]
        else:
            return dot(add_1s_column(predictors), self.coefficients)[0][0]

    def calculate_r2(self, test_predictors: array, test_responses: array, design=False):
        """Compute the fraction of variance in the response explained by the predictors"""
        return 1 - self.calculate_rss(test_predictors, test_responses, design)/self.calculate_tss(test_responses)

    def calculate_rss(self, test_predictors: array, test_responses: array, design=False):
        """Compute the RSS for a given set of data. Set design=True if test_predictors is given as a design matrix"""
        if design:
            errors = test_responses - dot(test_predictors, self.coefficients)
        else:
            errors = test_responses - self.predict(test_predictors)

        return dot(errors.T, errors)

    def calculate_squared_rse(self):
        """Compute the squared RSE (Residual standard error), an estimate for the variance of e in Y = f(X) + e"""
        return self.calculate_rss(self.training_design, self.training_response, design=True) / (
                self.training_n - self.features - 1)

    def calculate_coefficient_ci(self, confidence_level):
        """Compute the confidence interval for the regression coefficients at the confidence_level level of confidence"""
        """confidence_level should be in (0.00, 1.00)"""
        """RSE2() * inv(dot(...)) returns the covariance matrix for the """
        """The confidence interval is derived from (Bj - Bjhat)/SE(Bjhat) ~ t_{n-p-1}"""

        if not 0 < confidence_level < 1:
            raise Exception("Invalid level of confidence; confidence_level must be between 0 and 1")
        t_value = t.ppf(confidence_level / 2 + 0.5, self.training_n - self.features - 1)
        coeff_se = (diag(self.calculate_squared_rse() * inv(dot(self.training_design.T, self.training_design)))) ** .5
        result = []

        for (i, se) in enumerate(coeff_se):
            lower = self.coefficients[i][0] - se * t_value
            upper = self.coefficients[i][0] + se * t_value
            result.append((lower, upper))

        return result

    def calculate_tss(self, test_response):
        """Calculates Total Sum of Squares for a set of testing responses"""
        deviations = test_response - test_response.mean()
        return dot(deviations.T, deviations)

    def calculate_f_statistic(self, design=False):
        """calculates F-statistic for a regression model"""
        """F-statistic should be greater than 1"""
        """An F-statistic close to 1 indicates that H0 is correct"""
        """Assumes that the data has already been fit"""
        rss = self.calculate_rss(self.training_design, self.training_response, design)
        return ((self.calculate_tss(self.training_response) - rss) / self.features) / ((rss) / (self.training_n - self.features - 1))

    def calculate_t_statistic(self):
        coeff_se = (diag(self.calculate_squared_rse() * inv(dot(self.training_design.T, self.training_design)))) ** .5
        return_array = []
        for i in range(len(coeff_se)):
            return_array.append(self.coefficients[i,0]/coeff_se[i])
        return return_array

    def calculate_p_values(self, t_statistics:array):
        df = self.training_n - self.features - 1;
        return_array = []
        for t_stat in t_statistics:
            return_array.append(t.sf(abs(t_stat),df))
        return return_array



if __name__ == "__main__":
    data = read_data("data.csv")
    model = LinearRegression()
    model.fit(data[0], data[1])
    print("Predict X1 = 10, X2 = 3: ", end=" ")
    print(model.predict(array([[10, 3]])))
    print("Minimized RSS: ", end=" ")
    print(model.calculate_rss(data[0], data[1], True))
    print("Squared RSE: ", end=" ")
    print(model.calculate_squared_rse())
    print("95% Confidence Interval: ", end=" ")
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
