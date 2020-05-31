from read_data import add_1s_column, remove_1s_column
from numpy.linalg import inv
from numpy import dot, array, diag
import numpy as np
from scipy.stats import t, f


class LinearRegression:
    """This class contains data and methods necessary for performing linear regression analysis.

    This class supports both simple linear regression and multiple linear regression methods, but currently
    does not support interactions. Currently supported features are: simple and multiple linear fitting,
    RSS and RSE, R^2, t-statistics, p-values, and confidence intervals.
    """

    def __init__(self):
        self.coefficients = array([])
        self.training_design = array([])
        self.training_response = array([])
        self.features = 0
        self.training_n = 0

    def fit(self, design: array, response: array):
        """Fit the linear regression model to the training data.
        The least-squares normal equation is used to estimate the coefficients."""

        self.coefficients = dot(inv(dot(design.T, design)), dot(design.T, response))
        self.training_design = design
        self.training_response = response
        self.features = len(design.T)
        self.training_n = len(design)

    def predict(self, predictors: array, design=False):
        """Use the coefficients to predict the response(s) for given values for the predictors
        If predictors is in design-matrix format then design should be True"""
        return predict(predictors, self.coefficients, design)

    def calculate_r2(self, test_predictors: array, test_responses: array, design=False):
        """Compute the fraction of variance in the response explained by the predictors"""
        return calculate_r2(test_predictors, test_responses, self.coefficients, design)

    def calculate_rss(self, test_predictors: array, test_responses: array, design=False):
        """Compute the RSS for a given set of data. Set design=True if test_predictors is given as a design matrix"""
        return calculate_rss(test_predictors, test_responses, self.coefficients, design)

    def calculate_squared_rse(self):
        """Compute the squared RSE (Residual standard error), an estimate for the variance of e in Y = f(X) + e"""
        return calculate_squared_rse(self.training_design, self.training_response, self.coefficients,
                                     self.training_n - self.features - 1)

    def calculate_coefficient_ci(self, confidence_level):
        """Compute the confidence interval for the regression coefficients at the confidence_level level of confidence
        confidence_level should be in (0.00, 1.00)
        RSE2() * inv(dot(...)) returns the covariance matrix for the
        The confidence interval is derived from (Bj - Bjhat)/SE(Bjhat) ~ t_{n-p-1}"""
        return calculate_coefficient_ci(self.training_design, self.training_response, self.coefficients,
                                        confidence_level,
                                        self.training_n - self.features - 1)

    def calculate_tss(self, test_response: array):
        """Calculates Total Sum of Squares for a set of testing responses"""
        calculate_tss(test_response)

    def calculate_f_statistic(self, design=False):
        """calculates F-statistic for a regression model
        F-statistic should be greater than 1
        An F-statistic close to 1 indicates that H0 cannot be rejected"""
        return calculate_f_statistic(self.training_design, self.training_response, self.coefficients, design)

    def calculate_f_p_value(self, design=False):
        """The p-value in the hypothesis test for correlation between any of the variables and the response"""
        return calculate_f_p_value(self.training_design, self.training_response, self.coefficients, design)

    def calculate_t_statistic(self):
        """finds and returns an array of t-statistics for all coefficients"""
        return calculate_t_statistic(self.training_design, self.training_response, self.coefficients,
                                     self.training_n - self.features - 1)

    def calculate_p_values(self, t_statistics: array):
        """finds and returns an array of p-values for all coefficients"""
        return calculate_p_values(t_statistics, self.training_n - self.features - 1)

    def calculate_leverage_statistic(self):
        """Calculates the leverage statistics for training data"""
        return calculate_leverage_statistic(remove_1s_column(self.training_design))

    def calculate_vif_statistic(self):
        return calculate_vif(self.training_design,design=True)


def predict(predictors: array, coefficients: array, design=False):
    """Use the coefficients to predict the response(s) for given values for the predictors
    If predictors is in design-matrix format then design should be True"""
    if coefficients.size == 0:
        raise Exception("Cannot predict before fitting model")
    if design:
        return dot(predictors, coefficients)[0][0]
    else:
        return dot(add_1s_column(predictors), coefficients)[0][0]


def calculate_rss(test_predictors: array, test_responses: array, coefficients: array, design=False):
    """Compute the RSS for a given set of data. Set design=True if test_predictors is given as a design matrix"""
    if design:
        errors = test_responses - dot(test_predictors, coefficients)
    else:
        errors = test_responses - predict(test_predictors, coefficients, design)

    return dot(errors.T, errors)


def calculate_squared_rse(training_design: array, training_response: array, coefficients: array, df):
    """Compute the squared RSE (Residual standard error), an estimate for the variance of e in Y = f(X) + e"""
    return calculate_rss(training_design, training_response, coefficients, design=True) / (
        df)


def calculate_tss(test_response):
    """Calculates Total Sum of Squares for a set of testing responses"""
    deviations = test_response - test_response.mean()
    return dot(deviations.T, deviations)


def calculate_r2(test_predictors: array, test_responses: array, coefficients: array, design=False):
    """Compute the fraction of variance in the response explained by the predictors"""
    return 1 - calculate_rss(test_predictors, test_responses, coefficients, design) / calculate_tss(test_responses)


def calculate_coefficient_ci(training_design: array, training_response: array, coefficients: array, confidence_level,
                             df):
    """Compute the confidence interval for the regression coefficients at the confidence_level level of confidence
    confidence_level should be in (0.00, 1.00)
    RSE2() * inv(dot(...)) returns the covariance matrix for the
    The confidence interval is derived from (Bj - Bjhat)/SE(Bjhat) ~ t_{n-p-1}"""

    if not 0 < confidence_level < 1:
        raise Exception("Invalid level of confidence; confidence_level must be between 0 and 1")
    t_value = t.ppf(confidence_level / 2 + 0.5, df)
    coef_se = (diag(calculate_squared_rse(training_design, training_response, coefficients, df) * inv(
        dot(training_design.T, training_design)))) ** .5
    result = []

    for (i, se) in enumerate(coef_se):
        lower = coefficients[i][0] - se * t_value
        upper = coefficients[i][0] + se * t_value
        result.append((lower, upper))

    return result


def calculate_t_statistic(training_design: array, training_response: array, coefficients: array, df):
    """finds and returns an array of t-statistics for all coefficients"""
    coef_se = (diag(calculate_squared_rse(training_design, training_response, coefficients, df) * inv(
        dot(training_design.T, training_design)))) ** .5
    return_array = []
    for i in range(len(coef_se)):
        return_array.append(coefficients[i, 0] / coef_se[i])
    return return_array


def calculate_p_values(t_statistics: array, df):
    """finds and returns an array of p-values for all coefficients"""
    return_array = []
    for t_stat in t_statistics:
        return_array.append(t.sf(abs(t_stat), df))
    return return_array


def calculate_f_statistic(training_design: array, training_response: array, coefficients, design=False):
    """calculates F-statistic for a regression model
    F-statistic should be greater than 1
    An F-statistic close to 1 indicates that H0 cannot be rejected
    Assumes that the data has already been fit"""
    df = len(training_design) - len(training_design.T)
    rss = calculate_rss(training_design, training_response, coefficients, design)
    return ((calculate_tss(training_response) - rss) / (len(training_design.T) - 1)) / (
            rss / df)


def calculate_f_p_value(training_design, training_response, coefficients, design=False):
    """Find the p-value for the F statistic"""
    features = len(training_design.T) - 1
    n = len(training_design)

    return f.sf(calculate_f_statistic(training_design, training_response, coefficients, design), features - 1,
                n - features)


def calculate_leverage_statistic(training_design: array):
    """Calculates the leverage statistics for training data"""
    h = dot(training_design, dot(inv(dot(training_design.T, training_design)), training_design.T))
    return diag(h)


def calculate_vif(training_design: array, design=False):
    """Calculates VIF (Variation Inflation Factor) for each predictor"""
    design_matrix = remove_1s_column(training_design) if design else np.copy(training_design)
    vif_values = []
    col_size = np.shape(design_matrix)[1]
    for i in range(col_size):
        response = design_matrix[:,[i]]
        vif_model = LinearRegression()
        vif_design = np.delete(design_matrix,[i],1)
        vif_design = add_1s_column(vif_design)
        vif_model.fit(vif_design,response)
        r2 = vif_model.calculate_r2(vif_design,response,design)
        vif_values.append(1/(1-r2))

    return vif_values
