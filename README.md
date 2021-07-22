# linear-regression
Implementation of multiple linear regression in Python

# How to use the program
Edit `data.csv` using a program of your choice. It should be in the following format:

*X11 X21 X31 Y1*

*X12 X22 X32 Y2*

*X13 X23 X33 Y3* 
etc, so that the first column gives the first independent variable, the second gives the 2nd independent variable, and the rightmost column gives the dependent variable.

# Purpose
The program returns certain useful statistics about your data set, including:
- The linear regresion of *Y* on *X1, X2, ...* (giving the least-squares hyperplane, using the normal equation), that is, it returns coefficients *B0, B1, ...* so that *Y = B0 + B1X1 + B2X2* and the sum of the squared error is minimised.
- 99% confidence intervals on *B0, B1, ...* (an interval which has a 99% probability of containing the true value for *B0*, since the value calculated is just an estimate)
- The R^2 value (gives the fraction of variation in Y explained by the regression)
- The F-Statistic (indicates whether the regression is significant)
- Leverage statistics for each point (a measure of how far away the point's predictors are from those of other points)
- Variance inflation factors for each predictor (an indicator of correlation between predictors)
