from csv import reader
from numpy import array, append, asfarray

def add_1s_column(matrix:array) -> array:
    """Add a one's column to the left-most column of the array"""
    return append(array([[1]*len(matrix)]).T, matrix, axis=1)

def read_data(data_path:str, delimiter=",", use_headers=False) -> array:
    """Read data from the .csv file. Return the design matrix and the response values"""
    """If there is an error in the .csv file (for example, a non-numerical entry)"""
    """use_headers refers to whether or not the .csv file uses headers (for example, x1,x2,x3,y on the first row)"""
    """It is assumed that the rightmost column in the .csv accommodates the response values"""

    with open(data_path, "r") as datafile:
        data_reader = reader(datafile, delimiter=delimiter)
        
        response = array([])
        
        rows = array([row for row in data_reader])
        rows = rows[1:] if use_headers else rows # If headers are used, omit them, otherwise do nothing
        features = len(rows[0])-1 # Every row has has values for X1, X2, ... , Xp and y, this is features+1 columns
                                  # index of the response vector in rows = features

        design_matrix = add_1s_column(rows[:,:features])
        
        return (asfarray(design_matrix, float), asfarray(rows[:,features].reshape((-1,1)), float))

