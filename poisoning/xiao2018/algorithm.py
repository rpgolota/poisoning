import numpy as np
import sklearn.linear_model

# sklearn.linear_model.Ridge
# sklearn.linear_model.Lasso
# sklearn.linear_model.ElasticNet

def classifier(weights, biases, x):
    """Linear classifier 

    Args:
        weights (column vector): self-explanatory
        biases (column vector): self-explanatory
        x (column vector): column vector of features of data point

    Returns:
        column vector: weighted data point
    """
    return np.transpose(weights)*x - biases

def loss(classified, expected):
    """Quadratic loss function

    Args:
        classified (column vector): optput of linear classifier function
        y (column vector): expected input

    Returns:
        column vector: loss between classified result and expected
    """
    return (1/2) * np.square(classified - y)

def xiao2018():
    return 'Not implemented'