import sklearn

# sklearn.linear_model.Ridge
# sklearn.linear_model.Lasso
# sklearn.linear_model.ElasticNet

def classifier(weights, biases, x):
    pass

def loss(classified, y):
    """Quadratic loss function

    Args:
        classified (number): optput of linear classifier function
        y (number): other input of dataset

    Returns:
        number: loss number
    """
    return (1/2) * (classified - y)**2

def xiao2018():
    return 'Not implemented'