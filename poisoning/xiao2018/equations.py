from sklearn import linear_model
import numpy as np

"""
:: Notes for equation2 ::

- equation3 seems to be very similar, just maximizing, and before doing anything further I think it is worthwile
  to read fully and compile all next steps needed to start implementing more of the features from the paper

- obj provides a way to directly set the algorithm, and it is returned to provide a way to use it again
  a better way to do this would to make this a class, but that is when we decide how we will be
  integrating this with all the other components
  This might not be needed in the future

- might be good to implement a parameter input to directly pass onto the algorithms

- might need to specify solver type
- coef_ seems to get weights
- intercept_ seems to get the bias term

"""
def equation2(X, Y, **kwargs):
    
    """Equation 2 in paper. Implements LASSO, Ridge Regression, or Elastic Net

    Parameters:
        X: Data
        Y: Results
        Arguments:
            type : (str) : type of algorithm to use ::: {'lasso', 'ridge', 'elastic'}
                default => 'lasso'
            alpha : (number) : alpha to be used in linear model by sklearn
                default => 1.0
            object : (sklearn linear model) : use this already created object for fitting
                default => None
            return_object : (bool) : return the used object for fitting
                default => False unless object parameter is not None

    Raises:
        TypeError: Unknown parameters
        TypeError: Invalid algorithm type

    Returns:
        (W,B): Returns a 2 size tuple of the weights and biases as found by the algorithm
        (W,B,O): returns a 3 size tuple of the weights and biases and linear model as last member
    """
    
    algorithm = kwargs.pop('type', 'lasso')
    aplh = kwargs.pop('alpha', 1.0)
    obj = kwargs.pop('object', None)
    ret_obj = kwargs.pop('return_object', obj is not None)
    
    if len(kwargs):
        raise TypeError('Unknows parameters: ' + ', '.join(kwargs.keys()))
    
    if algorithm == 'lasso':
        algorithm = linear_model.Lasso(alpha=aplh)
    elif algorithm == 'ridge':
        algorithm = linear_model.Ridge(alpha=aplh)
    elif algorithm == 'elastic':
        algorithm = linear_model.ElasticNet(alpha=aplh)
    else:
        raise TypeError(f'Invalid algorithm type provided: {algorithm}')
    
    if obj is not None:
        algorithm = obj
    
    algorithm.fit(X, Y)
    
    if ret_obj:
        return algorithm.coef_, algorithm.intercept_, algorithm
    else:
        return algorithm.coef_, algorithm.intercept_

def equation7(X, Y, weights, biases, **kwargs):
    
    algorithm = kwargs.pop('type', 'lasso')
    aplh = kwargs.pop('alpha', 1.0)
    rho = kwargs.pop('rho', 0.5)
    
    X = np.array(X)
    Y = np.array(Y)
    weights = np.array(weights)
    biases = np.array(biases)
    
    n = len(X)
    v = None
    
    if algorithm == 'lasso':
        v = 0
    elif algorithm == 'ridge':
        v = np.identity(n)
    elif algorithm == 'elastic':
        v = (1 - rho) * np.identity(n)
    else:
        raise TypeError(f'Invalid algorithm type provided: {algorithm}')

    epsilon = sum([np.dot(i, i) for i in X]) / n
    mu = sum(X) / n
    M = np.dot(X, weights) + ((np.dot(weights, X) + biases) - Y) * np.identity(n)

    print(f'Epsilon: {epsilon}')
    print(f'mu: {mu}')
    print(f'M: {M}')

    r_matrix = np.array([M, weights.T]).T
    r_matrix *= -1/n
    
    print(f'r_matrix: {r_matrix}')
    
    l_matrix = np.array([[epsilon + alpha*v, mu],[mu.T, 1]])

    det = np.linalg.det(l_matrix)
    print(det)
    if det == 0:
        print("Determinant is zero")
    else:
        result = np.linalg.inv(l_matrix) * r_matrix

    return result[0], result[1]

def equation3(X, Y, **kwargs):
    pass