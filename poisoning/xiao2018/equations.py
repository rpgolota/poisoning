from sklearn import linear_model
from sys import float_info
import numpy as np

import sys, os

# current workaround to support imports from top level for things above the scope of this package
# don't remember if there is a better way

if __name__ == "__main__":
    sys.path.extend([f'../../../{name}' for name in os.listdir("../..") if os.path.isdir("../../" + name)])
else:
    sys.path.extend([f'../{name}' for name in os.listdir(".") if os.path.isdir(name)])

from poisoning.utils import PoisonLogger

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
        ValueError: Invalid X and Y dimensions
        TypeError: Unknown parameters
        TypeError: Invalid algorithm type

    Returns:
        (W,B): Returns a 2 size tuple of the weights and biases as found by the algorithm
        (W,B,O): returns a 3 size tuple of the weights and biases and linear model as last member
    """
    
    if kwargs:
        PoisonLogger.info(f'Got arguments: {kwargs}.')
    
    algorithm = kwargs.pop('type', 'lasso')
    aplh = kwargs.pop('alpha', 1.0)
    obj = kwargs.pop('object', None)
    ret_obj = kwargs.pop('return_object', obj is not None)
    
    if kwargs:
        raise TypeError('Unknown parameters: ' + ', '.join(kwargs.keys()))
    
    X = np.array(X)
    Y = np.array(Y)
    
    if (X.shape[0] != Y.shape[0]):
        raise ValueError('X and Y must be of the same dimension.')
    
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
    
    PoisonLogger.info('Calling sklearn fit on data.')
    algorithm.fit(X, Y)
    
    if ret_obj:
        return algorithm.coef_, algorithm.intercept_, algorithm
    else:
        return algorithm.coef_, algorithm.intercept_

def equation7(X, Y, weights, biases, **kwargs):
    
    """Equation 7 in paper. Returns the partial derivatives of the weights and biases with respect to the attack point.

    Parameters:
        X: Data
        Y: Results
        weights: weights
        biases: biases
        Arguments:
            type : (str) : type of algorithm to use ::: {'lasso', 'ridge', 'elastic'}
                default => 'lasso'
            alpha : (number) : alpha to be used in linear model by sklearn
                default => 1.0
            rho : (number) : rho used in sklearn for elastic net. Same as l1_ratio
                default => 0.5
                
    Raises:
        ValueError: Invalid X and Y dimensions
        TypeError: Unknown parameters
        TypeError: Invalid algorithm type

    Returns:
        Tuple: partial derivatives with respect to weights for (weights, biases)
    """
    
    if kwargs:
        PoisonLogger.info(f'Got arguments: {kwargs}')
    
    algorithm = kwargs.pop('type', 'lasso')
    alph = kwargs.pop('alpha', 1.0)
    rho = kwargs.pop('rho', 0.5)
    
    if kwargs:
        raise TypeError('Unknown parameters: ' + ', '.join(kwargs.keys()))
    
    X = np.array(X)
    Y = np.array(Y)
    weights = np.array(weights)
    
    if (X.shape[0] != Y.shape[0] or X.shape[0] != weights.shape[0]):
        raise ValueError('X and Y must be of the same dimension.')
    
    n = X.shape[1]
    PoisonLogger.info(f'Got second dimension of X:{n}.')
    
    if algorithm == 'lasso':
        v = 0
    elif algorithm == 'ridge':
        v = np.identity(n)
    elif algorithm == 'elastic':
        v = (1 - rho) * np.identity(n)
    else:
        raise TypeError(f'Invalid algorithm type provided: {algorithm}')

    PoisonLogger.info(f'Found identiy matrix.')

    PoisonLogger.info('Trying to find sigma.')
    sigma = sum([np.outer(i, i) for i in X.T]) / n # covariance does not give this
    sigma_term = sigma + alph * v
    PoisonLogger.info('Trying to find mu.')
    mu = np.mean(X, axis=0)
    PoisonLogger.info('Trying to find M.')
    M = np.outer(X, weights) + ((np.dot(weights, X) + biases) - Y) * np.identity(n) # Problem here

    # X = np.array([[1,2],[3,4]])
    # Y = np.array([7,8])
    # w = np.array([0.2, 3.2])
    # b = 0.11
    # equation7(X, Y, w, b)

    PoisonLogger.info('Concatenating matrices for final left matrix and right matrix.')
    l_matrix = np.vstack((sigma_term, mu))
    mu_append = np.append(mu, 1)
    l_matrix = np.hstack((l_matrix, np.array([mu_append]).T))
    r_matrix = np.concatenate((M, weights), axis=0) * (-1/n)

    PoisonLogger.info('Checking left_matrix for singulartiy.')
    if np.linalg.cond(l_matrix) < 1/float_info.epsilon:
        PoisonLogger.info(f'Inverting left matrix.')
        result = np.matmul(np.linalg.inv(l_matrix), r_matrix)
    else:
        PoisonLogger.info(f'Left matrix is singular, trying pinv instead.')
        result = np.matmul(np.linalg.pinv(l_matrix), r_matrix) # is using pseudo-inverse acceptible?
    
    return result