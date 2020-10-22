from sklearn import linear_model
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
    # add rho to the options available
    
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