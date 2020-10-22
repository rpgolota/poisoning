import numpy as np
from sys import float_info
import sys, os

# current workaround to support imports from top level for things above the scope of this package
# don't remember if there is a better way

if __name__ == "__main__":
    sys.path.extend([f'../../../{name}' for name in os.listdir("../..") if os.path.isdir("../../" + name)])
else:
    sys.path.extend([f'../{name}' for name in os.listdir(".") if os.path.isdir(name)])

from poisoning.utils import PoisonLogger

def r_term(weights, **kwargs):
    
    """The r-term as described in the paper in equation 4 and 5

    Parameters:
        Weights: The weights (turned into np.array)
        Arguments:
            type : (str) : type of algorithm to use ::: {'lasso', 'ridge', 'elastic'}
                default => 'lasso'
            rho : (float) : Term in the elastic net
                default => 0.5
            range_value : (float) : Range the sub-gradient returns if weight at that point is zero
                default => 0.0
    
    Raises:
        TypeError: Invalid range value (less than -1 or greater than 1)
        TypeError: Unknown parameters
        TypeError: Invalid algorithm type

    Returns:
        Vector of the same size as the weights: The derivative of Omega term with respect to the weights
    """
    
    if kwargs:
        PoisonLogger.info(f'Got arguments: {kwargs}')
    
    algorithm = kwargs.pop('type', 'lasso')
    rho = kwargs.pop('rho', 0.5)
    range_value = kwargs.pop('range_value', 0.0)
    
    if range_value < -1 or range_value > 1:
        raise TypeError('range_value cannot be greater than 1 or less than -1')
    
    if kwargs:
        raise TypeError('Unknows parameters: ' + ', '.join(kwargs.keys()))
    
    weights = np.array(weights)

    PoisonLogger.info('Finding subgradient.')

    if algorithm == 'lasso':
        return subgradient(weights, range_value)
    elif algorithm == 'ridge':
        return weights
    elif algorithm == 'elastic':
        return rho * subgradient(weights, range_value) + (1 - rho) * weights
    else:
        raise TypeError(f'Invalid algorithm type provided: {algorithm}')

def subgradient(weights, range_value=0.0):
    """Returns the subgradient as described in the paper as sub(w)

    Args:
        weights (vector): The weights
        range_value (float, optional): In between -1 and 1 inclusive. Defaults to 0.0.

    Returns:
        [type]: [description]
    """
    
    PoisonLogger.info(f'Got argument for range_value::{range_value}.')
    
    return np.array([-1 if i < 0 else 1 if i > 0 else range_value for i in weights])

def equation7(X, Y, ax, ay, weights, biases, **kwargs):
    
    """Equation 7 in paper. Returns the partial derivatives of the weights and biases with respect to the attack point.

    Parameters:
        X: Data
        Y: Results
        ax: Attack point
        ay: Attack point y value
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
    ax = np.array(ax)
    weights = np.array(weights)
    
    if (X.shape[0] != Y.shape[0] or X.shape[1] != ax.shape[0] or X.shape[1] != weights.shape[0]):
        raise ValueError('X and Y must have the same dimensions. ax and weights must have same dimensions. X second dimension must equal ax dimension.')
    
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
    sigma = sum([np.outer(i, i) for i in X]) / n
    sigma_term = sigma + alph * v
    PoisonLogger.info('Trying to find mu.')
    mu = np.mean(X, axis=0)
    PoisonLogger.info('Trying to find M.')
    M = np.outer(ax, weights) + ((np.dot(weights, ax) + biases) - ay) * np.identity(n)

    PoisonLogger.info('Concatenating matrices for final left matrix and right matrix.')
    l_matrix = np.vstack((sigma_term, mu))
    mu_append = np.append(mu, 1)
    l_matrix = np.hstack((l_matrix, np.array([mu_append]).T))
    r_matrix = np.vstack((M, weights)) * (-1/n)

    PoisonLogger.debug(f'\nWeights:\n{weights}')
    PoisonLogger.debug(f'\nM:\n{M}')
    PoisonLogger.debug(f'\nl_matrix:\n{l_matrix}')
    PoisonLogger.debug(f'\nr_matrix\n{r_matrix}')
    
    PoisonLogger.info('Checking left_matrix for singulartiy.')
    if np.linalg.cond(l_matrix) < 1/float_info.epsilon:
        PoisonLogger.info(f'Inverting left matrix.')
        result = np.matmul(np.linalg.inv(l_matrix), r_matrix)
        PoisonLogger.debug(f'\nresult inv:\n{result}')
    else:
        PoisonLogger.info(f'Left matrix is singular, trying pinv instead.')
        result = np.matmul(np.linalg.pinv(l_matrix), r_matrix)
        PoisonLogger.debug(f'\nresult pinv:\n{result}')

    return result[:-1], result[-1]

def equation4(X, Y, ax, ay, weights, biases, **kwargs):
    
    """Equation 4 in paper. Returns the partial derivatives of the weights and biases with respect to the attack point.

    Parameters:
        X: Data
        Y: Results
        ax: Attack point
        ay: Attack point y value
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
    ax = np.array(ax)
    weights = np.array(weights)
    
    if (X.shape[0] != Y.shape[0] or X.shape[1] != ax.shape[0] or X.shape[1] != weights.shape[0]):
        raise ValueError('X and Y must have the same dimensions. ax and weights must have same dimensions. X second dimension must equal ax dimension.')
    
    partial_weights, partial_biases = equation7(X, Y, ax, ay, weights, biases, **kwargs)
    last_term = alph * (np.matmul(r_term(weights, type=algorithm, rho=rho), partial_weights))
    PoisonLogger.debug(f'\nlast_term:\n{last_term}')
    
    result = [((np.dot(item[0], ax) + biases) - item[1]) * (np.matmul(item[0], partial_weights) + partial_biases) for item in zip(X, Y)]
    result = (sum(result) / X.shape[0]) + last_term
    
    PoisonLogger.debug(f'\nResult:\n{result}')
    
    return result