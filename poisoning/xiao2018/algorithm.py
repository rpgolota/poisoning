from sklearn import linear_model
from sys import float_info
import numpy as np
import scipy as sp

class xiao2018:
    
    def __init__(self, **kwargs):
        
        self.projection_range = kwargs.pop('projection', 1)
        self.alpha = kwargs.pop('alpha', 1.0)
        self.rho = kwargs.pop('rho', 0.5)
        self.max_iter = kwargs.pop('max_iter', None)
        self.range_value = kwargs.pop('range_value', 0.0)
        self.algorithm_type = kwargs.pop('type', 'lasso').lower()
        
        if kwargs:
            raise TypeError('Unknown parameters: ' + ', '.join(kwargs.keys()))
    
    @property
    def projection_range(self):
        return self._projection_range
    
    @projection_range.setter
    def projection_range(self, value):
        
        # add check for size
        # 1 => (-1, 1)
        # (-3, 4) => (-3, 4)
        # [1, 2, 3] => [(-1, 1), (-2, 2), (-3, 3)]
        # [(3,11), (-11,-3)] => [(3,11), (-11,-3)]
        
        if np.array(value).dtype == 'object':
                raise TypeError('Invalid input for projection range')
        if type(value) is tuple and len(value) == 2:
            self._projection_type = 'range'
            value = (value[0], value[1]) if value[0] < value[1] else (value[1], value[0])
        elif type(value) is list:
            if type(value[0]) is tuple and len(value[0]) == 2:
                self._projection_type = 'vector'
                value = [(v[0], v[1]) if v[0] < v[1] else (v[1], v[0]) for v in value]
            else:
                self._projection_type = 'vector'
                value = [(v, -1 * v) if v > 0 else (-1 * v, v) for v in value]
        else:
            if not isinstance(value, (int, float, complex)) or isinstance(value, bool):
                raise TypeError('Invalid input for projection range')
            self._projection_type = 'range'
            value = (-1 * value, value) if value > 0 else (value, -1 * value)
        
        self._projection_range = value
    
    @property
    def algorithm_type(self):
        return self._algorithm_type
    
    @algorithm_type.setter
    def algorithm_type(self, alg_type):
        if alg_type in ['lasso', 'l1']:
            self._linear_algorithm = linear_model.Lasso(alpha=self.alpha)
            self._algorithm_type = 'lasso'
        elif alg_type in ['ridge', 'l2', 'ridgeregression', 'ridge-regression', 'ridge regression']:
            self._linear_algorithm = linear_model.Ridge(alpha=self.alpha)
            self._algorithm_type = 'ridge'
        elif alg_type in ['elastic', 'elasticnet', 'elastic-net', 'elastic net']:
            self._linear_algorithm = linear_model.ElasticNet(alpha=self.alpha)
            self._algorithm_type = 'elastic'
        else:
            raise TypeError(f'Invalid linear algorithm type: {alg_type}')
    
    def _partial_derivatives(self, X, Y, ax, ay, weights, biases):
        
        n = X.shape[1]
        
        if self.algorithm_type == 'lasso':
            v = 0
        elif self.algorithm_type == 'ridge':
            v = np.identity(n)
        else:
            v = (1 - self.rho) * np.identity(n)

        sigma = sum([np.outer(i, i) for i in X]) / n
        sigma_term = sigma + self.alpha * v
        mu = np.mean(X, axis=0)
        M = np.outer(ax, weights) + ((np.dot(weights, ax) + biases) - ay) * np.identity(n)

        l_matrix = np.vstack((sigma_term, mu))
        mu_append = np.append(mu, 1)
        l_matrix = np.hstack((l_matrix, np.array([mu_append]).T))
        r_matrix = np.vstack((M, weights)) * (-1/n)

        if np.linalg.cond(l_matrix) < 1/float_info.epsilon:
            result = np.matmul(np.linalg.inv(l_matrix), r_matrix)
        else:
            result = np.matmul(np.linalg.pinv(l_matrix), r_matrix)

        return result[:-1], result[-1]
    
    def _gradient(self, X, Y, ax, ay):
        
        weights = np.array(self._linear_algorithm.coef_)
        biases = self._linear_algorithm.intercept_
        partial_weights, partial_biases = self._partial_derivatives(X, Y, ax, ay, weights, biases)
        last_term = self.alpha * (np.matmul(self._gradient_r_term(weights), partial_weights))
        result = [((np.dot(item[0], ax) + biases) - item[1]) * (np.matmul(item[0], partial_weights) + partial_biases) for item in zip(X, Y)]

        return (sum(result) / X.shape[0]) + last_term
    
    def _learn_model(self, X, Y):
        self._linear_algorithm.fit(X, Y)
    
    def _gradient_r_term(self, weights):
        
        if self.algorithm_type == 'lasso':
            return np.array([-1 if i < 0 else 1 if i > 0 else self.range_value for i in weights])
        elif self.algorithm_type == 'ridge':
            return weights
        else:
            return self.rho * np.array([-1 if i < 0 else 1 if i > 0 else self.range_value for i in weights]) + (1 - self.rho) * weights
    
    def _project(self, value):
        if self._projection_type == 'vector':
            return np.array([v if v >= proj[0] and v <= proj[1] else min(proj, key=lambda val : abs(val - v)) for v, proj in zip(value, self._projection_range)])
        else:
            return np.array([v if v >= self._projection_range[0] and v <= self._projection_range[1] else min(self._projection_range, key=lambda val : abs(val - v)) for v in value])
        
    def _perform_checks(self, X, Y, Attacks, Labels):
        
        if X.dtype == 'object':
            raise ValueError('Inconsistent element sizes for X')
        if Y.dtype == 'object':
            raise ValueError('Inconsistent element sizes for Y')
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same first dimensions.')
        if Y.ndim != 1:
            raise ValueError('Y must be one dimensional')
        
        if Labels.dtype == 'object':
            raise ValueError('Inconsistent element sizes for Labels')
        if Labels.ndim != 1:
            raise ValueError('Labels must be one dimensional')
        if Attacks.dtype == 'object':
            raise ValueError('Inconsistent element sizes for Attacks')
        if Attacks.shape[0] != Labels.shape[0]:
            raise ValueError('Attacks and Labels must have the same first dimensions.')
        if Attacks.shape[1] != X.shape[1]:
            raise ValueError('Attacks and X must have the same second dimensions.')
        
        if self._projection_type == 'vector' and Attacks.shape[1] != len(self._projection_range):
            raise ValueError('Projection range must be of the same size as feature size.')
        
    # X = [[1, 2, 3], [3, 4, 5]]
    # Y = [1, 2]
    # Attacks = [[3,2,1],
    #            [8,2,3]]
    # Attacks.shape[0] = 2
    # Attacks.shape[1] = 3
    # 
    # project_range = [1,2] = [(-1,1), (-2,2)]
    # 
    # _project(Attacks[0]) = [3,2,1] => [1, 1, 1]
    # 
    # Labels = [1, 0]
    
    def run(self, X, Y, Attacks, Labels):
        
        X = np.array(X)
        Y = np.array(Y)
        Attacks = np.array(Attacks)
        Prev_Attacks = np.array(Attacks)
        Labels = np.array(Labels)

        self._perform_checks(X, Y, Attacks, Labels)
        
        self.n_iter = 0
        while self.n_iter < self.max_iter if self.max_iter is not None else True:
            
            for attack, label in zip(Attacks, Labels):
                
                self._learn_model(np.append(X, [attack]), np.append(Y, label))
                d = self._project(attack + self._gradient(X, Y, attack, label)) - attack
            
            # condition to break (equation 3)
            if True:
                pass
            Prev_Attacks = np.array(Attacks)
            self.n_iter += 1