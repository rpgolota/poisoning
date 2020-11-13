from sklearn import linear_model
from sys import float_info
import numpy as np
import scipy as sp
import random

class xiao2018:
    """Dataset poisoning algorithm from 'Is Feature Selection Secure against Training Data Poisoning? H.Xiao et al. 2018'
    
    Parameters
    ----------
    type : {'lasso', 'l1', 'ridge', 'l2', 'ridgeregression', 'ridge-regression', 'ridge regression', 'elastic', 'elasticnet', 'elastic-net', 'elastic net'}, default='lasso'
        Type of linear model that will be used in the algorithm.
        
    beta : float, default=0.99
        Number that is raised to i-th power in line search, where i is the current line search iteration.
        
    rho : float, default=0.5
        Convex constant that describes the elastic-nex mixing parameter.
        
    sigma : float, default=1e-3
        Small positive constant used in bounding the line search.
        
    elsilon : float, default=1e-3
        Small positive consant used in bounding the algorithm.
        
    max_iter : int, default=1000
        Maximum iterations that the algorithm will go up to.
        
    max_lsearch_iter : int, default=10
        Maximum iterations that the line search will go up to.
        
    max_model_iter : int, default=1000
        Parameter that is passed into the linear model to bound iterations.
        
    model_tol : float, default=1e-3
        Paramter that is passed into the linear model as the tolerance.
    
    Attributes
    ----------
    
    n_iter : int
        Number of iterations completed.
    
    Raises
    ------
    TypeError
        If invalid type is given.
    
    """
    def __init__(self, *, type='lasso', beta=0.99,
                 rho=0.5, sigma=1e-3, epsilon=1e-3,
                 max_iter=1000, max_lsearch_iter=10,
                 max_model_iter=1000, model_tol=1e-3):

        self.beta = beta
        self.rho = rho
        self.sigma = sigma
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.max_lsearch_iter = max_lsearch_iter
        self.algorithm_type = type
        self.max_model_iter = max_model_iter
        self.model_tol = model_tol
        self._linear_algorithm = None
            
    @property
    def projection(self):
        """Returns the currently used projection.

        The setter will normalize the four possible inputs into either a tuple or a list of tuples,
        depending on the type of input that was given as input.
        
        Returns
        -------
        tuple or list of tuple
            Currently used projection.
            
        Raises
        ------
        TypeError
            If not the correct type of input for projection.
        """
        return self._projection
    
    @projection.setter
    def projection(self, value):
        
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
        
        self._projection = value
    
    @property
    def algorithm_type(self):
        """The currently used algorithm type.
        
        The setter will normalize the type into one of the three possible outputs.

        Returns
        -------
        {'lasso', 'ridge', 'elastic'}
            One of three types for the linear model.
            
        Raises
        ------
        TypeError
            If not set to one of the vailid inputs.
        """
        return self._algorithm_type
    
    @algorithm_type.setter
    def algorithm_type(self, alg_type):
        if alg_type in ['lasso', 'l1']:
            self._algorithm_type = 'lasso'
        elif alg_type in ['ridge', 'l2', 'ridgeregression', 'ridge-regression', 'ridge regression']:
            self._algorithm_type = 'ridge'
        elif alg_type in ['elastic', 'elasticnet', 'elastic-net', 'elastic net']:
            self._algorithm_type = 'elastic'
        else:
            raise TypeError(f'Invalid linear algorithm type: {alg_type}')
    
    @property
    def _linear_algorithm(self):
        if self.__linear_algorithm:
            return self.__linear_algorithm
        else:
            self._set_model()
            return self.__linear_algorithm
            
    @_linear_algorithm.setter
    def _linear_algorithm(self, data):
        self.__linear_algorithm = data
    
    def _set_model(self, **args):
        if self.algorithm_type == "lasso":
            self._linear_algorithm = linear_model.Lasso(**args)
        elif self.algorithm_type == "ridge":
            self._linear_algorithm = linear_model.Ridge(**args)
        else:
            self._linear_algorithm = linear_model.ElasticNet(**args)

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
        try:
            self._linear_algorithm.fit(X, Y)
        except:
            print('here')
    
    def _gradient_r_term(self, weights):
        
        if self.algorithm_type == 'lasso':
            return np.array([-1 if i < 0 else 1 if i > 0 else 0 for i in weights])
        elif self.algorithm_type == 'ridge':
            return weights
        else:
            return self.rho * np.array([-1 if i < 0 else 1 if i > 0 else 0 for i in weights]) + (1 - self.rho) * weights
    
    def _project(self, value):
        if self._projection_type == 'vector':
            return np.array([v if v >= proj[0] and v <= proj[1] else min(proj, key=lambda val : abs(val - v)) for v, proj in zip(value, self._projection)])
        else:
            return np.array([v if v >= self._projection[0] and v <= self._projection[1] else min(self._projection, key=lambda val : abs(val - v)) for v in value])
    
    def _regularize(self, weights):
        norm = np.linalg.norm(weights)
        if self.algorithm_type == 'lasso':
            return norm
        elif self.algorithm_type == 'ridge':
            return (1/2) * (norm**2)
        else:
            return (self.rho * norm) + (1 - self.rho) * ((1/2) * (norm**2))
    
    def _bounds(self, X, Y):
        length = X.shape[0]
        weights = self._linear_algorithm.coef_
        bias = self._linear_algorithm.intercept_
        return (1/length) * sum([(1/2) * ((np.dot(weights, x) - y) ** 2) for x, y in zip(X, Y)]) + (self.alpha * self._regularize(weights))
    
    def _find_alpha(self, X, Y):
    
        if len(X) < 5:
            cv = len(X)
        else:
            cv = None
    
        if self.algorithm_type == "lasso":
            reg = linear_model.LassoCV(cv=cv, max_iter=self.max_model_iter, tol=self.model_tol).fit(X, Y)
        elif self.algorithm_type == "ridge":
            reg = linear_model.RidgeCV(cv=cv).fit(X, Y) # sklearn.exceptions.UndefinedMetricWarning: R^2 score is not well-defined with less than two samples.
        else:
            reg = linear_model.ElasticNetCV(l1_ratio=self.rho, cv=cv, max_iter=self.max_model_iter, tol=self.model_tol).fit(X, Y)
    
        return reg.alpha_
    
    def _check_dataset(self, X, Y):
        
        if X.dtype == 'object':
            raise ValueError('Inconsistent element sizes for X')
        if Y.dtype == 'object':
            raise ValueError('Inconsistent element sizes for Y')
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same first dimensions.')
        if Y.ndim != 1:
            raise ValueError('Y must be one dimensional')
    
    def _perform_checks(self, X, Y, Attacks, Labels):
        
        self._check_dataset(X, Y)
        
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
        
        if self._projection_type == 'vector' and Attacks.shape[1] != len(self._projection):
            raise ValueError('Projection range must be of the same size as feature size.')
    
    def run(self, X, Y, Attacks, Labels, projection):
        """Runs the algorithm.

        Parameters
        ----------
        X : array_like
            Dataset with no labels.
            
        Y : array_like 
            Labels to dataset.
            
        Attacks : array_like
            Initial attack points.
            
        Labels : array_like 
            Labels to attack points.
            
        projection : float or tuple or list of float or list of tuple
            If float or tuple, the projection will be the same for all features,
            otherwise if a list, the projection will be described feature by feature.

        Returns
        -------
        array_like
            Final attack points, optimized by the algorithm.
            
        Raises
        ------
        TypeError
            When invalid projection is passed.
        
        ValueError
            When incorrect dimensions of X, Y, Attacks, Labels, or projection is passed.
        """
        self.projection = projection
        
        X = np.array(X)
        Y = np.array(Y)
        Attacks = np.array(Attacks)
        Prev_Attacks = np.array(Attacks)
        Labels = np.array(Labels)

        self._perform_checks(X, Y, Attacks, Labels)

        self.alpha = self._find_alpha(X, Y)
        model_args = {'alpha': self.alpha, 'max_iter': self.max_model_iter}
        if self.algorithm_type == 'elastic':
            model_args['l1_ratio'] = self.rho
        
        self._set_model(**model_args)
        
        self.n_iter = 0
        while self.n_iter < self.max_iter:
            
            New_attacks = []
            
            for attack, label in zip(Attacks, Labels):
                
                self._learn_model(np.vstack((X, attack)), np.append(Y, label))
                d = self._project(attack + self._gradient(X, Y, attack, label)) - attack
            
                n_line_iter = 0
                while n_line_iter < self.max_lsearch_iter:
                    eta = self.beta ** n_line_iter
                    new_attack = attack + eta*d
                    if self._bounds(np.array([new_attack]), np.array([label])) <= self._bounds(np.array([attack]), np.array([label])) - self.sigma * eta * (np.linalg.norm(d) ** 2):
                        break
                    n_line_iter += 1
                
                New_attacks.append(new_attack)
            
            Attacks = np.array(New_attacks)
            if abs(self._bounds(Attacks, Labels) - self._bounds(Prev_Attacks, Labels)) < self.epsilon:
                break
            Prev_Attacks = np.array(Attacks)
            self.n_iter += 1
        
        return Attacks
    
    def autorun(self, X, Y, num_attacks, projection, rInitial=False):    
        """Runs the algorithm with a certain number of initial attack points randomly chosen.

        Parameters
        ----------
        X : array_like
            Dataset with no labels.
            
        Y : array_like 
            Labels to dataset.
        num_attacks : int
            Number of attack points to randomly choose from dataset.
            
        projection : float or tuple or list of float or list of tuple
            If float or tuple, the projection will be the same for all features,
            otherwise if a list, the projection will be described feature by feature.
            
        rInitial : bool, default=False
            If true will also return the randomly chosen attack points.
            
        Returns
        -------
        array_like or tuple of array_like
            Either returns the optimized atack points, or a tuple of the optimized attack points and the original.

        Raises
        ------
        ValueError
            Same as xiao2018.run()
            
        TypeError
            Same as xiao2018.run()
        """
        self.projection = projection
        X_np = np.array(X)
        Y_np = np.array(Y)
        self._check_dataset(X_np, Y_np)
        
        Attacks = random.sample([x + [y] for x, y in zip(X, Y)], num_attacks)
        Labels = [row[-1] for row in Attacks]
        Attacks = [row[:-1] for row in Attacks]
        
        Result = self.run(X, Y, Attacks, Labels, projection)
        
        if rInitial:
            return Result, np.array(Attacks)
        else:
            return Result