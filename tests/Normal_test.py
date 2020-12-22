from poisoning.algorithm import xiao2018, frederickson2018
import numpy as np
import pytest
import random
import pandas as pd
import glob

import os
import json

SEARCH_PATH = os.path.normcase(os.getcwd() + '/tests/input/')

def mark_parameters(X=[], Y=[], xfail=[], *, mark=pytest.mark.xfail, reason=''):
    out = [(x, y) if (x, y) not in xfail else pytest.param(x, y, marks=mark(reason=reason)) for x in X for y in Y]
    return out

def find_inputs(name):
    onlyfiles = [f for f in os.listdir(SEARCH_PATH) if os.path.isfile(os.path.join(SEARCH_PATH, f))]
    return [this_file for this_file in onlyfiles if this_file.find(name) != -1]

def read_json(filename):
    with open(os.path.join(SEARCH_PATH, filename), 'r') as f:
        return json.load(f)

RHO = [0, 0.1, 0.9, 1]
MAX_ITER = [None, 10]
ALGORITHM_TYPE = ['lasso', 'l1', 'ridge', 'l2', 'ridgeregression', 'ridge-regression', 'ridge regression', 'elastic', 'elasticnet', 'elastic-net', 'elastic net']
ALGORITHM_TYPE_SMALL = ['l1', 'l2', 'elastic']

@pytest.mark.parametrize('type', ALGORITHM_TYPE)
@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_type(type, model):
    test = model(type=type)

@pytest.mark.parametrize('file', find_inputs('Input_test'))
@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_learn_model(file, model):
    inp = read_json(file)
    test = model()
    test._set_model(inp[0], inp[1])
    test._learn_model(inp[0], inp[1])

@pytest.mark.parametrize('type', ALGORITHM_TYPE_SMALL)
@pytest.mark.parametrize('file,model', mark_parameters(find_inputs('Input_test'), [xiao2018, frederickson2018], [('Input_test_1.json', frederickson2018)], mark=pytest.mark.skip, reason='Too complicated to refactor.'))
def test_gradient(type, file, model):
    inp = read_json(file)
    test = model(type=type)
    test._set_model(inp[0], inp[1])
    test._learn_model(inp[0], inp[1])
    res = test._gradient(np.array(inp[0]), inp[1], inp[2][0], inp[3][0])
    assert res.shape[0] == len(inp[0][0])

@pytest.mark.parametrize('file,model', mark_parameters(find_inputs('Input_test'), [xiao2018, frederickson2018], [('Input_test_1.json', frederickson2018)], mark=pytest.mark.skip, reason='Too complicated to refactor.'))
def test_bounds(file, model):
    inp = read_json(file)
    test = model()
    test._set_model(inp[0], inp[1])
    test._learn_model(inp[0], inp[1])
    res = test._bounds(np.array(inp[0]), np.array(inp[1]))
    
@pytest.mark.parametrize('type', ALGORITHM_TYPE_SMALL)
@pytest.mark.parametrize('size', [1,5,10,20])
@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_gradient_r_term(size, type, model):
    weights = np.random.rand(size)
    test = model(type=type)
    res = test._gradient_r_term(weights)
    assert len(res) == len(weights)

@pytest.mark.parametrize('size', [1,5,10,20])
@pytest.mark.parametrize('proj_type', ['scalar', 'tuple', 'vectorscalar', 'vectortuple'])
@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_project(proj_type, size, model):
    vec = [random.randint(-10, 10) for i in range(size)]
    switch = {  'scalar': random.randint(-10, 10),
                'tuple': (random.randint(-10, 10), random.randint(-10, 10)),
                'vectorscalar': [random.randint(-10, 10) for i in range(size)],
                'vectortuple': [(random.randint(-10, 10), random.randint(-10, 10)) for i in range(size)]
            }
    test = model()
    test.projection = switch[proj_type]
    res = test._project(vec)
    assert len(res) == len(vec)

@pytest.mark.parametrize('type', ALGORITHM_TYPE_SMALL)
@pytest.mark.parametrize('file', find_inputs('Input_test'))
@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_manual(type, file, model):
    inp = read_json(file)
    test = model(type=type)
    res = test.run(*inp, 1)
    assert res.shape[0] == len(inp[2]) and res.shape[1] == len(inp[2][0])

@pytest.mark.parametrize('num', [1, 2])
@pytest.mark.parametrize('file', find_inputs('Input_test'))
@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_autorun(file, num, model):
    inp = read_json(file)
    test = model()
    res, _ = test.autorun(inp[0], inp[1], num, 1)
    assert len(res) == num

@pytest.mark.parametrize('type', ALGORITHM_TYPE_SMALL)
@pytest.mark.parametrize('file', map(os.path.normpath, glob.glob('tests/input/*.csv')))
@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_dataset(file, type, model):
    
    dataset = pd.read_csv(file, sep=",", header=None)

    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,-1].values

    n_attacks = 1

    test = model(type=type)
    res, _ = test.autorun(X, Y, n_attacks, 0.5)
    assert res.shape[0] == n_attacks and res.shape[1] == len(X[1])