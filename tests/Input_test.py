from poisoning.xiao2018.algorithm import xiao2018
import numpy as np
import pytest
import random
from tests.shared import get_inputs, find_inputs, read_json

PROJECTION = [1, (1,2), [1,2], [(1,2), (3,4)]]
ALPHA = [0, 0.1, 0.9, 1]
RHO = [0, 0.1, 0.9, 1]
MAX_ITER = [None, 10]
RANGE_VALUE = [0, 1]
ALGORITHM_TYPE = ['lasso', 'l1', 'ridge', 'l2', 'ridgeregression', 'ridge-regression', 'ridge regression', 'elastic', 'elasticnet', 'elastic-net', 'elastic net']
ALGORITHM_TYPE_SMALL = ['l1', 'l2', 'elastic']

@pytest.mark.parametrize('type', ALGORITHM_TYPE)
def test_type(type):
    test = xiao2018(type=type)
    
@pytest.mark.parametrize('proj', PROJECTION)
def test_projection_input(proj):
    test = xiao2018(projection=proj)

@pytest.mark.parametrize('file', find_inputs('Input_test'))
def test_learn_model(file):
    inp = read_json(file)
    test = xiao2018()
    test._learn_model(inp[0], inp[1])

@pytest.mark.parametrize('type', ALGORITHM_TYPE_SMALL)
@pytest.mark.parametrize('file', find_inputs('Input_test'))
def test_gradient(file, type):
    inp = read_json(file)
    test = xiao2018(type=type)
    test._learn_model(inp[0], inp[1])
    res = test._gradient(np.array(inp[0]), inp[1], inp[2][0], inp[3][0])
    assert res.shape[0] == len(inp[0][0])

@pytest.mark.parametrize('type', ALGORITHM_TYPE_SMALL)
@pytest.mark.parametrize('size', [1,5,10,20])
def test_gradient_r_term(size, type):
    weights = np.random.rand(size)
    test = xiao2018(type=type)
    res = test._gradient_r_term(weights)
    assert len(res) == len(weights)

@pytest.mark.parametrize('size', [1,5,10,20])
@pytest.mark.parametrize('proj_type', ['scalar', 'tuple', 'vectorscalar', 'vectortuple'])
def test_project(proj_type, size):
    vec = [random.randint(-10, 10) for i in range(size)]
    switch = {  'scalar': random.randint(-10, 10),
                'tuple': (random.randint(-10, 10), random.randint(-10, 10)),
                'vectorscalar': [random.randint(-10, 10) for i in range(size)],
                'vectortuple': [(random.randint(-10, 10), random.randint(-10, 10)) for i in range(size)]
            }
    test = xiao2018(projection=switch[proj_type])
    res = test._project(vec)
    assert len(res) == len(vec)

@pytest.mark.xfail(reason='Not finished implementing.')
@pytest.mark.parametrize('type', ALGORITHM_TYPE_SMALL)
@pytest.mark.parametrize('file', find_inputs('Input_test'))
def test_run(file, type):
    inp = read_json(file)
    test = xiao2018(type=type)
    res = test.run(*inp)
    assert res.shape[0] == len(inp[0]) and res.shape[1] == len(inp[0][0])