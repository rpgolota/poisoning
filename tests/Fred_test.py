from poisoning.algorithm import frederickson2018
import numpy as np
import pytest
from tests.shared import get_inputs, find_inputs, read_json

RHO = [0, 0.1, 0.9, 1]
MAX_ITER = [None, 10]
ALGORITHM_TYPE = ['lasso', 'l1', 'ridge', 'l2', 'ridgeregression', 'ridge-regression', 'ridge regression', 'elastic', 'elasticnet', 'elastic-net', 'elastic net']
ALGORITHM_TYPE_SMALL = ['l1', 'l2', 'elastic']

@pytest.mark.xfail(reason='Bug in _k_nearest code')
@pytest.mark.parametrize('outlier_type', ['nearest neighbor'])
@pytest.mark.parametrize('type', ALGORITHM_TYPE_SMALL)
@pytest.mark.parametrize('file', find_inputs('Input_test'))
def test_manual(file, type, outlier_type):
    inp = read_json(file)
    test = frederickson2018(type=type, outlier_type=outlier_type)
    res = test.run(*inp, 1)
    assert res.shape[0] == len(inp[2]) and res.shape[1] == len(inp[2][0])