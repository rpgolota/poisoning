from poisoning.xiao2018.algorithm import xiao2018
import numpy as np
import pytest
from tests.shared import get_inputs, find_inputs, read_json

def test_passing():
    xiao2018(type='lasso', projection=(-3,4), alpha=0.7, rho=0.5, max_iter=None, range_value=0)

def test_unknown_parameters():
    
    with pytest.raises(TypeError, match='Unknown parameters:') as err:
        xiao2018(iterations=1000)

    with pytest.raises(TypeError, match='Unknown parameters:') as err:
        xiao2018(other='haha')

    with pytest.raises(TypeError, match='Unknown parameters:') as err:
        xiao2018(lasso='3')

    with pytest.raises(TypeError, match='Unknown parameters:') as err:
        xiao2018(name='name')

@pytest.mark.filterwarnings("ignore:Creating an ndarray")
def test_projection_invalid_input():
    
    with pytest.raises(TypeError, match='Invalid input for projection range') as err:
        xiao2018(projection='string')
        
    with pytest.raises(TypeError, match='Invalid input for projection range') as err:
        xiao2018(projection='string')
    
    with pytest.raises(TypeError, match='Invalid input for projection range') as err:
        xiao2018(projection=[1, [2,3], 4, [3,5,6]])

@pytest.mark.parametrize('type', ['l-1', 'l-2', 'other', 'unknown', 'thingy', 'elastik', 'l1l2'])
def test_algorithm_invalid_type(type):
    
    with pytest.raises(TypeError, match='Invalid linear algorithm type:') as err:
        xiao2018(type=type)

@pytest.mark.filterwarnings("ignore:Creating an ndarray")
def test_X_inconsistent():
    file_contents = read_json('Fail_test_1.json')
    algo = xiao2018()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[0] = np.array([[1,2,3], [2,2]])
    
    with pytest.raises(ValueError, match='Inconsistent element sizes for X') as err:
        algo._perform_checks(*inputs)

@pytest.mark.filterwarnings("ignore:Creating an ndarray")
def test_Y_inconsistent():
    file_contents = read_json('Fail_test_1.json')
    algo = xiao2018()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[1] = np.array([1,2,[2,3]])
    
    with pytest.raises(ValueError, match='Inconsistent element sizes for Y') as err:
        algo._perform_checks(*inputs)

def test_X_Y_dim1():
    file_contents = read_json('Fail_test_1.json')
    algo = xiao2018()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[1] = np.array([1,2,2,3])
    
    with pytest.raises(ValueError, match='X and Y must have the same first dimensions.') as err:
        algo._perform_checks(*inputs)

def test_Y_one_dimensional():
    file_contents = read_json('Fail_test_1.json')
    algo = xiao2018()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[1] = np.array([[1,2], [3,4]])
    
    with pytest.raises(ValueError, match='Y must be one dimensional') as err:
        algo._perform_checks(*inputs)

@pytest.mark.filterwarnings("ignore:Creating an ndarray")
def test_Labels_inconsistent():
    file_contents = read_json('Fail_test_1.json')
    algo = xiao2018()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[3] = np.array([1,2,[2,3]])
    
    with pytest.raises(ValueError, match='Inconsistent element sizes for Labels') as err:
        algo._perform_checks(*inputs)

def test_Labels_one_dimensional():
    file_contents = read_json('Fail_test_1.json')
    algo = xiao2018()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[3] = np.array([[1,2], [3,4]])
    
    with pytest.raises(ValueError, match='Labels must be one dimensional') as err:
        algo._perform_checks(*inputs)

@pytest.mark.filterwarnings("ignore:Creating an ndarray")
def test_Attacks_inconsistent():
    file_contents = read_json('Fail_test_1.json')
    algo = xiao2018()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[2] = np.array([[1,2,3], [2,2]])
    
    with pytest.raises(ValueError, match='Inconsistent element sizes for Attacks') as err:
        algo._perform_checks(*inputs)

def test_Attacks_Labels_dim1():
    file_contents = read_json('Fail_test_1.json')
    algo = xiao2018()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[3] = np.array([1,2,2,3])
    
    with pytest.raises(ValueError, match='Attacks and Labels must have the same first dimensions.') as err:
        algo._perform_checks(*inputs)

def test_X_Attacks_dim2():
    file_contents = read_json('Fail_test_1.json')
    algo = xiao2018()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[2] = np.array([[1,2,3]])
    
    with pytest.raises(ValueError, match='Attacks and X must have the same second dimensions.') as err:
        algo._perform_checks(*inputs)

def test_Attacks_projection_size():
    file_contents = read_json('Fail_test_1.json')
    algo = xiao2018(projection=[1,2,3])

    inputs = [np.array(inp) for inp in file_contents]
    
    with pytest.raises(ValueError, match='Projection range must be of the same size as feature size.') as err:
        algo._perform_checks(*inputs)
