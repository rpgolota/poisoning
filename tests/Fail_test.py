from poisoning.algorithm import xiao2018, frederickson2018
import numpy as np
import pytest

def failed_data():
    dat = [
            [[1, 2], [3, 4]],
            [1, 2],
            [[1,2]],
            [2]
        ]
    return dat
    
@pytest.mark.filterwarnings("ignore:Creating an ndarray")
@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_projection_invalid_input(model):
    
    with pytest.raises(TypeError, match='Invalid input for projection range') as err:
        model().projection = 'string'
        
    with pytest.raises(TypeError, match='Invalid input for projection range') as err:
        model().projection = (1,2,3)
    
    with pytest.raises(TypeError, match='Invalid input for projection range') as err:
        model().projection = [1, [2,3], 4, [3,5,6]]

@pytest.mark.parametrize('type', ['l-1', 'l-2', 'other', 'unknown', 'thingy', 'elastik', 'l1l2'])
@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_algorithm_invalid_type(type, model):
    
    with pytest.raises(TypeError, match='Invalid linear algorithm type:') as err:
        model(type=type)

@pytest.mark.filterwarnings("ignore:Creating an ndarray")
@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_X_inconsistent(model):
    file_contents = failed_data()
    algo = model()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[0] = np.array([[1,2,3], [2,2]])
    
    with pytest.raises(ValueError, match='Inconsistent element sizes for X') as err:
        algo._perform_checks(*inputs)

@pytest.mark.filterwarnings("ignore:Creating an ndarray")
@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_Y_inconsistent(model):
    file_contents = failed_data()
    algo = model()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[1] = np.array([1,2,[2,3]])
    
    with pytest.raises(ValueError, match='Inconsistent element sizes for Y') as err:
        algo._perform_checks(*inputs)

@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_X_Y_dim1(model):
    file_contents = failed_data()
    algo = model()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[1] = np.array([1,2,2,3])
    
    with pytest.raises(ValueError, match='X and Y must have the same first dimensions.') as err:
        algo._perform_checks(*inputs)

@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_Y_one_dimensional(model):
    file_contents = failed_data()
    algo = model()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[1] = np.array([[1,2], [3,4]])
    
    with pytest.raises(ValueError, match='Y must be one dimensional') as err:
        algo._perform_checks(*inputs)

@pytest.mark.filterwarnings("ignore:Creating an ndarray")
@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_Labels_inconsistent(model):
    file_contents = failed_data()
    algo = model()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[3] = np.array([1,2,[2,3]])
    
    with pytest.raises(ValueError, match='Inconsistent element sizes for Labels') as err:
        algo._perform_checks(*inputs)

@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_Labels_one_dimensional(model):
    file_contents = failed_data()
    algo = model()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[3] = np.array([[1,2], [3,4]])
    
    with pytest.raises(ValueError, match='Labels must be one dimensional') as err:
        algo._perform_checks(*inputs)

@pytest.mark.filterwarnings("ignore:Creating an ndarray")
@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_Attacks_inconsistent(model):
    file_contents = failed_data()
    algo = model()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[2] = np.array([[1,2,3], [2,2]])
    
    with pytest.raises(ValueError, match='Inconsistent element sizes for Attacks') as err:
        algo._perform_checks(*inputs)

@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_Attacks_Labels_dim1(model):
    file_contents = failed_data()
    algo = model()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[3] = np.array([1,2,2,3])
    
    with pytest.raises(ValueError, match='Attacks and Labels must have the same first dimensions.') as err:
        algo._perform_checks(*inputs)

@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_X_Attacks_dim2(model):
    file_contents = failed_data()
    algo = model()

    inputs = [np.array(inp) for inp in file_contents]
    inputs[2] = np.array([[1,2,3]])
    
    with pytest.raises(ValueError, match='Attacks and X must have the same second dimensions.') as err:
        algo._perform_checks(*inputs)

@pytest.mark.parametrize('model', [xiao2018, frederickson2018])
def test_Attacks_projection_size(model):
    file_contents = failed_data()
    algo = model()
    algo.projection = [1,2,3]

    inputs = [np.array(inp) for inp in file_contents]
    
    with pytest.raises(ValueError, match='Projection range must be of the same size as feature size.') as err:
        algo._perform_checks(*inputs)