from poisoning.xiao2018.gradient import r_term, subgradient
import numpy as np
import pytest

class Test_r_term:
    
    okay_weights = np.array([11, 9.2, -1.1, 3.2, -2.3, 0.2, 0.0, -2.3, 0, 23])
    
    def test_invalid_parameters(self):
        
        with pytest.raises(TypeError) as info:
            r_term(Test_r_term.okay_weights, this=3)
    
        with pytest.raises(TypeError) as info:
            r_term(Test_r_term.okay_weights, other='haha')

    def test_invalid_type(self):
        
        with pytest.raises(TypeError) as info:
            r_term(Test_r_term.okay_weights, type='not allowed')
    
        with pytest.raises(TypeError) as info:
            r_term(Test_r_term.okay_weights, other='elastic net')

    def test_invalid_range(self):
        
        with pytest.raises(TypeError) as info:
            r_term(Test_r_term.okay_weights, range_value=-2)
    
        with pytest.raises(TypeError) as info:
            r_term(Test_r_term.okay_weights, range_value=3)
            
        r_term(Test_r_term.okay_weights, range_value=1)
        r_term(Test_r_term.okay_weights, range_value=-1)
    
    def test_lasso(self):
        assert all([a == b for a, b in zip(subgradient(Test_r_term.okay_weights), r_term(Test_r_term.okay_weights, type='lasso'))])
    
    def test_ridge(self):
        assert all([a == b for a, b in zip(Test_r_term.okay_weights, r_term(Test_r_term.okay_weights, type='ridge'))])
    
    def test_elastic(self):
        assert all([a == b for a, b in zip((0.5 * subgradient(Test_r_term.okay_weights) + (1 - 0.5) * Test_r_term.okay_weights), r_term(Test_r_term.okay_weights, type='elastic'))])
    
    def test_elastic_rho(self):
        pass
    
class Test_subgradient:
    
    okay_weights = np.array([11, 9.2, -1.1, 3.2, -2.3, 0.2, 0.0, -2.3, 0, 23])
    okay_range_value = 0.3
    
    def test_value(self):
        assert all([a == b for a, b in zip([1, 1, -1, 1, -1, 1, 0, -1, 0, 1], subgradient(Test_subgradient.okay_weights))])
    
    def test_value_range_value(self):
        
        assert all([a == b for a, b in zip([1, 1, -1, 1, -1, 1, 0.3, -1, 0.3, 1], subgradient(Test_subgradient.okay_weights, Test_subgradient.okay_range_value))])