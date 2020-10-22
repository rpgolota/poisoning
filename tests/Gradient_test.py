from poisoning.xiao2018.gradient import r_term, subgradient, equation7, equation4
import numpy as np
import pytest
from tests.shared import find_inputs

class Test_r_term_fail:
    
    okay_weights = np.array([11, 9.2, -1.1, 3.2, -2.3, 0.2, 0.0, -2.3, 0, 23])

    @pytest.mark.parametrize('param', [{'this':3}, {'test123':332}, {'other':'haha'}, {'thing':'t'}])
    def test_parameters(self, param):
        with pytest.raises(TypeError) as info:
            r_term(Test_r_term_fail.okay_weights, **param)

    @pytest.mark.parametrize('tp', ['notlasso', 3, 92, 'other', 9.2, True])
    def test_type(self, tp):
        with pytest.raises(TypeError) as info:
            r_term(Test_r_term_fail.okay_weights, type=tp)

    @pytest.mark.parametrize('value', [-2, 3])
    def test_range(self, value):
        with pytest.raises(TypeError) as info:
            r_term(Test_r_term_fail.okay_weights, range_value=value)

class Test_r_term:
    
    okay_weights = np.array([11, 9.2, -1.1, 3.2, -2.3, 0.2, 0.0, -2.3, 0, 23])
    
    def test_lasso(self):
        assert all([a == b for a, b in zip(subgradient(Test_r_term.okay_weights), r_term(Test_r_term.okay_weights, type='lasso'))])
    
    def test_ridge(self):
        assert all([a == b for a, b in zip(Test_r_term.okay_weights, r_term(Test_r_term.okay_weights, type='ridge'))])
    
    def test_elastic(self):
        assert all([a == b for a, b in zip((0.5 * subgradient(Test_r_term.okay_weights) + (1 - 0.5) * Test_r_term.okay_weights), r_term(Test_r_term.okay_weights, type='elastic'))])
    
    @pytest.mark.parametrize('value', [1, -1])
    def test_range(self, value):    
        r_term(Test_r_term_fail.okay_weights, range_value=value)
        
    @pytest.mark.skip(reason='Not implemented')
    def test_elastic_rho(self):
        pass


class Test_subgradient:
    
    okay_weights = np.array([11, 9.2, -1.1, 3.2, -2.3, 0.2, 0.0, -2.3, 0, 23])
    okay_range_value = 0.3
    
    def test_value(self):
        assert all([a == b for a, b in zip([1, 1, -1, 1, -1, 1, 0, -1, 0, 1], subgradient(Test_subgradient.okay_weights))])
    
    def test_value_range_value(self):
        
        assert all([a == b for a, b in zip([1, 1, -1, 1, -1, 1, 0.3, -1, 0.3, 1], subgradient(Test_subgradient.okay_weights, Test_subgradient.okay_range_value))])

    
class Test_Equation_7_fail:
    okay_X = [[1, 2], [3, 4]]
    okay_Y = [1, 2]
    okay_ax = [3, 1]
    okay_ay = 3.2
    okay_weights = [0.2, 1.3]
    okay_biases = 0.1
    okay_lambda = 0.23
    
    @pytest.mark.parametrize('param', [{'this':3}, {'test123':332}, {'other':'haha'}, {'thing':'t'}])
    def test_parameters(self, param):
        with pytest.raises(TypeError) as info:
            equation7(Test_Equation_7_fail.okay_X, Test_Equation_7_fail.okay_Y, Test_Equation_7_fail.okay_ax, Test_Equation_7_fail.okay_ay, Test_Equation_7_fail.okay_weights, Test_Equation_7_fail.okay_biases, **param)        

    @pytest.mark.parametrize('tp', ['notlasso', 3, 92, 'other', 9.2, True])
    def test_type(self, tp):
        with pytest.raises(TypeError) as info:
            equation7(Test_Equation_7_fail.okay_X, Test_Equation_7_fail.okay_Y, Test_Equation_7_fail.okay_ax, Test_Equation_7_fail.okay_ay, Test_Equation_7_fail.okay_weights, Test_Equation_7_fail.okay_biases, type=tp)
    
    @pytest.mark.parametrize('addx,addy,adda,addw', [([[3,3]],[],[],[]),([],[3],[],[]),([],[],[3],[]),([],[],[],[3])])
    def test_dimensions(self, addx,addy,adda,addw):
        
        with pytest.raises(ValueError) as info:
            equation7(Test_Equation_7_fail.okay_X + addx, Test_Equation_7_fail.okay_Y + addy, Test_Equation_7_fail.okay_ax + adda, Test_Equation_7_fail.okay_ay, Test_Equation_7_fail.okay_weights + addw, Test_Equation_7_fail.okay_biases)
            
class Test_Equation_7:
    
    @pytest.mark.parametrize('inputs', find_inputs('eq7'))
    @pytest.mark.parametrize('tp', ['lasso', 'ridge', 'elastic'])
    @pytest.mark.parametrize('const', [(0.7,0.7),(0.3,0.4)])
    def test_inputs(self, tp, inputs, const):
        X = inputs[0]
        Y = inputs[1]
        ax = inputs[2]
        ay = inputs[3]
        weights = inputs[4]
        biases = inputs[5]
        res = equation7(X, Y, ax, ay, weights, biases, type=tp, alpha=const[0], rho=const[1])
        
class Test_Equation_4_fail:
    okay_X = [[1, 2], [3, 4]]
    okay_Y = [1, 2]
    okay_ax = [3, 1]
    okay_ay = 3.2
    okay_weights = [0.2, 1.3]
    okay_biases = 0.1
    okay_lambda = 0.23
    
    @pytest.mark.parametrize('param', [{'this':3}, {'test123':332}, {'other':'haha'}, {'thing':'t'}])
    def test_parameters(self, param):
        with pytest.raises(TypeError) as info:
            equation4(Test_Equation_4_fail.okay_X, Test_Equation_4_fail.okay_Y, Test_Equation_4_fail.okay_ax, Test_Equation_4_fail.okay_ay, Test_Equation_4_fail.okay_weights, Test_Equation_4_fail.okay_biases, **param)        

    @pytest.mark.parametrize('tp', ['notlasso', 3, 92, 'other', 9.2, True])
    def test_type(self, tp):
        with pytest.raises(TypeError) as info:
            equation4(Test_Equation_4_fail.okay_X, Test_Equation_4_fail.okay_Y, Test_Equation_4_fail.okay_ax, Test_Equation_4_fail.okay_ay, Test_Equation_4_fail.okay_weights, Test_Equation_4_fail.okay_biases, type=tp)
    
    @pytest.mark.parametrize('addx,addy,adda,addw', [([[3,3]],[],[],[]),([],[3],[],[]),([],[],[3],[]),([],[],[],[3])])
    def test_dimensions(self, addx,addy,adda,addw):
        
        with pytest.raises(ValueError) as info:
            equation4(Test_Equation_4_fail.okay_X + addx, Test_Equation_4_fail.okay_Y + addy, Test_Equation_4_fail.okay_ax + adda, Test_Equation_4_fail.okay_ay, Test_Equation_4_fail.okay_weights + addw, Test_Equation_4_fail.okay_biases)
            
class Test_Equation_4:
    
    @pytest.mark.parametrize('inputs', find_inputs('eq4'))
    @pytest.mark.parametrize('tp', ['lasso', 'ridge', 'elastic'])
    @pytest.mark.parametrize('const', [(0.7,0.7),(0.3,0.4)])
    def test_inputs(self, tp, inputs, const):
        X = inputs[0]
        Y = inputs[1]
        ax = inputs[2]
        ay = inputs[3]
        weights = inputs[4]
        biases = inputs[5]
        res = equation4(X, Y, ax, ay, weights, biases, type=tp, alpha=const[0], rho=const[1])