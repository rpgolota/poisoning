from poisoning.xiao2018.equations import equation2
import sklearn.linear_model as LM
import numpy as np
import pytest
from tests.shared import find_inputs

class Test_Equation_2_fail:
    
    @pytest.mark.parametrize('inputs', find_inputs('eq2'))
    @pytest.mark.parametrize('param', [{'this':3}, {'test123':332}, {'other':'haha'}, {'thing':'t'}])
    def test_parameters(self, param, inputs):
        X = inputs[0]
        Y = inputs[1]
        with pytest.raises(TypeError) as info:
            equation2(X, Y, **param)
    
    @pytest.mark.parametrize('inputs', find_inputs('eq2'))
    @pytest.mark.parametrize('tp', ['notlasso', 3, 92, 'other', 9.2, True])
    def test_type(self, tp, inputs):
        X = inputs[0]
        Y = inputs[1]
        with pytest.raises(TypeError) as info:
            equation2(X, Y, type=tp)
    
    @pytest.mark.parametrize('inputs', find_inputs('eq2'))
    @pytest.mark.parametrize('addx,addy', [([],[3]),([[1,2]],[]),([[1,2]],[3, 4])])
    def test_dimensions(self, addx, addy, inputs):
        X = inputs[0]
        Y = inputs[1]
        with pytest.raises(ValueError) as info:
            equation2(np.array(X + addx, dtype=object), np.array(Y + addy, dtype=object))

class Test_Equation_2:
    
    @pytest.mark.parametrize('inputs', find_inputs('eq2'))
    @pytest.mark.parametrize('alph', [1.0, 0.33, 0.72]) # less than 0.33 alpha caused convergence warnings... might need to increase iteration amout or increase the tolerance
    @pytest.mark.parametrize('tp', ['lasso', 'ridge', 'elastic'])
    def test_results(self, tp, alph, inputs):
        X = inputs[0]
        Y = inputs[1]
        res = equation2(X, Y, type=tp, alpha=alph)
        alg = LM.Lasso(alpha=alph) if tp == 'lasso' else LM.Ridge(alpha=alph) if tp == 'ridge' else LM.ElasticNet(alpha=alph)
        accurate = alg.fit(X, Y)
        accurate = (accurate.coef_, accurate.intercept_)
        
        assert all([r == a for (r, a) in zip(res[0], accurate[0])])
        assert res[1], accurate[1]