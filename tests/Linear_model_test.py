from poisoning.xiao2018.equations import equation2
import sklearn.linear_model as LM
import pytest

class Test_Equation_2_fail:
    
    okay_X = [[1, 2], [3, 4]]
    okay_Y = [1, 2]
    okay_lambda = 0.23
    
    @pytest.mark.parametrize('param', [{'this':3}, {'test123':332}, {'other':'haha'}, {'thing':'t'}])
    def test_parameters(self, param):
        with pytest.raises(TypeError) as info:
            equation2(Test_Equation_2_fail.okay_X, Test_Equation_2_fail.okay_Y, **param)        
    
    @pytest.mark.parametrize('tp', ['notlasso', 3, 92, 'other', 9.2, True])
    def test_type(self, tp):
        with pytest.raises(TypeError) as info:
            equation2(Test_Equation_2_fail.okay_X, Test_Equation_2_fail.okay_Y, type=tp)
    
    @pytest.mark.parametrize('addx,addy', [([],[3]),([[1,2]],[]),([[1,2]],[3, 4])])
    def test_dimensions(self, addx, addy):
        with pytest.raises(ValueError) as info:
            equation2(Test_Equation_2_fail.okay_X + addx, Test_Equation_2_fail.okay_Y + addy)

class Test_Equation_2:
    
    okay_X = [[1, 2], [3, 4]]
    okay_Y = [1, 2]
    
    @pytest.mark.parametrize('alph', [1.0, 0.23])
    @pytest.mark.parametrize('tp', ['lasso', 'ridge', 'elastic'])
    def test_results(self, tp, alph):
        res = equation2(Test_Equation_2.okay_X, Test_Equation_2.okay_Y, type=tp, alpha=alph)
        alg = LM.Lasso(alpha=alph) if tp == 'lasso' else LM.Ridge(alpha=alph) if tp == 'ridge' else LM.ElasticNet(alpha=alph)
        
        accurate = alg.fit(Test_Equation_2.okay_X, Test_Equation_2.okay_Y)
        accurate = (accurate.coef_, accurate.intercept_)
        
        assert all([r == a for (r, a) in zip(res[0], accurate[0])])
        assert res[1], accurate[1]