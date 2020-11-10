
# @pytest.mark.parametrize('type', ALGORITHM_TYPE_SMALL)
# @pytest.mark.parametrize('file', find_inputs('Input_dataset'))
# def test_dataset(file, type):
#     inp = read_json(file)
#     test = xiao2018(type=type, max_iter=10)
#     res = test.run(*inp)
#     print(res - np.array(inp[2]))
#     assert res.shape[0] == len(inp[2]) and res.shape[1] == len(inp[2][0])

# @pytest.mark.skip(reason='Too large to run.')
# @pytest.mark.parametrize('type', ALGORITHM_TYPE_SMALL)
# @pytest.mark.parametrize('file', find_inputs('Input_large_dataset'))
# def test_large_dataset(file, type):
#     inp = read_json(file)
#     test = xiao2018(type=type, max_iter=10)
#     res = test.run(*inp)
#     print(res - np.array(inp[2]))
#     assert res.shape[0] == len(inp[2]) and res.shape[1] == len(inp[2][0])