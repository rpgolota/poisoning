from pBench import *
import poisoning
import json

def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

files = {'files/Input_dataset_musk-1.json':('musk1', 1000),
         'files/Input_dataset_musk-2.json':('musk2', 1000),
         'files/Input_large_dataset_miniboone.json':('miniboone', 1)}

projections = [1]

for file in files:
    for projection in projections:
        X, Y, Attacks, Labels = read_json(file)
        with pBench() as b:
            A = poisoning.xiao2018(max_iter=files[file][1])
            A.run(X, Y, Attacks, Labels, 1)

        b.write(f'{files[file][0]}_P{projection}.txt')