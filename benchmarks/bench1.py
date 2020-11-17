from pBench import *
import poisoning
import json

def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

algorithms = {'fred': poisoning.frederickson2018}

files = {'files/Input_dataset_musk-1.json':'musk1',
         'files/Input_dataset_musk-2.json':'musk2',
         'files/Input_large_dataset_miniboone.json':'miniboone'}

projections = [1]
num_attacks = [1, 2, 3, 4, 5]

for algorithm in algorithms:
    for file in files:
        for projection in projections:
            for attack in num_attacks:
                X, Y, _, _ = read_json(file)
                with pBench() as b:
                    A = algorithms[algorithm](max_iter=1)
                    A.autorun(X, Y, attack, projection)

                sample_size = len(Y)
                feature_size = len(X[0])
                
                text = f'File: {file}\nSamples: {sample_size}   Features: {feature_size}\n\n'
                filename = f'{files[file]}_A{attack}_P{projection}.txt'
                
                with open(f'{algorithm}/' + filename, 'w') as f:
                    f.write(text)
                    
                b.write(f'{algorithm}/' + filename, flag='a')