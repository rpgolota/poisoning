from pBench import *
import poisoning
import json

def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    algorithms = {'xiao': poisoning.xiao2018}

    folder = 'fast'

    files = {'files/Input_dataset_musk-1.json':'musk1'}
    # files = {'files/Input_large_dataset_miniboone.json':'miniboone'}

    projections = [1,2,3,4,5]
    num_attacks = [10,20,30,40,50,60,70,80,90,100]

    for algorithm in algorithms:
        for file in files:
            for projection in projections:
                for attack in num_attacks:
                    X, Y, _, _ = read_json(file)
                    with pBench_fast() as b:
                        A = algorithms[algorithm](max_iter=1)
                        A.autorun(X, Y, attack, projection)

                    sample_size = len(Y)
                    feature_size = len(X[0])
                    
                    text = f'File: {file}\nSamples: {sample_size}   Features: {feature_size}\n\n'
                    filename = f'{files[file]}_A{attack}_P{projection}.txt'
                    
                    with open(f'{folder}/{algorithm}/' + filename, 'w') as f:
                        f.write(text)
                        
                    b.write(f'{folder}/{algorithm}/' + filename, flag='a')

if __name__ == '__main__':
    main()