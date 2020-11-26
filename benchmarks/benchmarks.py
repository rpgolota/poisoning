import warnings
warnings.filterwarnings('ignore')

from pBench import pBench_fast
import poisoning
import json
import argparse
import alive_progress
import time
import os
import csv
from datetime import datetime


# gets the command line arguments for the benchmarks
def get_arguments():
    parser = argparse.ArgumentParser(description='Run benchmarks for poisoning.')
    parser.add_argument('argfiles',
                        nargs='+',
                        action='append',
                        help='Provides information about running benchmarks.')
    parser.add_argument('-o',
                        '--out',
                        dest='out',
                        nargs=1,
                        type=str,
                        default=None,
                        help='Output filename.')
    parser.add_argument('-p',
                        '--prefix',
                        dest='prefix',
                        action='store_true',
                        help='Prefix output filename with date and time. Default is true if no output filename is provided.')

    args = parser.parse_args()
        
    args.argfiles = ['files/Argfile_1.json'] if not args.argfiles else sum(args.argfiles, [])
   
    return args

# reads json file contianing the information about how to run the benchmark
def read_argfile(filename):
    types = {'xiao':[poisoning.xiao2018], 'fred':[poisoning.frederickson2018], 'both':[poisoning.xiao2018, poisoning.frederickson2018]}
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    data['model'] = types[data['model']]
    
    return data

def read_argfiles(argfiles):
    data = []
    for argfile in argfiles:
        data.append(read_argfile(argfile))
    return data

def get_data(dataset, attacks):
    with open(dataset, 'r') as f:
        X, Y, _, _ = json.load(f)
    
    if type(attacks) is list:
        Attacks = X[attacks[0]:attacks[1]]
        Labels = Y[attacks[0]:attacks[1]]
    elif type(attacks) is int:
        attacks = int(len(X) * (attacks/100))
        Attacks = X[0:attacks]
        Labels = Y[0:attacks]

    return X, Y, Attacks, Labels, (len(X), len(X[0])), attacks

# run the benchmark by creating the necessary class and running the correct type of function, timing it and returning time
# convert information into data that can be written into csv
def run_benchmark(class_type, dataset, attacks, projection, arguments):
    
    X, Y, Attacks, Labels, dataset_size, n_attacks = get_data(dataset, attacks)
    
    with pBench_fast() as bench:
        model = class_type(**arguments)
        model.run(X, Y, Attacks, Labels, projection)
    
    return dataset, 'xiao2018' if class_type == poisoning.xiao2018 else 'frederickson2018', dataset_size[0], dataset_size[1], len(Attacks), projection, bench.get_time()

class bench_results:
    def __init__(self, filename, prefix=False):
        if filename:
            self.filename = datetime.now().strftime("[%d-%m-%Y][%H.%M.%S]") + '_' if prefix else '' + filename
        else:
            self.filename = datetime.now().strftime("[%d-%m-%Y][%H.%M.%S]") + '_poisoning_benchmark'
        self.data = []
        
    def add(self, data):
        self.data.append(data)
        
    def write(self):
        with open(self.filename + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Dataset", "Type", "Samples", "Features", "Attacks", "Projection", "Time"])
            writer.writerows(self.data)

def main():
    args = get_arguments()
    
    argfile_data = read_argfiles(args.argfiles)
    results = bench_results(args.out, args.prefix)
    bar_iterations = sum([len(aData['datasets']) * 
                          len(aData['model']) * 
                          len(aData['attacks']) *
                          len(aData['model_args']) * 
                          len(aData['projections']) * 
                          aData['iter'] 
                          for aData in argfile_data])
    
    paths = [os.path.dirname(os.path.realpath(f)) for f in args.argfiles]
    with alive_progress.alive_bar(bar_iterations) as bar:
        for data, f_path in zip(argfile_data, paths):
            for i in range(data['iter']):
                for type_args in data['model_args']:
                    for dataset in data['datasets']:
                        for type in data['model']:
                            for projection in data['projections']:
                                for attack in data['attacks']:
                                    results.add(run_benchmark(type, 
                                                          os.path.join(f_path, dataset), 
                                                          attack, 
                                                          projection, 
                                                          type_args))
                                bar()

    results.write()

if __name__ == "__main__":
    main()