import os
import json

def find_inputs(name):
    path = os.path.normcase(os.getcwd() + '/tests/inputs/')
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return [load_inputs(os.path.join(path, this_file)) for this_file in onlyfiles if this_file.find(name) != -1]

def load_inputs(filename):
    with open(filename, 'r') as f:
        return json.load(f)