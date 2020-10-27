import os
import json

SEARCH_PATH = os.path.normcase(os.getcwd() + '/tests/inputs/')

def find_inputs(name):
    onlyfiles = [f for f in os.listdir(SEARCH_PATH) if os.path.isfile(os.path.join(SEARCH_PATH, f))]
    return [this_file for this_file in onlyfiles if this_file.find(name) != -1]

def get_inputs(name):
    return [read_json(this_file) for this_file in find_inputs(name)]

def read_json(filename):
    with open(os.path.join(SEARCH_PATH, filename), 'r') as f:
        return json.load(f)