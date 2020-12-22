import pandas as pd
from poisoning import xiao2018

def test_sanity():
    
    dataset = pd.read_csv('tests/input/balloons.csv', sep=",", header=None)

    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,-1].values

    model = xiao2018(type='elastic')
    # poison 10 percent of the dataset, with a boundary box of 0 to 2 for all features
    poisoned, labels = model.autorun(X, Y, 0.1, (0,2))
    
if __name__ == "__main__":
    test_sanity()