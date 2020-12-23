Introduction
============

Poisoning is a Python library for poisoning datasets using gradient ascent, targeting feature selection.

Requirements
------------

Rich works with OSX, Linux and Windows.

Rich requires Python 3.8 and above.

Installation
------------

You can install poisoning by downloading from `github <https://github.com/rpgolota/poisoning>`_, and then installing with pip::

    pip install poisoning

A shortcut is to run the following command::

    pip install git+https://github.com/rpgolota/poisoning/

Quick Start
-----------

The quickest way to get started is to simply import the required model form poisoning, and then run it with autorun on a dataset::

    import pandas as pd
    from poisoning import xiao2018

    dataset = pd.read_csv('spect_test.csv', sep=",", header=None)

    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,-1].values

    model = xiao2018(type='elastic')
    # poison 10 percent of the dataset, with a boundary box of 0 to 2 for all features
    poisoned, labels = model.autorun(X, Y, 0.1, (0,2))

The recommended way of extracting the Samples (X) and labels (Y) from a csv file is using pandas for a quick start.

Multiprocessing
---------------

Multiprocessing is enabled by default, but can be disabled by setting the parallel flag to False::
    
    model = xiao2018(parallel=False)


.. note::
    Windows users must make use of a name guard for multiprocessing.

::

    if __name__ == '__main__':
        model = xiao2018()