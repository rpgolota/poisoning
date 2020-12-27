<h1 align="center">Dataset Poisoning</h2>
<h4 align="center">Poisoning datasets using gradient ascent, targeting feature selection.</h2>

<p align="center">
<a href="https://github.com/rpgolota/poisoning/actions?query=workflow%3A%22Code+Test%22"><img alt="Build Status" src="https://github.com/rpgolota/poisoning/workflows/Code%20Test/badge.svg"></a>
<a href="https://github.com/rpgolota/poisoning/actions?query=workflow%3A%22Pip+Test%22"><img alt="Actions Status" src="https://github.com/rpgolota/poisoning/workflows/Pip%20Test/badge.svg"></a>
<a href="https://github.com/rpgolota/poisoning/actions?query=workflow%3ALint"><img alt="Actions Status" src="https://github.com/rpgolota/poisoning/workflows/Lint/badge.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

<p align="center">
<b><a href="https://rpgolota.github.io/poisoning/build/html/index.html">Documentation</a></b>
</p>

---

### Examples
The following is an example of using xiao2018 to poison a gaussian distribution.

![Poisoning Example](examples/poisoning_example_xiao.png)

The following is an example of using frederickson2018 to poison a gaussian distribution.

![Poisoning Example](examples/poisoning_example_frederickson.png)

### Installation

You can install poisoning by cloning from this repository, and then installing with pip:

```console
pip install poisoning
```

A shortcut is to run the following command:

```console
pip install git+https://github.com/rpgolota/poisoning/
```

### Get started
```python
import pandas as pd
from poisoning import xiao2018

dataset = pd.read_csv('spect_test.csv', sep=",", header=None)

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

model = xiao2018(type='elastic')
# poison 10 percent of the dataset, with a boundary box of 0 to 2 for all features
poisoned, labels = model.autorun(X, Y, 0.1, (0,2))
```

##### Windows
If you are on windows, make sure to use a name guard if multiprocessing is enabled (default),

```python
if __name__ == '__main__':
    model = xiao2018()
    ...
```
or disable multiprocessing altogether.
```python
model = xiao2018(parallel=False)
...
```
