# FINDER

-----------
by Andreas Nold and Pietro Verzelli


`FINDER` is a (meta)-clustering algorithm developed for Single Molecule Localization Microscopy (SMLM).
It was introduced in our paper 
['Unbiased choice of global clustering parameters for single-molecule localization microscopy'](https://www.nature.com/articles/s41598-022-27074-1).


Please consider citing it:

```
@article{verzelli2022unbiased,
  title={Unbiased choice of global clustering parameters for single-molecule localization microscopy},
  author={Verzelli, Pietro and Nold, Andreas and Sun, Chao and Heilemann, Mike and Schuman, Erin M and Tchumatchenko, Tatjana},
  journal={Scientific Reports},
  volume={12},
  number={1},
  pages={22561},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```

For some examples of its usage and all the code used to generate the experiments of the paper, you can refer to 
[this repo](https://github.com/NoldAndreas/FINDER).

# Installation

----------------

The simplest way to install Spektral is from PyPi:
```shell
pip install finder_smlm 
```

To install FINDER on Google Colab:

```python
! pip install finder_smlm
```

## Using FINDER

---------------

Using `FINDER` is really simple.
To run the `FINDER` algorithm on your own localization data, add

```python
from finder import Finder

FD      = Finder()
labels  = FD.fit(XC)
result_ = FD.selected_parameters
```

to your code, analogous to DBSCAN in the `sklearn.cluster` package.
FINDER will choose global clustering parameters according to the overall noise levels / the robustness detected in the dataset.
