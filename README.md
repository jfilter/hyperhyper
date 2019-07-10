# hyper-hyper

# Installation

If you have an Intel CPU, it's recommended to use the MKL library for performance reason for numpy.

```
conda install -c intel intelpython3_core
```

Verify mkl_info is not None

```python
>>> import numpy
>>> numpy.__config__.show()
```
