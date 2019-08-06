# hyperhyper

## Installation

If you have an Intel CPU, it's recommended to use the MKL library for performance reason for numpy.

```bash
conda install -c intel intelpython3_core
```

Verify mkl_info is not None

```python
>>> import numpy
>>> numpy.__config__.show()
```

## Usage

like this!

## Scientific Background

This software is based on the following papers:

-   Improving Distributional Similarity with Lessons Learned from Word Embeddings, Omer Levy, Yoav Goldberg, Ido Dagan, TACL 2015. [Paper](https://aclweb.org/anthology/papers/Q/Q15/Q15-1016/) [Code](https://bitbucket.org/omerlevy/hyperwords)
    > Recent trends suggest that neural-network-inspired word embedding models outperform traditional count-based distributional models on word similarity and analogy detection tasks. We reveal that much of the performance gains of word embeddings are due to certain system design choices and hyperparameter optimizations, rather than the embedding algorithms themselves. Furthermore, we show that these modifications can be transferred to traditional distributional models, yielding similar gains. In contrast to prior reports, we observe mostly local or insignificant performance differences between the methods, with no global advantage to any single approach over the others.
-   The Influence of Down-Sampling Strategies on SVD Word Embedding Stability, Johannes Hellrich, Bernd Kampe, Udo Hahn, NAACL 2019. [Paper](https://aclweb.org/anthology/papers/W/W19/W19-2003/) [Code](https://github.com/hellrich/hyperwords) [Code](https://github.com/hellrich/embedding_downsampling_comparison)
    > The stability of word embedding algorithms, i.e., the consistency of the word representations they reveal when trained repeatedly on the same data set, has recently raised concerns. We here compare word embedding algorithms on three corpora of different sizes, and evaluate both their stability and accuracy. We find strong evidence that down-sampling strategies (used as part of their training procedures) are particularly influential for the stability of SVD-PPMI-type embeddings. This finding seems to explain diverging reports on their stability and lead us to a simple modification which provides superior stability as well as accuracy on par with skip-gram embedding

I tried to the port the software to Python3. But I realized that it too hard so I created this package. You can find my verion on Github: https://github.com/jfilter/hyperwords

But I extracted only the parts tha

For systems using OpenBLAS, I highly recommend setting `export OPENBLAS_NUM_THREADS=1`. This disables its internal multithreading ability, which leads to substantial speedups for this package. Likewise for Intel MKL, setting `export MKL_NUM_THREADS=1` should also be set.

Remove pmi and svd folders
`find . -name pmi -exec rm -rf {} \;`

Remove pmi and svd folders
`find . -name pmi -exec rm -rf {} \;`

## Development

1. Install [pipenv](https://docs.pipenv.org/en/latest/).
2. ```bash
   git clone https://github.com/jfilter/hyperhyper && cd hyperhyper &&
   pipenv install
   ```
3. ```bash
   pytest tests
   ```

## License

BSD-2-Clause.
