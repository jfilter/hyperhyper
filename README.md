# hyperhyper [![Build Status](https://travis-ci.com/jfilter/hyperhyper.svg?branch=master)](https://travis-ci.com/jfilter/hyperhyper) [![PyPI](https://img.shields.io/pypi/v/hyperhyper.svg)](https://pypi.org/project/hyperhyper/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hyperhyper.svg)](https://pypi.org/project/hyperhyper/)

Python library for count-based word embeddings. The easiest way to construct word embeddings from _small data_. Still work in progress.

Building upon the work by Omer Levy et al. for [Hyperwords](https://bitbucket.org/omerlevy/hyperwords).

## Why?

Training word embeddings on _small_ datasets should be easy. Even though there exists efficient library for creating word embeddings (word2vec, gensim), they focus on large amounts of data. And: They only construct word embeddings _if_ you have enough data. But there exists alternative methods, based on counting and matrix magic. But so far, there wasn't an efficient implementation. This Python library goes into the direction. Although there is still enough possibles for more efficient programming.

The other methods exist, because you will run into memory issues for large vocabularies, e.g. 200k words. But for most cases, smaller vocabularies such as 25k or 50k are enough.

## Installation

```bash
pip install hyperhyper
```

If you have an Intel CPU, it's recommended to use the MKL library for performance reasons. The most easy way: Install numpy with this packages that intel provides.

```bash
conda install -c intel intelpython3_core
pip install hyperhyper
```

Verify mkl_info is not None:

```python
>>> import numpy
>>> numpy.__config__.show()
```

For systems using OpenBLAS, I highly recommend setting `export OPENBLAS_NUM_THREADS=1`. This disables its internal multithreading ability, which leads to substantial speedups for this package. Likewise for Intel MKL, `export MKL_NUM_THREADS=1` should be set.

## Usage

```python
import hyperhyper as hy

corpus = hy.Corpus.from_file('news.2010.en.shuffled')
bunch = hy.Bunch("news_bunch", corpus)
vectors, results = bunch.svd(keyed_vectors=True)

results['results'][1]
>>> {'name': 'en_ws353',
 'score': 0.6510955349164682,
 'oov': 0.014164305949008499,
 'fullscore': 0.641873218557878}

vectors.most_similar('berlin')
>>> [('vienna', 0.6323208808898926),
 ('frankfurt', 0.5965485572814941),
 ('munich', 0.5737138986587524),
 ('amsterdam', 0.5511572360992432),
 ('stockholm', 0.5423270463943481)]
```

See [examples](./examples) for more. (TODO)

## Scientific Background

This software is based on the following papers:

-   Improving Distributional Similarity with Lessons Learned from Word Embeddings, Omer Levy, Yoav Goldberg, Ido Dagan, TACL 2015. [Paper](https://aclweb.org/anthology/papers/Q/Q15/Q15-1016/) [Code](https://bitbucket.org/omerlevy/hyperwords)
    > Recent trends suggest that neural-network-inspired word embedding models outperform traditional count-based distributional models on word similarity and analogy detection tasks. We reveal that much of the performance gains of word embeddings are due to certain system design choices and hyperparameter optimizations, rather than the embedding algorithms themselves. Furthermore, we show that these modifications can be transferred to traditional distributional models, yielding similar gains. In contrast to prior reports, we observe mostly local or insignificant performance differences between the methods, with no global advantage to any single approach over the others.
-   The Influence of Down-Sampling Strategies on SVD Word Embedding Stability, Johannes Hellrich, Bernd Kampe, Udo Hahn, NAACL 2019. [Paper](https://aclweb.org/anthology/papers/W/W19/W19-2003/) [Code](https://github.com/hellrich/hyperwords) [Code](https://github.com/hellrich/embedding_downsampling_comparison)
    > The stability of word embedding algorithms, i.e., the consistency of the word representations they reveal when trained repeatedly on the same data set, has recently raised concerns. We here compare word embedding algorithms on three corpora of different sizes, and evaluate both their stability and accuracy. We find strong evidence that down-sampling strategies (used as part of their training procedures) are particularly influential for the stability of SVD-PPMI-type embeddings. This finding seems to explain diverging reports on their stability and lead us to a simple modification which provides superior stability as well as accuracy on par with skip-gram embedding

I tried to the port the orignal software "Hyperwords" to Python 3 but I ended re-writing large parts of it. However, here is my version on Github: https://github.com/jfilter/hyperwords

## Development

1. Install [pipenv](https://docs.pipenv.org/en/latest/).
2. `git clone https://github.com/jfilter/hyperhyper && cd hyperhyper && pipenv install && pipenv shell`
3. `python -m spacy download en_core_web_sm`
4. `pytest tests`

## Future Work / TODO

-   evaluation for analogies
-   replace pipenv if they still don't ship any newer release
-   implement counting in a more efficient programming language, e.g. Cython.

## Why is this library named `hyperhyper`?

[![Scooter – Hyper Hyper (Song)](https://img.youtube.com/vi/7Twnmhe948A/0.jpg)](https://www.youtube.com/watch?v=7Twnmhe948A "Scooter – Hyper Hyper")

## License

BSD-2-Clause.

## Sponsoring

This work was created as part of a [project](https://github.com/jfilter/ptf) that was funded by the German [Federal Ministry of Education and Research](https://www.bmbf.de/en/index.html).

<img src="./bmbf_funded.svg">
