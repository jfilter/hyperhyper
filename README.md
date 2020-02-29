# hyperhyper [![Build Status](https://travis-ci.com/jfilter/hyperhyper.svg?branch=master)](https://travis-ci.com/jfilter/hyperhyper) [![PyPI](https://img.shields.io/pypi/v/hyperhyper.svg)](https://pypi.org/project/hyperhyper/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hyperhyper.svg)](https://pypi.org/project/hyperhyper/)

Python Library to Construct Word Embeddings for Small Data. Still work in progress.

Building upon the work by Omer Levy et al. for [Hyperwords](https://bitbucket.org/omerlevy/hyperwords).

## Why?

Nowadays, [word embeddings](https://en.wikipedia.org/wiki/Word_embedding) are mostly associated with [Word2vec](https://en.wikipedia.org/wiki/Word2vec) or [fastText](https://en.wikipedia.org/wiki/FastText). Those approaches focus on scenarios, where an abundance of data is available. But to make them work, you also need a lot of data. This is not always the case. There exists alternative methods based on counting word pairs and some math magic around matrix operations. They need less data. This Python library implements the approaches (somewhat) efficiently (but there is there is still room for improvement.)

`hyperhyper` is based on [a paper](https://aclweb.org/anthology/papers/Q/Q15/Q15-1016/) from 2015. The authors, Omer Levy et al., published their research code as [Hyperwods](https://bitbucket.org/omerlevy/hyperwords).
I [tried](https://github.com/jfilter/hyperwords) to the port their original software to Python 3 but I ended up re-writing large parts of it. So this library was born.

Limitations: With `hyperhyper` you will run into (memory) problems, if you need large vocabularies (set of possible words). It's fine if you have a vocabulary up until 50k. Word2vec and fastText especially solve this [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality).

## Installation

```bash
pip install hyperhyper
```

If you have an Intel CPU, it's recommended to use the MKL library for `numpy`. It can be challening to correctly set up MKL. A package by intel may help you.

```bash
conda install -c intel intelpython3_core
pip install hyperhyper
```

Verify wheter `mkl_info` is present:

```python
>>> import numpy
>>> numpy.__config__.show()
```

Disable internal multithreading ability of MKL or OpenBLAS.

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

This speeds up computation because we are using multiprocessing on an outer loop.

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

See [examples](./examples) for more.

The general concepts:

-   Preprocess data once and save them in a `bunch`
-   Cache all results and also record their perfomance on test data

More documenation may be forthcoming. Until then you have to read the [source code](./hyperhyper).

## Scientific Background

This software is based on the following papers:

-   Improving Distributional Similarity with Lessons Learned from Word Embeddings, Omer Levy, Yoav Goldberg, Ido Dagan, TACL 2015. [Paper](https://aclweb.org/anthology/papers/Q/Q15/Q15-1016/) [Code](https://bitbucket.org/omerlevy/hyperwords)
    > Recent trends suggest that neural-network-inspired word embedding models outperform traditional count-based distributional models on word similarity and analogy detection tasks. We reveal that much of the performance gains of word embeddings are due to certain system design choices and hyperparameter optimizations, rather than the embedding algorithms themselves. Furthermore, we show that these modifications can be transferred to traditional distributional models, yielding similar gains. In contrast to prior reports, we observe mostly local or insignificant performance differences between the methods, with no global advantage to any single approach over the others.
-   The Influence of Down-Sampling Strategies on SVD Word Embedding Stability, Johannes Hellrich, Bernd Kampe, Udo Hahn, NAACL 2019. [Paper](https://aclweb.org/anthology/papers/W/W19/W19-2003/) [Code](https://github.com/hellrich/hyperwords) [Code](https://github.com/hellrich/embedding_downsampling_comparison)
    > The stability of word embedding algorithms, i.e., the consistency of the word representations they reveal when trained repeatedly on the same data set, has recently raised concerns. We here compare word embedding algorithms on three corpora of different sizes, and evaluate both their stability and accuracy. We find strong evidence that down-sampling strategies (used as part of their training procedures) are particularly influential for the stability of SVD-PPMI-type embeddings. This finding seems to explain diverging reports on their stability and lead us to a simple modification which provides superior stability as well as accuracy on par with skip-gram embedding

## Development

1. Install [pipenv](https://docs.pipenv.org/en/latest/).
2. `git clone https://github.com/jfilter/hyperhyper && cd hyperhyper && pipenv install && pipenv shell`
3. `python -m spacy download en_core_web_sm`
4. `pytest tests`

## Contributing

If you have a **question**, found a **bug** or want to propose a new **feature**, have a look at the [issues page](https://github.com/jfilter/hyperhyper/issues).

**Pull requests** are especially welcomed when they fix bugs or improve the code quality.

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
