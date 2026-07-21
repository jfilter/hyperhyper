# Usage guide

`hyperhyper` builds count-based word embeddings (PPMI + SVD) for small corpora, an
implementation of Levy & Goldberg (2015). This guide documents the public API as
it exists in the code. For the "why", see the [README](../README.md); this is the
"how".

The workflow is always the same three steps:

1. Build a **`Corpus`** from your texts (preprocessing + vocabulary).
2. Persist it into a **`Bunch`** — an on-disk directory that caches every
   intermediate matrix and records evaluation results in a SQLite database.
3. Ask the bunch for an embedding with **`bunch.pmi(...)`** or **`bunch.svd(...)`**.

```python
import hyperhyper as hy

corpus = hy.Corpus.from_file("news.2010.en.shuffled")
bunch = hy.Bunch("news_bunch", corpus)

vectors, results = bunch.svd(keyed_vectors=True)
vectors.most_similar("berlin")
```

## Building a `Corpus`

A `Corpus` holds the preprocessed, tokenized texts plus a `Vocab` (the token ↔ id
mapping). There are four constructors; they differ only in where the text comes
from and which preprocessing function is the default.

| Constructor | Input | Default preprocessing |
| --- | --- | --- |
| `Corpus.from_file(input_path, limit=None, **kwargs)` | A single text file, one sentence per line | `tokenize_texts_parallel` (whitespace, no spaCy) |
| `Corpus.from_sents(texts, vocab=None, preproc_func=tokenize_texts_parallel, lang="en", **kwargs)` | A list of sentence strings | `tokenize_texts_parallel` |
| `Corpus.from_texts(texts, preproc_func=texts_to_sents, **kwargs)` | A list of documents (each split into sentences) | `texts_to_sents` (spaCy sentence splitting + lemmatization) |
| `Corpus.from_text_files(base_dir, preproc_func=texts_to_sents, view_fraction=1, lang="en", seed=1312, **kwargs)` | A folder of `*.txt` files | `texts_to_sents` |

Notes:

- **`from_file` / `from_sents`** default to the whitespace tokenizer
  (`tokenize_texts_parallel`), which needs no spaCy model. `from_file` reads the
  file, splits on line breaks, and passes the lines to `from_sents`; `limit` caps
  the number of lines read.
- **`from_texts` / `from_text_files`** default to `texts_to_sents`, which uses
  spaCy to split documents into sentences and (by default) lemmatizes and drops
  stop words. This requires the `full` extra and the `en_core_web_sm` model.
- **`from_text_files`** is for corpora too large to hold in memory: it reads a
  directory of `*.txt` files and keeps texts on disk. `view_fraction` lets the
  vocabulary be estimated from a random subset of the files (`seed` makes that
  sampling reproducible).
- **`lang`** (default `"en"`) selects which bundled evaluation datasets are used
  later and must match the language of your texts.
- **`**kwargs`** flow through to `Vocab.filter`, which controls the vocabulary:
  `no_below=0`, `no_above=1`, `keep_n=50000`, `keep_tokens=None` (the same
  arguments as gensim's `filter_extremes`). `keep_n` is the vocabulary-size cap;
  `hyperhyper` is designed for vocabularies up to ~50k.

## The `Bunch`

```python
Bunch(path, corpus=None, force_overwrite=False, text_chunk_size=None)
```

A `Bunch` is a directory on disk. Creating one with a `corpus` writes the corpus
and its text chunks under `path`. Reopening an existing bunch is
`Bunch(path)` — omit the corpus, and it is loaded back from `path/corpus.pkl`.

- `force_overwrite=True` replaces an existing bunch (it requires a `corpus`, so it
  cannot accidentally delete a bunch you meant to reopen).
- `text_chunk_size` is how many sentences go into one on-disk chunk; `None` (the
  default) sizes the chunks automatically from the corpus. Chunking is the unit of
  parallelism *and* part of the deterministic result, so leave it at the default
  unless you have a reason not to.
- `Bunch` is a context manager (`with hy.Bunch(...) as bunch:`), and `close()`
  disposes of the SQLite connection — useful in long parameter sweeps.

Everything a bunch computes is cached on disk, keyed by a digest of the *effective*
arguments (defaults included), so calling `bunch.svd(...)` twice with the same
parameters recomputes nothing.

## Getting embeddings: `pmi` and `svd`

### `bunch.pmi(...)`

```python
bunch.pmi(neg=1, cds=0.75, pair_args=None, keyed_vectors=False, evaluate=True, **kwargs)
```

Returns the (P)PMI embedding — the high-dimensional sparse representation.

### `bunch.svd(...)`

```python
bunch.svd(dim=500, eig=0, neg=1, cds=0.75, impl="scipy", impl_args=None,
          pair_args=None, keyed_vectors=False, evaluate=True, **kwargs)
```

Returns a low-dimensional dense embedding obtained by truncated SVD of the PPMI
matrix. This is usually what you want.

### Shared parameters

- **`neg`** (default `1`): number of negative samples; shifts the PMI matrix by
  `log(neg)` before the positive clip (PPMI).
- **`cds`** (default `0.75`): context distribution smoothing exponent applied to
  the context counts when forming the PMI matrix.
- **`pair_args`** (default `None`): a dict of arguments forwarded to `count_pairs`
  (see below). Loose keyword arguments in `**kwargs` are forwarded there too, so
  `bunch.svd(window=5)` and `bunch.svd(pair_args={"window": 5})` are equivalent.
- **`keyed_vectors`** (default `False`): if `True`, the embedding is converted to
  a gensim [`KeyedVectors`](https://radimrehurek.com/gensim/models/keyedvectors.html)
  object (so you get `.most_similar`, `.similarity`, etc.). If `False`, you get the
  internal embedding object (`PPMIEmbedding` or `SVDEmbedding`), which works in
  terms of integer token ids.
- **`evaluate`** (default `True`): if `True`, the embedding is scored on the
  bundled word-similarity datasets and the result is both returned and recorded in
  the bunch's database.

### `svd`-only parameters

- **`dim`** (default `500`): number of singular vectors kept — the embedding
  dimensionality.
- **`eig`** (default `0`): eigenvalue weighting exponent. `0` uses the left
  singular vectors unweighted (`ut`); `1` scales by the singular values; other
  values by `s ** eig`.
- **`impl`** (default `"scipy"`): the SVD backend. One of `"scipy"` (exact
  truncated SVD via `scipy.sparse.linalg.svds`), `"gensim"` (randomized), or
  `"scikit"` (randomized via scikit-learn, which needs the `full` extra).
- **`impl_args`** (default `None`): a dict of extra keyword arguments passed
  straight to the chosen backend.

### Return values

The return shape depends on `evaluate`:

- `evaluate=True` (default): returns a **tuple** `(embedding, eval_results)`.
- `evaluate=False`: returns just the `embedding`.

The `embedding` is either the internal object or a `KeyedVectors`, per
`keyed_vectors`.

```python
# internal embedding + scores
embedding, eval_results = bunch.svd()

# gensim KeyedVectors + scores
vectors, eval_results = bunch.svd(keyed_vectors=True)

# just the vectors, no evaluation
vectors = bunch.svd(keyed_vectors=True, evaluate=False)
```

## The evaluation result

`eval_results` (returned when `evaluate=True`) is a dict:

```python
{
    "micro": 0.42,      # line-weighted average score across datasets
    "macro": 0.39,      # dataset-weighted average score
    "results": [        # one entry per evaluation dataset
        {
            "name": "en_ws353",   # dataset name, prefixed with the language
            "score": 0.45,        # Spearman correlation with human judgements
            "oov": 0.08,          # fraction of pairs skipped as out-of-vocabulary
            "fullscore": 0.41,    # score penalized by the OOV fraction
        },
        ...
    ],
}
```

(The numbers above are illustrative.) `micro` / `macro` are `nan` when no dataset
had any in-vocabulary pair — the normal outcome when a small domain corpus does
not overlap the test data at all.

You can also evaluate an existing embedding directly:

- `bunch.eval_sim(embd)` → word-similarity results (the same dict shape above).
- `bunch.eval_analogy(embd)` → word-analogy results, same shape but `score` is an
  accuracy in `[0, 1]`.

## `count_pairs` parameters

These shape the co-occurrence matrix. Pass them to `pmi` / `svd` as loose keyword
arguments or inside `pair_args`. Full signature:

```python
count_pairs(corpus, window=2, dynamic_window="deter", decay_rate=0.25,
            delete_oov=True, subsample="deter", subsample_factor=1e-5,
            seed=1312, min_count=0)
```

- **`window`** (default `2`): maximum distance between two tokens for them to count
  as a co-occurring pair.
- **`dynamic_window`** (default `"deter"`): how within-window pairs are weighted.
  `None` counts every in-window pair as 1; `"deter"` weights each pair
  deterministically by `(window + 1 - distance) / window`; `"prob"` draws a random
  effective window per token; `"decay"` weights by exponential decay of distance
  (rate `decay_rate`).
- **`decay_rate`** (default `0.25`): the decay rate used only when
  `dynamic_window="decay"`.
- **`delete_oov`** (default `True`): drop out-of-vocabulary tokens before counting,
  so window spans close over the gaps rather than counting through them.
- **`subsample`** (default `"deter"`): frequent-word subsampling. `None` disables
  it; `"deter"` scales pair counts by the word2vec keep factor deterministically;
  `"prob"` drops tokens at random with that keep probability (the two agree in
  expectation).
- **`subsample_factor`** (default `1e-5`): word2vec's `sample` parameter. A word is
  subsampled once its count exceeds `subsample_factor * total_tokens`; the
  threshold is on the token-count scale, so a value from a word2vec/hyperwords
  setup carries over unchanged.
- **`seed`** (default `1312`): seed for the randomized modes (`dynamic_window="prob"`,
  `subsample="prob"`), which makes them reproducible.
- **`min_count`** (default `0`): prune matrix entries whose count is below this,
  which can greatly reduce memory.

The deterministic modes draw no random numbers, so runs with identical parameters
produce identical results — down to bit-identity for configurations whose counts
are exact in float32.

## Querying past runs: `bunch.results(...)`

Every call with `evaluate=True` writes a row into the bunch's `results.db`. Query
it back with:

```python
bunch.results(query=None, order="micro_results desc", limit=100)
```

Returns a list of row dicts (best first by default). Each row contains the run's
parameters — including the resolved `pair_args__*` columns — and score columns
such as `micro_results`, `macro_results`, and per-dataset `<name>_score`,
`<name>_oov`, `<name>_fullscore`.

- **`query`** filters by exact parameter match, e.g. `query={"dim": 500}` or
  `query={"impl": "scipy"}`. Nested parameters use the flattened form, e.g.
  `query={"pair_args": {"window": 2}}`.
- **`order`** is a `column [asc|desc]` expression (validated against a strict
  whitelist).
- **`limit`** caps the number of rows (`None` for no limit).

This makes it easy to sweep parameters and then pull back the best-scoring
configurations without recomputing anything.
