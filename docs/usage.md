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
from. **All four now default to the same lightweight tokenizer**
(`tokenize_texts_parallel_v2`, no spaCy).

| Constructor | Input | Default preprocessing |
| --- | --- | --- |
| `Corpus.from_file(input_path, limit=None, **kwargs)` | A single text file, one sentence per line | `tokenize_texts_parallel_v2` (whitespace, no spaCy) |
| `Corpus.from_sents(texts, vocab=None, preproc_func=tokenize_texts_parallel_v2, lang="en", **kwargs)` | A list of sentence strings | `tokenize_texts_parallel_v2` |
| `Corpus.from_texts(texts, preproc_func=tokenize_texts_parallel_v2, **kwargs)` | A list of documents | `tokenize_texts_parallel_v2` |
| `Corpus.from_text_files(base_dir, preproc_func=tokenize_texts_parallel_v2, view_fraction=1, lang="en", seed=1312, **kwargs)` | A folder of `*.txt` files | `tokenize_texts_parallel_v2` |

> **Breaking change (ADR 0002).** `from_texts` and `from_text_files` used to
> default to `texts_to_sents` (spaCy sentence-splitting + lemmatization, needs a
> model). They now default to the lightweight `tokenize_texts_parallel_v2`, so the
> package is "lightweight by default" on every constructor. This changes the
> vocabulary — and therefore every similarity/analogy number — for **new** corpora
> built with the default; existing bunches are unaffected (they pickle their
> tokenizer by reference, under its old name). The spaCy path is unchanged and
> still available: pass `preproc_func=texts_to_sents` explicitly. Note this for the
> CHANGELOG as a breaking evaluation change.

### The v2 tokenizer

`tokenize_texts_parallel_v2` applies `tokenize_string_v2(text, lower=True,
normalize_digits=False)` per line. Compared with the legacy v1 tokenizer
(`tokenize_string`, still the default only for bunches built before this change),
v2:

- **NFC-normalizes first**, so an accented word written as decomposed Unicode
  (`café` as `cafe` + a combining accent) is no longer silently split into a
  separate, accent-stripped vocab entry.
- **Canonicalizes typographic variants** — the curly apostrophe `’` (U+2019) and
  the Unicode hyphens/dashes (U+2010, U+2011, …) become ASCII `'` and `-`.
- **Extracts rather than destroys**: `city's`, `ice-cream` and `don't` stay whole
  (v1 shattered them into `city`/`s`, `ice`/`cream`). Everything else still splits
  on non-word characters. Dotted forms like `U.S.A.` are *not* rescued — adding
  `.` to the joiner would glue `word.Next` together in scraped text.
- **Keeps digits by default.** `normalize_digits=False` (the default) leaves `2001`
  as `2001`. Pass `normalize_digits=True` to restore the legacy digit→`0` behaviour
  — this is the Levy-Goldberg-Dagan / hyperwords convention this package
  reimplements, useful when numerals should collapse into one token.

`tokenize_texts_parallel_v2` uses the defaults (`lower=True`,
`normalize_digits=False`). To run v2 with non-default options, supply a small
top-level wrapper as `preproc_func` (see the `preproc_func` contract below) — not
a `lambda` or `functools.partial`, which are not picklable by reference.

Notes:

- **`from_file`** reads the file, splits on line breaks, and passes the lines to
  `from_sents`; `limit` caps the number of lines read.
- **`texts_to_sents`** (the spaCy path) is still a first-class, opt-in
  `preproc_func`: it splits documents into sentences and (by default) lemmatizes
  and drops stop words. This requires the `full` extra and the `en_core_web_sm`
  model. Pass it explicitly to any constructor.
- **`from_text_files`** is for corpora too large to hold in memory: it reads a
  directory of `*.txt` files and keeps texts on disk. `view_fraction` lets the
  vocabulary be estimated from a random subset of the files (`seed` makes that
  sampling reproducible).
- **`lang`** (default `"en"`) selects which bundled evaluation datasets are used
  later and must match the language of your texts. Bundled: `en` and `de`
  (similarity + analogy), `fr` (analogy), `sv` and `da` (similarity). Every
  bundled file names its source, license and citation in its own `#` preamble.
  Coverage is limited by *licensing*, not by effort — most published similarity
  sets state no license at all, and only sets with an explicit permissive
  license evidenced on the artifact itself are bundled (see ADR 0001, P3). For
  anything else, and for your own domain data, use `data_dir`.
- **`**kwargs`** flow through to `Vocab.filter`, which controls the vocabulary:
  `no_below=0`, `no_above=1`, `keep_n=50000`, `keep_tokens=None` (the same
  arguments as gensim's `filter_extremes`). `keep_n` is the vocabulary-size cap;
  `hyperhyper` is designed for vocabularies up to ~50k.

### The `preproc_func` contract

`preproc_func` is the package's **pluggability hook** — the one place you replace
the tokenizer. It is a `Callable[[list[str]], list[list[str]]]`: given a list of
raw text strings, it returns one token list per string. The built-in tokenizers
(`tokenize_texts_parallel_v2`, `tokenize_texts`, and the spaCy `texts_to_sents`)
are all exactly this shape; a spaCy or Hugging Face tokenizer plugs in the same
way — as a user-supplied callable, not a built-in class hierarchy or registry.

Two hard requirements, both because the function is *stored and shipped*, not just
called:

- **It must be a picklable, top-level function.** The chosen `preproc_func` is
  saved into `corpus.pkl` (by module + qualified name) and sent across process
  pools. A `lambda`, a nested function, or a `functools.partial` breaks both — use
  a module-level `def`. To run a built-in with non-default options, wrap it:

  ```python
  # tokenizers.py  (a real importable module)
  from hyperhyper.preprocessing import tokenize_texts_v2

  def tokenize_domain(texts):
      # keep digits collapsed for this numeral-heavy corpus
      return tokenize_texts_v2(texts, normalize_digits=True)
  ```
  ```python
  import hyperhyper as hy
  from tokenizers import tokenize_domain

  corpus = hy.Corpus.from_texts(docs, preproc_func=tokenize_domain)
  ```

- **The same function is applied to the corpus *and* to the evaluation test
  words.** Evaluation reuses `corpus.preproc_fun` on the words of every test set,
  so your tokenizer and the scored vocabulary can never drift apart. That is also
  why the tokenizer's identity (its qualname) is recorded with each result — see
  [Querying past runs](#querying-past-runs-bunchresults).

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
- **`add_context`** (default `False`): if `True`, use the paper's `w+c`
  representation `U·Σ^eig + V·Σ^eig` (word vectors plus context vectors) instead
  of `U·Σ^eig` alone. A distinct cache entry from the word-only embedding.
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
  accuracy in `[0, 1]`. Pass `objective="mul"` for the 3CosMul objective (Levy &
  Goldberg 2014), which is usually stronger on analogies than the default
  `objective="add"` (3CosAdd).

### Evaluating on a domain corpus

The bundled sets measure **general language**. On a small, domain-specific
corpus — the case this package is built for — they are largely out-of-vocabulary
and therefore measure close to nothing. Check that first, before training:

```python
bunch.dataset_coverage(kind="ws")   # fraction of each dataset's rows in-vocabulary
```

If coverage is near zero, the answer is not a better general-language dataset.
It is a task built from **your** domain's own data. Two are supported, both
scored on gold that is a *membership fact* rather than a judgement — which is
why they can be built without anyone rating anything:

- `bunch.eval_synonym(embd, data_dir=...)` — **synonym multiple choice**. Each
  row is `target answer distractor1 … distractorK`; the row is correct when the
  answer is the target's nearest candidate. Gold comes from a glossary or
  thesaurus entry.
- `bunch.eval_category(embd, data_dir=...)` — **category purity**. Each row is
  `word category`; the score is the fraction of words whose nearest neighbour in
  the dataset shares their category. Gold comes from a taxonomy.

Neither is bundled: a general-language synonym set would recreate the very
problem these solve. Build them from your own data with
`tools/build_domain_tasks/`, which writes `<out>/<lang>/<kind>/<name>.tsv` —
exactly the layout `data_dir` expects.

**Both scores have a chance floor above 0.** Synonym accuracy floors at
`1/(K+1)` (0.25 with three distractors), and each `eval_category` result carries
a `baseline` next to its `score` because purity's floor depends on the category
sizes — a score below that baseline is worse than random. Comparing either
against 0 will make a useless embedding look respectable.

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
  `"prob"` drops tokens at random with that keep probability but keeps the slot,
  so the window spans but does not reach past a dropped token (the paper's *clean*
  variant); `"dirty"` removes dropped tokens entirely so the window closes up and
  reaches further (the variant Levy, Goldberg & Dagan 2015 report). `"deter"` and
  `"prob"` agree in expectation; `"dirty"` genuinely differs — it creates
  co-occurrences across a dropped frequent word that the others cannot.
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
parameters — including the resolved `pair_args__*` columns and a `tokenizer`
column holding the corpus tokenizer's qualname (e.g. `tokenize_texts_parallel_v2`
vs the legacy `tokenize_texts`) — and score columns such as `micro_results`,
`macro_results`, and per-dataset `<name>_score`, `<name>_oov`, `<name>_fullscore`.
The `tokenizer` column is what keeps v1 and v2 numbers attributable and stops them
colliding: two runs that differ only in their tokenizer stay distinct rows.

- **`query`** filters by exact parameter match, e.g. `query={"dim": 500}` or
  `query={"impl": "scipy"}`. Nested parameters use the flattened form, e.g.
  `query={"pair_args": {"window": 2}}`.
- **`order`** is a `column [asc|desc]` expression (validated against a strict
  whitelist).
- **`limit`** caps the number of rows (`None` for no limit).

This makes it easy to sweep parameters and then pull back the best-scoring
configurations without recomputing anything.
