"""
Timing harness for `hyperhyper.count_pairs`.

    python bench/bench_pair_counts.py [n_sentences] [sentence_length] [repeats]

Builds a synthetic Zipfian corpus with `numpy.random.default_rng(SEED)`, times
the pair counting over a handful of representative configurations, and prints a
table. Defaults are sized to finish in well under a minute.

WHAT IS TIMED, AND WHY IT IS NOT `count_pairs`
==============================================
The main table times the counting *core*: every chunk run through
`CountPairsClosure` serially, in this process. It does not go through
`count_pairs`.

That is deliberate, and it was measured rather than assumed. `count_pairs`
starts a `ProcessPoolExecutor` per call, which costs a flat ~3.5s on
macOS/spawn and then spreads the actual counting across every core. At 240k
tokens every configuration timed at 3.2s -- indistinguishable from an empty
call. At 2.4M tokens (10x the work) they moved to 3.7-4.9s, so better than 80%
of the measurement was still pool startup. Benchmarking through the pool would
have reported "no change" for a rewrite that made the inner loop twice as fast.
Timing the core single-threaded makes the number proportional to the thing
being optimized. One end-to-end row is printed underneath for scale.

The corpus is regenerated from the seed on every run, so two invocations on two
different revisions see byte-identical input. Each row also carries `nnz`, the
matrix `sum`, and a `checksum` (the sum of the nonzero data as float64): if a
supposedly faster revision changes any of those on a deterministic
configuration, it changed the answer, and `tests/test_pair_counts_equivalence.py`
is where to go find out how.


COMPARING TWO REVISIONS
=======================

The script only measures the checkout it is run from, so compare by running it
twice and diffing. `bench/` is untracked by the revision under test, so keep a
copy outside the worktree:

    cp bench/bench_pair_counts.py /tmp/bench.py

    # baseline
    git stash               # or: git switch <baseline-rev>
    python /tmp/bench.py 60000 12 3 > /tmp/before.txt

    # the optimization
    git stash pop           # or: git switch <optimized-rev>
    python /tmp/bench.py 60000 12 3 > /tmp/after.txt

    diff -y /tmp/before.txt /tmp/after.txt

Read the diff in this order:

  1. `checksum` / `sum` / `nnz` on the three deterministic rows (`w2 dyn=deter`,
     `w5 dyn=decay`, `w10 dyn=deter`) MUST be unchanged. These columns are the
     cheap smoke test; the real gate is
     `pytest tests/test_pair_counts_equivalence.py -m slow`, which compares
     whole matrices against the frozen snapshot in `bench/reference.py`.
  2. The two `prob` rows will differ in `checksum` even when the rewrite is
     correct -- vectorizing the RNG changes the draw order. Only their timings
     are comparable.
  3. `best` (the fastest of `repeats` runs) and `tok/s` are the numbers to
     quote. `mean` is there to show how noisy the machine is; if `mean - best`
     is a large fraction of `best`, close things and run it again.
  4. The end-to-end line will improve far less than the core, because most of
     it is pool startup. That is expected and is not evidence the optimization
     failed.

One seam to watch: `make_closure` below reaches into `pair_counts` to build a
`CountPairsClosure` the way `count_pairs` does. If the rewrite changes how a
worker is configured, that function is the thing to update.
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hyperhyper
from hyperhyper import pair_counts
from hyperhyper.preprocessing import tokenize_texts

SEED = 20260721
DEFAULT_SENTENCES = 60000
DEFAULT_SENTENCE_LENGTH = 12
DEFAULT_REPEATS = 2
VOCAB_SIZE = 400
CHUNK_SIZE = 2000
ZIPF_EXPONENT = 1.07

# (label, kwargs). Deliberately short -- the core timing is single-threaded, so
# every entry costs real seconds.
#
# `subsample="deter"` is absent on purpose: it is a post-merge rescaling of the
# finished matrix, not part of the counting, so it would time identically to
# `subsample=None` and just pad the table. `subsample="prob"` IS part of the
# counting (it drops tokens inside the loop) and is therefore included.
CONFIGS = [
    ("w2 dyn=deter", {"window": 2, "dynamic_window": "deter", "subsample": None}),
    ("w5 dyn=decay", {"window": 5, "dynamic_window": "decay", "subsample": None}),
    ("w10 dyn=deter", {"window": 10, "dynamic_window": "deter", "subsample": None}),
    ("w5 dyn=prob", {"window": 5, "dynamic_window": "prob", "subsample": None}),
    ("w5 sub=prob", {"window": 5, "dynamic_window": None, "subsample": "prob"}),
]

# large enough that subsampling actually drops tokens instead of erasing the
# corpus; see the same constant in tests/test_pair_counts_equivalence.py
SUBSAMPLE_FACTOR = 6e-3


def build_corpus(directory, n_sentences, sentence_length):
    """
    A Zipfian synthetic corpus, reproducible from `SEED` alone.

    Zipf rather than uniform because the pair matrix's sparsity pattern -- and
    therefore the cost of building it -- depends on how skewed the token
    distribution is, and real corpora are skewed.
    """
    rng = np.random.default_rng(SEED)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    # gensim's tokenizer keeps alphabetic tokens only, so no digits in the words
    words = np.array(
        [
            alphabet[i // (26 * 26) % 26] + alphabet[i // 26 % 26] + alphabet[i % 26]
            for i in range(VOCAB_SIZE)
        ]
    )
    weights = 1.0 / np.arange(1, VOCAB_SIZE + 1) ** ZIPF_EXPONENT
    weights /= weights.sum()

    drawn = rng.choice(VOCAB_SIZE, size=(n_sentences, sentence_length), p=weights)
    sents = [" ".join(words[row]) for row in drawn]

    corpus = hyperhyper.Corpus.from_texts(sents, preproc_func=tokenize_texts)
    corpus.texts_to_file(Path(directory) / "texts", CHUNK_SIZE)
    return corpus


def make_closure(corpus, kwargs):
    """
    Build the same `CountPairsClosure` that `count_pairs` hands to its workers.

    It calls `pair_counts.make_count_closure`, the *same* translation
    `count_pairs` uses, rather than reproducing it. This function used to keep a
    hand-written copy, and the copy drifted: when the `dirty` subsampling variant
    landed the copy was not updated, so every benchmark run died with
    `AttributeError: ... has no attribute 'subsampler_dirty'`. A benchmark that
    cannot run is worse than no benchmark, because it looks like a safety net.
    """
    total_tokens = sum(corpus.counts.values())
    subsample_value = SUBSAMPLE_FACTOR * total_tokens
    subsample = kwargs.get("subsample")

    subsampler_prob = None
    if subsample in ("prob", "dirty"):
        subsampler_prob = pair_counts.subsample_keep_probabilities(
            corpus.counts, subsample_value
        )

    return pair_counts.make_count_closure(
        corpus,
        window=kwargs.get("window", 2),
        dynamic_window=kwargs.get("dynamic_window"),
        decay_rate=0.25,
        delete_oov=True,
        subsample=subsample,
        subsampler_prob=subsampler_prob,
        seed=1312,
    )


def time_core(corpus, kwargs, repeats):
    """
    Time the counting itself: every chunk, serially, in this process.

    This is the primary measurement. Timing `count_pairs` end to end mostly
    measures `ProcessPoolExecutor` -- the pool costs a flat few seconds to
    start and then splits the work across every core, so at any corpus small
    enough to benchmark comfortably the loop under optimization disappears into
    the noise. Running the same work single-threaded puts it back in view and
    makes the number proportional to what the rewrite changes.
    """
    closure = make_closure(corpus, kwargs)
    timings = []
    matrix = None
    for _ in range(repeats):
        start = time.perf_counter()
        matrix = None
        for path in sorted(corpus.texts):
            m = closure(path)
            matrix = m if matrix is None else matrix + m
        timings.append(time.perf_counter() - start)
    return timings, matrix


def time_end_to_end(corpus, kwargs, repeats):
    """
    The full `count_pairs`, pool and all -- the number a user actually feels.
    """
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        hyperhyper.count_pairs(corpus, subsample_factor=SUBSAMPLE_FACTOR, **kwargs)
        timings.append(time.perf_counter() - start)
    return timings


def main(argv):
    n_sentences = int(argv[0]) if len(argv) > 0 else DEFAULT_SENTENCES
    sentence_length = int(argv[1]) if len(argv) > 1 else DEFAULT_SENTENCE_LENGTH
    repeats = int(argv[2]) if len(argv) > 2 else DEFAULT_REPEATS

    with tempfile.TemporaryDirectory() as tmp:
        build_start = time.perf_counter()
        corpus = build_corpus(tmp, n_sentences, sentence_length)
        build_seconds = time.perf_counter() - build_start

        print()
        print("hyperhyper count_pairs benchmark")
        print(f"  seed             {SEED}")
        print(f"  sentences        {n_sentences}")
        print(f"  sentence length  {sentence_length}")
        print(f"  tokens           {sum(corpus.counts.values())}")
        print(f"  vocab            {corpus.vocab.size}")
        print(f"  chunks           {len(corpus.texts)}")
        print(f"  repeats          {repeats}")
        print(f"  corpus build     {build_seconds:.2f}s")
        print()
        print("counting core, single process (the number the rewrite should move)")
        print()

        header = (
            f"{'config':<24} {'best':>8} {'mean':>8} {'tok/s':>12} {'nnz':>10} "
            f"{'sum':>16} {'checksum':>20}"
        )
        print(header)
        print("-" * len(header))

        total_tokens = sum(corpus.counts.values())
        for label, kwargs in CONFIGS:
            timings, matrix = time_core(corpus, kwargs, repeats)
            checksum = float(np.float64(matrix.data).sum())
            print(
                f"{label:<24} {min(timings):>8.3f} {np.mean(timings):>8.3f} "
                f"{total_tokens / min(timings):>12,.0f} {matrix.nnz:>10} "
                f"{float(matrix.sum()):>16.6g} {checksum:>20.10g}",
                flush=True,
            )

        # one end-to-end row for scale: how much of a real call is the pool
        e2e_label, e2e_kwargs = CONFIGS[0]
        e2e = time_end_to_end(corpus, e2e_kwargs, repeats)
        print()
        print(
            f"end to end (count_pairs, process pool, {e2e_label}): "
            f"best {min(e2e):.3f}s over {len(corpus.texts)} chunks"
        )
        print()
        print("checksum/sum/nnz must not move on the deterministic rows; the two")
        print("'prob' rows are expected to move once the RNG is vectorized.")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
