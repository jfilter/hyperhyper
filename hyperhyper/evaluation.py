"""
Evaluate the performance of embeddings with word simularities and word analogies.

Can't use the evaluation methods in gensim because the keyed vector structure does not work for PPMI.
So we have to caculate the metrics ourselves.
"""

from pathlib import Path

import numpy as np
from scipy.stats.stats import spearmanr

from . import evaluation_datasets

try:
    from importlib.resources import path
except ImportError:
    # backport for Python <3.7
    from importlib_resources import path


def read_test_data(lang, type):
    """
    read test data that is stored within the module
    """
    with path(evaluation_datasets, lang) as eval_dir:
        for file in eval_dir.glob(f"{type}/*.txt"):
            yield file


def to_item(li):
    """
    squeeze
    """
    if isinstance(li, list):
        if len(li) == 0:
            return None
        if len(li) == 1:
            return li[0]
        return to_item(li[0])
    return li


def setup_test_tokens(p, keep_len):
    """
    Read in traning data from files and discard comments (etc.)
    """
    lines = Path(p).read_text().split("\n")
    lines = [l.split() for l in lines]
    lines = [l for l in lines if len(l) == keep_len]
    return zip(*lines)


def eval_similarity(vectors, token2id, preproc_fun, lang="en"):
    """
    evaluate word similarity on several test datasets
    """
    line_counts, spear_results, full_results = [], [], []

    for data in read_test_data(lang, "ws"):
        results = []

        token1, token2, sims = setup_test_tokens(data, 3)
        # preprocess tokens 'in batch'
        token1, token2 = preproc_fun(token1), preproc_fun(token2)
        lines = list(zip(token1, token2, sims))
        for x, y, sim in lines:
            x, y = to_item(x), to_item(y)

            # not sure it the lines below are needed
            # if x is None or y is None:
            #     continue

            # skip over OOV
            if x in token2id and y in token2id:
                results.append((vectors.similarity(token2id[x], token2id[y]), sim))

        if len(results) == 0:
            print("not enough results for this dataset: ", data.name)
            continue

        actual, expected = zip(*results)
        spear_res = spearmanr(actual, expected)[0]
        spear_results.append(spear_res)
        line_counts.append(len(results))
        oov = (len(lines) - len(results)) / len(lines)

        full_results.append(
            {
                "name": f"{lang}_{data.stem}",
                "score": spear_res,
                "oov": oov,
                "fullscore": spear_res * (1 - oov),  # consider the portion of OOV
            }
        )

    micro_avg = sum([x * y for x, y in zip(line_counts, spear_results)]) / sum(
        line_counts
    )
    macro_avg = sum(spear_results) / len(spear_results)
    return {"micro": micro_avg, "macro": macro_avg, "results": full_results}


# TODO:

# analogies
def eval_analogies(vectors, token2id, preproc_fun, lang="en"):
    line_counts, full_results = [], []

    for data in read_test_data(lang, "analogy"):
        results = []

        line_tokens = setup_test_tokens(data, 4)
        line_tokens = [preproc_fun(t) for t in line_tokens]
        lines = list(zip(*line_tokens))
        for tokens in lines:
            tokens = [to_item(x) for x in tokens]
            # skip over OOV
            if not all([x in token2id for x in tokens]):
                continue

            tokens = [token2id[x] for x in tokens]
            a, a_, b, b_ = tokens
            guesses = vectors.most_similar_vectors([a, b], [a_])
            result = 1 if b_ in guesses else 0
            results.append(result)

        if len(results) == 0:
            print("not enough results for this dataset: ", data.name)
            continue

        sum_results = sum(results)
        line_counts.append(len(results))
        oov = (len(lines) - len(results)) / len(lines)

        full_results.append(
            {
                "name": f"{lang}_{data.stem}",
                "score": sum_results,
                "oov": oov,
                "fullscore": sum_results * (1 - oov),  # consider the portion of OOV
            }
        )

    scores = [x['score'] for x in full_results]
    micro_avg = sum([x * y for x, y in zip(line_counts, scores)]) / sum(
        line_counts
    )
    macro_avg = sum(scores) / len(scores)
    return {"micro": micro_avg, "macro": macro_avg, "results": full_results}
