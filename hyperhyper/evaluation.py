from pathlib import Path

import numpy as np
from scipy.stats.stats import spearmanr

from . import evaluation_datasets  # the package containing the file

try:
    # import importlib.resources as pkg_resources
    from importlib.resources import path

except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    # import importlib_resources as pkg_resources
    from importlib_resources import path


def read_test_data(lang, type):
    with path(evaluation_datasets, lang) as pad:
        for x in pad.glob(f"{type}/*.txt"):
            yield x


# get item to list
def to_item(li):
    if isinstance(li, list):
        if len(li) == 0:
            return None
        if len(li) == 1:
            return li[0]
        return to_item(li[0])
    return li

# cant use the evaluation stuff in gensim because the keyed vector strucutre does not ork for PPMI
# non keyed vectors (used for PPMI)

# word similarity
def eval_similarity(vectors, token2id, preproc_fun, lang="en"):
    line_counts = []
    spear_results = []
    full_results = []

    for data in read_test_data(lang, 'ws'):
        results = []
        lines = Path(data).read_text().split("\n")
        lines = [l.split() for l in lines]
        lines = [l for l in lines if len(l) == 3]
        for x, y, sim in lines:
            x = to_item(preproc_fun(x))
            y = to_item(preproc_fun(y))

            # skip over OOV
            if x is None or y is None:
                continue

            if x in token2id and y in token2id:
                results.append((vectors.similarity(token2id[x], token2id[y]), sim))
        if len(results) == 0:
            print("not enough results for this dataset: ", data.name)
            continue
        actual, expected = zip(*results)
        spear_res = spearmanr(actual, expected)[0]
        spear_results.append(spear_res)
        line_counts.append(len(results))
        oov = len(lines) - len(results)
        full_results.append(
            {"name": data.stem, "score": spear_res, "oov": oov / len(lines)}
        )

    micro_avg = sum([x * y for x, y in zip(line_counts, spear_results)]) / sum(
        line_counts
    )
    macro_avg = sum(spear_results) / len(spear_results)
    return {"micro": micro_avg, "macro": macro_avg, "results": full_results}

# analogies
def eval_analogies(vectors, token2id, preproc_fun, lang="en"):
    sims = prepare_similarities(vectors, token2id)

    for data in read_test_data(lang, 'ws'):
        correct_add = 0.0
        correct_mul = 0.0
        lines = Path(data).read_text().split("\n")
        lines = [l.split() for l in lines]
        lines = [l for l in lines if len(l) == 3]
        
    for a, a_, b, b_ in data:
        b_add, b_mul = guess(representation, sims, xi, a, a_, b)
        if b_add == b_:
            correct_add += 1
        if b_mul == b_:
            correct_mul += 1
    return correct_add / len(data), correct_mul / len(data)


def prepare_similarities(representation, token2id):
    vocab_representation = representation.m[
        [representation.wi[w] if w in representation.wi else 0 for w in vocab]
    ]
    sims = vocab_representation.dot(representation.m.T)

    dummy = None
    for w in vocab:
        if w not in representation.wi:
            dummy = representation.represent(w)
            break
    if dummy is not None:
        for i, w in enumerate(vocab):
            if w not in representation.wi:
                vocab_representation[i] = dummy

    if type(sims) is not np.ndarray:
        sims = np.array(sims.todense())
    else:
        sims = (sims + 1) / 2
    return sims


def guess(representation, sims, xi, a, a_, b):
    sa = sims[xi[a]]
    sa_ = sims[xi[a_]]
    sb = sims[xi[b]]

    add_sim = -sa + sa_ + sb
    if a in representation.wi:
        add_sim[representation.wi[a]] = 0
    if a_ in representation.wi:
        add_sim[representation.wi[a_]] = 0
    if b in representation.wi:
        add_sim[representation.wi[b]] = 0
    b_add = representation.iw[np.nanargmax(add_sim)]

    mul_sim = sa_ * sb * np.reciprocal(sa + 0.01)
    if a in representation.wi:
        mul_sim[representation.wi[a]] = 0
    if a_ in representation.wi:
        mul_sim[representation.wi[a_]] = 0
    if b in representation.wi:
        mul_sim[representation.wi[b]] = 0
    b_mul = representation.iw[np.nanargmax(mul_sim)]

    return b_add, b_mul
