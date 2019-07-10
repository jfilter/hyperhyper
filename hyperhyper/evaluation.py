from pathlib import Path

from scipy.stats.stats import spearmanr

from . import evaluation_datasets  # the package containing the file

try:
    # import importlib.resources as pkg_resources
    from importlib.resources import path

except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    # import importlib_resources as pkg_resources
    from importlib_resources import path


def wc_l(path):
    return len(open(path).readlines())


def read_vectors(lang):
    with path(evaluation_datasets, lang) as pad:
        for x in pad.glob("ws/*.txt"):
            yield x


# keyed vectors
def eval_similarity(vectors, lang="en", **kwargs):
    line_counts = []
    spear_results = []

    for x in read_vectors(lang):
        print(x)
        try:
            pear, spear, oov_ratio = vectors.evaluate_word_pairs(str(x), **kwargs)
        except:
            continue
        print(x.name, pear, spear, oov_ratio)
        line_counts.append(wc_l(x) * oov_ratio)
        spear_results.append(spear)
    print(line_counts, spear_results)

    if len(line_counts) == 0:
        return 0

    score = sum([x * y for x, y in zip(line_counts, spear_results)]) / sum(line_counts)
    print(score)
    return score


# non keyed vectors (used for PPMI)
def embedding_eval_sim(vectors, token2id, preproc_fun, lang="en"):
    line_counts = []
    spear_results = []
    full_results = []

    for data in read_vectors(lang):
        results = []
        lines = Path(data).read_text().split("\n")
        lines = [l.split() for l in lines]
        lines = [l for l in lines if len(l) == 3]
        for x, y, sim in lines:
            x = preproc_fun(x)
            y = preproc_fun(y)

            if isinstance(x, list):
                if len(x) == 0:
                    continue
                x = x[0]

            if isinstance(y, list):
                if len(y) == 0:
                    continue
                y = y[0]

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
