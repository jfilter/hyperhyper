import logging
from pathlib import Path

from . import pair_counts, pmi
from .dataset import Corpus
from .utils import load_matrix, save_matrix

logger = logging.getLogger(__name__)


class Bunch:
    def __init__(self, path, corpus=None):
        self.path = Path(path)
        if corpus is None:
            self.corpus = Corpus.load(str(self.path / "corpus.pkl"))
        else:
            self.path.mkdir(parents=True, exist_ok=True)
            self.corpus = corpus
            self.corpus.save(str(self.path / "corpus.pkl"))

    def dict_to_path(self, folder, dict):
        filename = "_".join([f"{k}_{v}" for k, v in dict.items()])
        if len(filename) == 0:
            filename = "default"

        filename += ".pkl"
        full_path = self.path / folder / filename
        return full_path

    def pair_counts(self, **kwargs):
        pair_path = self.dict_to_path("pair_counts", kwargs)
        if pair_path.is_file():
            logger.info("retrieved already saved pair count")
            return load_matrix(pair_path)

        logger.info("create new pair counts")
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        count_matrix = pair_counts.count_pairs(self.corpus, **kwargs)
        save_matrix(pair_path, count_matrix)
        return count_matrix

    def pmi(self, cds=0.75, pair_args={}):
        pmi_path = self.dict_to_path("pmi", {"cds": cds, **pair_args})
        if pmi_path.is_file():
            logger.info("retrieved already saved pmi")
            return load_matrix(pmi_path)
        logger.info("create new pmi")
        counts = self.pair_counts(**pair_args)
        pmi_matrix = pmi.calc_pmi(counts, cds)
        pmi_path.parent.mkdir(parents=True, exist_ok=True)
        save_matrix(pmi_path, pmi_matrix)
        return pmi_matrix

    # def svd(self, dimension=500, neg=5, cds=0.75, pair_args={}):
    #     p = pmi(cds, pair_args)
