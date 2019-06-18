import logging
from logging import NullHandler

from . import embedding, utils
from .bunch import Bunch
from .pair_counts import count_pairs
from .dataset import Corpus, Vocab

logging.getLogger(__name__).addHandler(NullHandler())
