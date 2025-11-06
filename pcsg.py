from time import time
from typing import Union

import torch
from torch import nn

from utils import SequenceDataset, Grammar, Node, Tree, Sequence

class PCSG(Grammar):
    pass

class PCSGDataset(SequenceDataset):
    pass