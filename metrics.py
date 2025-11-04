from typing import Union

import torch
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel

from lm import val_epoch
from lstm import LSTM
from utils import SequenceDataset

# sequence-wise spearman rho over whole test set - get true probs vs LM predicted probs
def spearman_rho(
    test_set: SequenceDataset,
    model: Union[LSTM, GPT2LMHeadModel]
):
    p_true = [test_set.grammar.p_seq(seq).item() for seq in test_set]
    p_model = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(
            torch.utils.data.DataLoader(test_set, batch_size=32)
        ):
            inputs = test_set.grammar.batch_tokenize(
                batch,
                return_tensors='pt',
                truncate_length=model.n_positions
            )
    
    model.train()
    return spearmanr(p_true, p_model)

# token-wise averaged cross-entropy over whole test set
def test_set_cross_entropy(
    test_set: SequenceDataset,
    model: Union[LSTM, GPT2LMHeadModel]
):
    pass