import torch
from transformers import GPT2Config, GPT2LMHeadModel

from pcfg import PCFG, PCFGDataset

def train_model(
    model: GPT2LMHeadModel,
    grammar: PCFG,
    train_data: PCFGDataset,
    test_data: PCFGDataset
):
    # training loop
    for i, batch in enumerate(train_data):

        batch = grammar.batch_tokenize(batch, return_tensors='pt')

