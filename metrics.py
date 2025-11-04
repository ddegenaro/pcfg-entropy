from typing import Union

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel

from lstm import LSTM
from utils import SequenceDataset, SequenceDataLoader

def get_sequence_probabilities(logits, input_ids, attention_mask): # GENERATED WITH CLAUDE
    """
    Compute the probability the LM assigned to each sequence.
    
    Args:
        logits: (B, T, vocab_size) - raw logits from LM head
        input_ids: (B, T) - token indices
        attention_mask: (B, T) - 1 for real tokens, 0 for padding
    
    Returns:
        seq_probs: (B,) - probability assigned to each sequence
    """
    # Convert logits to probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, vocab_size)
    
    # Gather log probabilities of the true tokens
    # Expand input_ids to match log_probs shape for gathering
    token_log_probs = torch.gather(
        log_probs, 
        dim=-1, 
        index=input_ids.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)
    
    # Mask out padding tokens
    token_log_probs = token_log_probs * attention_mask
    
    # Sum log probabilities for each sequence (equivalent to multiplying probabilities)
    seq_log_probs = token_log_probs.sum(dim=1)  # (B,)
    
    # Convert back to probabilities if desired
    seq_probs = torch.exp(seq_log_probs)  # (B,)
    
    return seq_probs, seq_log_probs



def both_metrics(
    val_data: SequenceDataset,
    model: Union[LSTM, GPT2LMHeadModel],
    p_true: list[float]
):

    """
    Sequence-wise spearman rho over whole test set - get true probs vs LM predicted probs.
    """

    p_model = []

    if type(model) == LSTM:
        def tokenize(batch):
            return val_data.grammar.batch_tokenize(
                batch,
                return_tensors='pt'
            )
    elif type(model) == GPT2LMHeadModel:
        def tokenize(batch):
            return val_data.grammar.batch_tokenize(
                batch,
                return_tensors='pt',
                truncate_length=model.n_positions
            )
    else:
        raise TypeError(f'model should not be of type {type(model)}')
    
    total_loss = 0.
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(
            SequenceDataLoader(val_data, batch_size=32)
        ):
            inputs = tokenize(batch)
            outputs = model(**inputs)
            logits = outputs.logits
            loss = outputs.loss.item()
            tokens = inputs['attention_mask'].sum().item()
            total_loss += loss * tokens # reconstructed total loss over these tokens
            total_tokens += tokens
            p_model.extend(
                get_sequence_probabilities(logits, inputs['input_ids'], inputs['attention_mask'])[0].tolist()
            )
    
    model.train()
    return spearmanr(p_true, p_model), (total_loss / total_tokens)