from typing import Union
from math import isnan

from tqdm import tqdm
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel

from lstm import LSTM
from utils import SequenceDataLoader

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
    # seq_probs = torch.exp(seq_log_probs)  # (B,)
    
    return seq_log_probs

# def both_metrics(
#     val_data_loader: SequenceDataLoader,
#     model: Union[LSTM, GPT2LMHeadModel],
#     p_true_by_len: dict[int, list[float]],
#     device: str,
#     verbose: bool
# ):

#     """
#     Sequence-wise spearman rho over whole test set - get true probs vs LM predicted probs.
#     """

#     p_model = []
    
#     total_loss = 0.
#     total_tokens = 0

#     if verbose:
#         iterable = tqdm(val_data_loader, total=len(val_data_loader))
#     else:
#         iterable = val_data_loader
    
#     model.eval()
#     with torch.no_grad():
#         for batch in iterable:

#             batch['labels'] = batch['input_ids']

#             for key in batch:
#                 batch[key] = batch[key].to(device)

#             outputs = model(**batch)
#             logits = outputs.logits
#             loss = outputs.loss.item()
#             tokens = batch['attention_mask'].sum().item()
#             total_loss += loss * tokens # reconstructed total loss over these tokens
#             total_tokens += tokens
#             p_model.extend(
#                 get_sequence_probabilities(
#                     logits,
#                     batch['input_ids'],
#                     batch['attention_mask']
#                 )[0].tolist()
#             )
    
#     model.train()
#     return spearmanr(p_true, p_model), (total_loss / total_tokens)


def both_metrics(
    val_data_loader: SequenceDataLoader,
    model: Union[LSTM, GPT2LMHeadModel],
    log_p_true_by_len: dict[int, list[float]],
    device: str,
    verbose: bool
):

    """
    Sequence-wise spearman rho over whole test set, grouped by sequence length.
    Returns mean spearman rho across all lengths, plus dictionaries with per-length results.
    """

    log_p_model_by_len = {}
    
    total_loss = 0.
    total_tokens = 0

    if verbose:
        iterable = tqdm(val_data_loader, total=len(val_data_loader))
    else:
        iterable = val_data_loader
    
    model.eval()
    with torch.no_grad():
        for batch in iterable:

            batch['labels'] = batch['input_ids']

            for key in batch:
                batch[key] = batch[key].to(device)

            outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss.item()
            tokens = batch['attention_mask'].sum().item()
            total_loss += loss * tokens # reconstructed total loss over these tokens
            total_tokens += tokens
            
            seq_log_probs = get_sequence_probabilities(
                logits,
                batch['input_ids'],
                batch['attention_mask']
            ).tolist()
            
            # Get sequence lengths (number of non-padding tokens)
            seq_lengths = batch['attention_mask'].sum(dim=1).tolist()
            
            # Group probabilities by sequence length
            for seq_len, prob in zip(seq_lengths, seq_log_probs):
                if seq_len not in log_p_model_by_len:
                    log_p_model_by_len[seq_len] = []
                log_p_model_by_len[seq_len].append(prob)
    
    model.train()
    
    # Compute spearman correlation for each sequence length
    spearman_by_len = {}
    pvalue_by_len = {}
    spearman_values = []
    seq_lens = []
    
    for seq_len in sorted(log_p_true_by_len.keys()):
        if seq_len in log_p_model_by_len:
            log_p_true = log_p_true_by_len[seq_len]
            log_p_model = log_p_model_by_len[seq_len]
            
            if len(log_p_true) > 2 and len(log_p_model) > 2:  # Need at least 2 samples for correlation
                rho, pval = spearmanr(log_p_true, log_p_model, nan_policy='omit')
                if not isnan(rho):
                    spearman_by_len[seq_len] = rho
                    if not isnan(pval):
                        pvalue_by_len[seq_len] = pval
                    else:
                        pvalue_by_len[seq_len] = 0.
                    spearman_values.append(rho)
                    seq_lens.append(seq_len)
    
    mean_spearman = sum(spearman_values) / len(spearman_values) if spearman_values else 0.0
    
    return (
        seq_lens,
        mean_spearman,
        spearman_by_len,
        pvalue_by_len,
        (total_loss / total_tokens)
    )