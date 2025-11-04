import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMOutput:

    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits

class LSTM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        n_embd: int,
        n_hidden: int,
        n_layer: int
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.n_embd = n_embd
        self.n_layer = n_layer

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.n_embd,
            padding_idx=self.pad_token_id
        )

        self.lstm = nn.LSTM(
            input_size=n_embd,
            hidden_size=n_hidden,
            num_layers=n_layer,
            batch_first=True,
            dropout=0.1
        )

    def forward(self, input_ids, attention_mask):
        # Embed first (before packing)
        embedded = self.embedding(input_ids)

        # Then pack the embedded sequences
        lengths = attention_mask.sum(1)
        packed_embedded = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through LSTM
        lstm_out, (h, c) = self.lstm(packed_embedded)

        # Unpack LSTM output
        unpacked_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Project to vocabulary size
        logits = self.lm_head(unpacked_out)  # Shape: (B, T, vocab_size)
        
        # Compute cross-entropy loss only on non-padded tokens
        # Flatten batch and time dimensions
        loss = None
        if input_ids is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            
            # Flatten for loss computation
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_mask = shift_mask.view(-1)
            
            # Compute cross-entropy, ignoring padded positions
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits, shift_labels)
            loss = (loss * shift_mask).sum() / shift_mask.sum()
        
        return LSTMOutput(loss=loss, logits=logits)