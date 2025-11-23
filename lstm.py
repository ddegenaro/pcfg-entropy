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
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.n_embd,
            padding_idx=self.pad_token_id
        )

        self.lstm = nn.LSTM(
            input_size=self.n_embd,
            hidden_size=self.n_hidden,
            num_layers=self.n_layer,
            batch_first=True,
            dropout=0.1
        )

        self.lm_head = nn.Linear(self.n_hidden, self.vocab_size)

    def forward(self, input_ids, attention_mask, labels):

        # Embed first (before packing)
        embedded = self.embedding(input_ids)

        # Then pack the embedded sequences
        lengths = attention_mask.sum(1)

        packed_embedded = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through LSTM
        lstm_out, _ = self.lstm(packed_embedded)

        # Unpack LSTM output
        unpacked_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Project to vocabulary size
        logits = self.lm_head(unpacked_out)  # Shape: (B, T, vocab_size)
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute cross-entropy, ignoring padded positions
        loss = self.loss_fn(shift_logits.permute(0, 2, 1), shift_labels)
        
        return LSTMOutput(loss=loss, logits=logits)