import torch
from torch import nn

class LSTM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        eos_token_id: int,
        n_embd: int,
        n_layer: int
    ):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.n_embd = n_embd
        self.n_layer = n_layer

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.n_embd,
            padding_idx=self.pad_token_id
        )

        self.lstm = nn.LSTM(
            
        )

    def forward(self):
        pass