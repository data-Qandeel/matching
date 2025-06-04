import torch
import torch.nn as nn

class DeezyMatchEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        rnn_hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.5,
        padding_idx: int = 0,
        ff_hidden_dim: int = 2048,
        final_embedding_dim: int = 1024
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.padding_idx = padding_idx
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.rnn_output_dim = rnn_hidden_dim * 2
        self.ffn = nn.Sequential(
            nn.Linear(self.rnn_output_dim * 3, ff_hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, final_embedding_dim),
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != self.padding_idx).float()
        emb = self.embedding(x)

        output, hidden = self.rnn(emb)
        output_masked = output * mask.unsqueeze(-1)

        # --- Apply Pooling Strategies ---
        mean_pooling = torch.sum(output_masked, dim=1) / torch.sum(mask, dim=1, keepdim=True)
        max_pooling, _ = torch.max(output_masked, dim=1)
        
        hidden = hidden.view(self.rnn.num_layers, 2, -1, self.rnn.hidden_size)
        last_forward = hidden[-1, 0]
        last_backward = hidden[-1, 1]
        last_hidden = torch.cat((last_forward, last_backward), dim=-1)

        # --- Concatenate Pooling Strategies ---
        pooled_features = torch.cat((mean_pooling, max_pooling, last_hidden), dim=-1)

        return self.ffn(pooled_features)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_once(x)
