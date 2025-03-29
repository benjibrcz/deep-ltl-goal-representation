from torch import nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, d_model: int, n_head: int, num_layers: int, dim_feedforward: int, dropout: float=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        self.positional_encoding = PositionalEncoding(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0, std=0.02)

    def forward(self, x: torch.tensor, lens: torch.tensor) -> torch.tensor:
        # Input shape: B,L,D; lens shape: B
        max_len = x.shape[1]
        src_padding_mask = torch.arange(max_len)[None, :] >= lens[:, None]
        # Amend padding mask for CLS token
        src_padding_mask = torch.cat((torch.zeros((x.shape[0], 1), dtype=torch.bool), src_padding_mask), dim=1)
        src_padding_mask = src_padding_mask.to(x.device)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.positional_encoding(x)
        res = self.encoder(x, src_key_padding_mask=src_padding_mask)
        return res[:, 0, :]