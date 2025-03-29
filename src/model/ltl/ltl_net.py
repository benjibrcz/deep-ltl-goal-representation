import torch
from torch import nn

from model.ltl.transformer_model import TransformerModel
from preprocessing import BatchedReachAvoidSequences


class LTLNet(nn.Module):
    def __init__(
            self,
            embedding: nn.Module,
            set_net_config: 'SetNetConfig',
            num_rnn_layers: int,
            use_transformer: bool = False
    ):
        super().__init__()
        self.embedding = embedding
        self.set_network = set_net_config.build(embedding.embedding_dim)
        embedding_dim = self.set_network.embedding_size
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = TransformerModel(
                d_model=2 * embedding_dim,
                n_head=4,
                num_layers=num_rnn_layers,
                dim_feedforward=512,
                dropout=0.1
            )
        else:
            self.rnn = nn.GRU(input_size=2 * embedding_dim, hidden_size=2 * embedding_dim, num_layers=num_rnn_layers,
                              batch_first=True)
        self.embedding_dim = 2 * embedding_dim

    def forward(self, batched_seqs: tuple[tuple[torch.tensor, torch.tensor], tuple[torch.tensor, torch.tensor]]
                                    | BatchedReachAvoidSequences) -> torch.tensor:
        if isinstance(batched_seqs, BatchedReachAvoidSequences):
            (reach_lens, reach_data), (avoid_lens, avoid_data) = batched_seqs.all()
        else:
            (reach_lens, reach_data), (avoid_lens, avoid_data) = batched_seqs
        assert (reach_lens == avoid_lens).all()
        reach = self.embedding(reach_data)
        reach = self.set_network(reach)
        avoid = self.embedding(avoid_data)
        avoid = self.set_network(avoid)
        x = torch.cat([reach, avoid], dim=-1)
        if self.use_transformer:
            return self.transformer(x, reach_lens)
        else:
            seq = nn.utils.rnn.pack_padded_sequence(x, reach_lens, batch_first=True, enforce_sorted=False)
            _, h = self.rnn(seq)
            return h[-1, ...]
