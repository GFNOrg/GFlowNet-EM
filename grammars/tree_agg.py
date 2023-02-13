import torch.nn as nn
import torch
import math
from my_transformer import TransformerEncoderLayer, TransformerEncoder, LayerNorm

from gflownet_parser import create_position_ids


class transformer_binary_agg(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        norm_first,
        nlayers,
        batch_first=True,
        activation="relu",
    ):
        nn.Module.__init__(self)
        self.embedding_pos = nn.Embedding(3, d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            activation=activation,
            norm_first=norm_first,
        )
        encoder_norm = LayerNorm(d_model, eps=1e-5)
        self.final_norm = LayerNorm(d_model, eps=1e-5)
        self.model_encoder = TransformerEncoder(encoder_layer, nlayers, encoder_norm)

    def forward(self, root, left, right):
        x = torch.stack([root, left, right], dim=0)
        x += self.embedding_pos.weight
        return self.final_norm(self.model_encoder(x.unsqueeze(0))).squeeze(0)[0]


class transformer_unary_agg(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        norm_first,
        nlayers,
        batch_first=True,
        activation="relu",
    ):
        nn.Module.__init__(self)
        self.embedding_pos = nn.Embedding(2, d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            activation=activation,
            norm_first=norm_first,
        )
        encoder_norm = LayerNorm(d_model, eps=1e-5)
        self.final_norm = LayerNorm(d_model, eps=1e-5)
        self.model_encoder = TransformerEncoder(encoder_layer, nlayers, encoder_norm)

    def forward(self, root, left):
        x = torch.stack([root, left], dim=0)
        x += self.embedding_pos.weight
        return self.final_norm(self.model_encoder(x.unsqueeze(0))).squeeze(0)[0]


class trivial_unary_agg(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        norm_first,
        nlayers,
        batch_first=True,
        activation="relu",
    ):
        nn.Module.__init__(self)

    def forward(self, root, left):
        return root


class trivial_binary_agg(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        norm_first,
        nlayers,
        batch_first=True,
        activation="relu",
    ):
        nn.Module.__init__(self)

    def forward(self, root, left, right):
        return root


class skipmlp_unary_agg(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        norm_first,
        nlayers,
        batch_first=True,
        activation="relu",
    ):
        nn.Module.__init__(self)
        self.inp = nn.Linear(d_model * 2, d_model)
        self.hidden = nn.Sequential(
            *sum(
                [
                    [nn.ReLU(), nn.Linear(d_model, d_model)]
                    if i != nlayers - 1
                    else [nn.ReLU(), nn.Linear(d_model, d_model)]
                    for i in range(nlayers)
                ],
                [],
            )
        )

    def forward(self, root, left):
        return root + self.hidden(self.inp(torch.cat([root, left], dim=0)))


class skipmlp_binary_agg(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        norm_first,
        nlayers,
        batch_first=True,
    ):
        nn.Module.__init__(self)
        self.inp = nn.Linear(d_model * 3, d_model)
        self.hidden = nn.Sequential(
            *sum(
                [
                    [nn.ELU(), nn.Linear(d_model, d_model)]
                    if i != nlayers - 1
                    else [nn.ELU(), nn.Linear(d_model, d_model)]
                    for i in range(nlayers)
                ],
                [],
            )
        )

    def forward(self, root, left, right):
        return root + self.hidden(self.inp(torch.cat([root, left, right], dim=0)))


class simplemlp_binary_agg(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        norm_first,
        nlayers,
        batch_first=True,
    ):
        nn.Module.__init__(self)
        self.imp = nn.Linear(d_model * 2, d_model)
        self.hidden = nn.Sequential(
            *sum(
                [
                    [nn.ELU(), nn.Linear(d_model, d_model)]
                    if i != nlayers - 1
                    else [nn.ELU(), nn.Linear(d_model, d_model)]
                    for i in range(nlayers)
                ],
                [],
            )
        )

    def forward(self, root, left, right):
        return self.hidden(self.imp(torch.cat([left, right], dim=0)))


class aggregated_embedding(nn.Module):
    def __init__(self, n_vocab, d_model, n_nts=0, agg_type="trivial"):
        """
        agg_type:
            - trivial : returns the embedding of the root node
            - skipmlp : uses a skip mlp
            - transformer : uses a transformer
        """
        nn.Module.__init__(self)
        self.d_model = d_model
        self.n_vocab = n_vocab
        self.n_nts = n_nts
        self.embedding_tgt = nn.Embedding(n_vocab + n_nts, d_model)
        if agg_type == "trivial":
            unary_agg, binary_agg = trivial_unary_agg, trivial_binary_agg
        elif agg_type == "skipmlp":
            unary_agg, binary_agg = skipmlp_unary_agg, skipmlp_binary_agg
        elif agg_type == "simplemlp":
            unary_agg = None
            binary_agg = simplemlp_binary_agg
        elif agg_type == "transformer":
            unary_agg, binary_agg = transformer_unary_agg, transformer_binary_agg
        if unary_agg is not None:
            self.uni_agg = unary_agg(
                d_model=d_model,
                nhead=4,
                dim_feedforward=4 * d_model,
                dropout=0,
                norm_first=True,
                nlayers=2,
            )
        else:
            self.uni_agg = lambda x, y: y
        self.bin_agg = binary_agg(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            dropout=0,
            norm_first=True,
            nlayers=2,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(
            self.embedding_tgt.weight, mean=0.0, std=1 / self.d_model**0.5
        )

    def forward(self, tgt, ignore_data=False, ebm_cache=False):
        encoded_tgt = torch.cat(
            [
                x.left.get_emb(
                    self.bin_agg,
                    self.uni_agg,
                    self.embedding_tgt.weight,
                    ignore_data=ignore_data,
                    ebm_cache=ebm_cache,
                ).unsqueeze(0)
                for x in tgt
            ],
            axis=0,
        ).to(self.embedding_tgt.weight.device)
        return encoded_tgt


class TreeModel(torch.nn.Module):
    def __init__(self, aggregator, d_model, device):
        super().__init__()
        self.aggregator = aggregator
        self.device = device
        self.output = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ELU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.ELU(),
            nn.Linear(d_model * 2, 1),
        )

    def forward(self, batch_trees):
        if len(batch_trees) > 0:
            agg_embed = self.aggregator(batch_trees, ignore_data=True, ebm_cache=True)
            return self.output(agg_embed)
        else:
            return torch.tensor([]).to(self.device)
