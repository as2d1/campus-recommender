"""Simplified DeepFM model for course-project ranking."""

from __future__ import annotations

import torch
from torch import nn


class DeepFM(nn.Module):
    """DeepFM with sparse embeddings, FM terms, dense inputs, and DNN."""

    def __init__(
        self,
        field_dims: list[int],
        dense_dim: int,
        embed_dim: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        self.field_dims = field_dims
        self.num_sparse_fields = len(field_dims)
        self.dense_dim = dense_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims

        self.first_order_embeddings = nn.ModuleList(
            [nn.Embedding(field_dim, 1) for field_dim in field_dims]
        )
        self.feature_embeddings = nn.ModuleList(
            [nn.Embedding(field_dim, embed_dim) for field_dim in field_dims]
        )
        self.dense_first_order = nn.Linear(dense_dim, 1) if dense_dim > 0 else None

        dnn_input_dim = self.num_sparse_fields * embed_dim + dense_dim
        layers: list[nn.Module] = []
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(dnn_input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            dnn_input_dim = hidden_dim
        self.dnn = nn.Sequential(*layers) if layers else nn.Identity()
        self.dnn_output = nn.Linear(dnn_input_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embedding parameters."""

        for embedding in self.first_order_embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)
        for embedding in self.feature_embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)

    def fm_first_order(self, sparse_x: torch.Tensor, dense_x: torch.Tensor | None) -> torch.Tensor:
        """FM linear part."""

        sparse_first = [
            embedding(sparse_x[:, idx])
            for idx, embedding in enumerate(self.first_order_embeddings)
        ]
        first_order = torch.stack(sparse_first, dim=1).sum(dim=1)
        if self.dense_first_order is not None and dense_x is not None:
            first_order = first_order + self.dense_first_order(dense_x)
        return first_order

    def fm_second_order(self, sparse_x: torch.Tensor) -> torch.Tensor:
        """FM pairwise interaction part."""

        embeddings = [
            embedding(sparse_x[:, idx])
            for idx, embedding in enumerate(self.feature_embeddings)
        ]
        embed_stack = torch.stack(embeddings, dim=1)
        square_of_sum = torch.sum(embed_stack, dim=1) ** 2
        sum_of_square = torch.sum(embed_stack ** 2, dim=1)
        second_order = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return second_order

    def dnn_part(self, sparse_x: torch.Tensor, dense_x: torch.Tensor | None) -> torch.Tensor:
        """DNN high-order feature part."""

        embeddings = [
            embedding(sparse_x[:, idx])
            for idx, embedding in enumerate(self.feature_embeddings)
        ]
        dnn_input = torch.cat(embeddings, dim=1)
        if dense_x is not None and self.dense_dim > 0:
            dnn_input = torch.cat([dnn_input, dense_x], dim=1)
        return self.dnn_output(self.dnn(dnn_input))

    def forward(self, sparse_x: torch.Tensor, dense_x: torch.Tensor | None = None) -> torch.Tensor:
        """Return logits for binary positive-feedback prediction."""

        first_order = self.fm_first_order(sparse_x, dense_x)
        second_order = self.fm_second_order(sparse_x)
        dnn_output = self.dnn_part(sparse_x, dense_x)
        logits = first_order + second_order + dnn_output + self.bias
        return logits.squeeze(1)

    def predict_proba(self, sparse_x: torch.Tensor, dense_x: torch.Tensor | None = None) -> torch.Tensor:
        """Return sigmoid probabilities."""

        return torch.sigmoid(self.forward(sparse_x, dense_x))
