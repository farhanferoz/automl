"""Embedding table with one extra row for categories the network has not learned (never seen in training, or hidden on purpose).

Searched before writing: `preprocessing.py` (`OrderedTargetEncoder`, `OneHotEncoder`) encodes categories for tabular models and has an
unknown-category rule of its own, but nothing in `models/` gave a neural embedding a "not seen" slot; `nn.Embedding` alone fails or
returns an untrained random row. This module is generic: any network with an index-to-vector table can use it.
"""

import torch
import torch.nn as nn


class UnknownCategoryEmbedding(nn.Module):
    """`nn.Embedding` over `n_categories` known rows plus one "unknown" row (index `n_categories`), initialised to zeros.

    The known rows are created and initialised exactly as a plain `nn.Embedding(n_categories, dim)` followed by
    `nn.init.normal_(weight, std=init_std)` would (same random draws), and the unknown row is appended without drawing anything, so
    adding the row changes neither the known rows nor any later initialisation.

    Which categories had training rows is stored in the buffer `seen`; an index with `seen == False` is looked up as unknown, in
    every mode. In training mode, with probability `unknown_rate` per row (vectorised, torch's global generator) an index is replaced by
    unknown as well, so the network learns a useful vector for the row. A rate of 0 draws nothing.
    """

    seen: torch.Tensor

    def __init__(self, n_categories: int, embedding_dim: int, init_std: float, seen: torch.Tensor, unknown_rate: float = 0.0) -> None:
        """Builds the table.

        Args:
            n_categories: number of known categories (indices `0 .. n_categories - 1`).
            embedding_dim: width of each vector.
            init_std: standard deviation of the normal draw initialising the known rows.
            seen: bool tensor `(n_categories,)`, True for the categories that have training rows.
            unknown_rate: probability, in training mode, of replacing a row's category by unknown (0 to 1, exclusive of 1).
        """
        super().__init__()
        if not 0.0 <= unknown_rate < 1.0:
            raise ValueError(f"unknown_rate must be in [0, 1), got {unknown_rate}.")
        known = nn.Embedding(n_categories, embedding_dim)
        nn.init.normal_(known.weight, std=init_std)
        self.n_categories, self.unknown_rate = n_categories, unknown_rate
        self.embedding = nn.Embedding.from_pretrained(torch.cat([known.weight.detach(), torch.zeros(1, embedding_dim)]), freeze=False)
        self.register_buffer("seen", seen.to(torch.bool).clone())

    @property
    def weight(self) -> torch.Tensor:
        """The `(n_categories + 1, embedding_dim)` table; the last row is the unknown row."""
        return self.embedding.weight

    def n_unseen(self, index: torch.Tensor) -> int:
        """How many of the rows' category indices have no training rows (they are looked up as unknown)."""
        return int((~self.seen[index]).sum().item())

    def forward(self, index: torch.Tensor) -> torch.Tensor:
        """`(B,)` long category indices to `(B, embedding_dim)` vectors."""
        index = torch.where(self.seen[index], index, torch.full_like(index, self.n_categories))
        if self.training and self.unknown_rate > 0.0:
            hide = torch.rand(index.shape, device=index.device) < self.unknown_rate
            index = torch.where(hide, torch.full_like(index, self.n_categories), index)
        return self.embedding(index)
