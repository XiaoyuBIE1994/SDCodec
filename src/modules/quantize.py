"""
Modified from DAC: https://github.com/descriptinc/descript-audio-codec
Correct some arguement description
"""
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .layers import WNConv1d, Jitter


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        "z_q" : Tensor[B x D x T]
            Quantized continuous representation of input
        "vq/commitment_loss" : Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        "vq/codebook_loss" : Tensor[1]
            Codebook loss to update the codebook
        "codes" : Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        "latents" : Tensor[B x D' x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 1024,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.

        Returns
        -------
        "z_q" : Tensor[B x D x T]
            Quantized continuous representation of input
        "codes" : Tensor[B x N x T]
            Codebook indices for each codebook
        "latents" : Tensor[B x N*D' x T]
            Concatenated projected latents (continuous representation of input before quantization)
        "vq/commitment_loss" : Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook entries
        "vq/codebook_loss" : Tensor[1]
            Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Codebook indices for each codebook

        Returns
        -------
        "z_q" : Tensor[B x D x T]
            Quantized continuous representation of input
        "z_p" : Tensor[B x N*D' x T]
            Concatenated quantized codes before up-projection
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N*D x T]
            Concatenated projected latents (continuous representation of input before quantization)

        Returns
        -------
        "z_q" : Tensor[B x D x T]
            Quantized representation of full-projected space
        "z_p" : Tensor[B x N*D' x T]
            Concatenated quantized codes before up-projection
        "codes" : Tensor[B x N x T]
            Codebook indices for each codebook
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


class MultiSourceRVQ(nn.Module):
    """
    Parallele RVQ for multiple sources
    """
    def __init__(
        self,
        tracks: list[str] = ['speech', 'music', 'sfx'],
        input_dim: int = 1024,
        n_codebooks: list[int] = [12, 12, 12],
        codebook_size: list[int] = [1024, 1024, 1024],
        codebook_dim: list[int] = [8, 8, 8],
        quantizer_dropout: float = 0.0,
        code_jit_prob: list[float] = [0.0, 0.0, 0.0],
        code_jit_size: list[int] = [3, 5, 3],
        shared_codebooks: int = 8,
    ):
        super().__init__()
        self.tracks = tracks
        self.input_dim  = input_dim
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.code_jit_prob = code_jit_prob
        self.code_jit_size = code_jit_size
        self.shared_codebooks = shared_codebooks

        self.rvq_dict = nn.ModuleDict()
        self.jitter_dict =  nn.ModuleDict()

        for i, t in enumerate(self.tracks):
            self.rvq_dict[t] = ResidualVectorQuantize(
                input_dim=self.input_dim,
                n_codebooks=self.n_codebooks[i]-self.shared_codebooks,
                codebook_size=self.codebook_size[i],
                codebook_dim=self.codebook_dim[i],
                quantizer_dropout=self.quantizer_dropout,
            )
            self.jitter_dict[t] = Jitter(
                p=self.code_jit_prob[i],
                size=self.code_jit_size[i],
            )

        if shared_codebooks > 0:
            self.shared_rvq = ResidualVectorQuantize(
                input_dim=self.input_dim,
                n_codebooks=self.shared_codebooks,
                codebook_size=self.codebook_size[0],
                codebook_dim=self.codebook_dim[0],
                quantizer_dropout=self.quantizer_dropout,
            )

    def forward(self, track_name, feats, n_quantizers: int = None):
        assert track_name in self.tracks, '{} not in model tracks: {}'.format(track_name, self.tracks)
        
        # number of quantizer to be used
        if n_quantizers is None:
            n_quantizers = self.n_codebooks[self.tracks.index(track_name)]

        # quantize
        z, codes, latents, commitment_loss, codebook_loss = self.rvq_dict[track_name](
            feats, n_quantizers
        )

        # shared codebook
        if self.shared_codebooks > 0 and n_quantizers > self.rvq_dict[track_name].n_codebooks:
            z_shard, codes_shard, latents_shard, commitment_loss_shard, codebook_loss_shard = self.shared_rvq(
            feats-z, n_quantizers-self.rvq_dict[track_name].n_codebooks
        )
            z = z + z_shard
            codes = torch.cat((codes, codes_shard), dim=1) # (B, N, T)
            latents = torch.cat((latents, latents_shard), dim=1) # (B, N*D', T)
            commitment_loss += commitment_loss_shard
            codebook_loss += codebook_loss_shard

        # jitter, ignored if prob <= 0
        z = self.jitter_dict[track_name](z) # (B, D, T)

        return z, codes, latents, commitment_loss, codebook_loss


if __name__ == "__main__":
    rvq = MultiSourceRVQ(quantizer_dropout=0.0)
    feats = torch.randn(16, 1024, 20)
    z_q, codes, latents, commitment_loss, codebook_loss = rvq('speech', feats)
    print(z_q.shape)
    print(codes.shape)
    print(latents.shape)
    print(commitment_loss)
    print(codebook_loss)

    # z_q, z_p, codes = rvq.from_latents(latents)
    # print(z_q[0,0])
    # print(codes[0,0])