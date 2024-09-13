"""
Basic NN module from DAC
"""
import math

import torch
from torch import nn

from .layers import Snake1d, WNConv1d, WNConvTranspose1d, TransformerSentenceEncoderLayer


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        k = 7 # kernal size for the first conv
        pad = ((k - 1) * dilation) // 2 # 2*p - d*(k-1)
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=k, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2 # identical in-out channel 
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y
    

class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.floor(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)

    
class DACEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 1024,
    ):
        super().__init__()
        # Create first convolution
        layers = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlockd_models that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            layers += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        layers += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*layers)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DACEncoderTrans(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        att_d_model: int = 512,
        att_nhead: int = 8,
        att_ff: int = 2048,
        att_norm_first: bool = False,
        att_layers: int = 1,
        d_latent: int = 1024,
    ):
        super().__init__()
        # Create first convolution
        layers = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlockd_models that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            layers += [EncoderBlock(d_model, stride=stride)]

        # Create convolution
        layers += [
            Snake1d(d_model),
            WNConv1d(d_model, att_d_model, kernel_size=3, padding=1),
        ]

        # Create attention layers
        layers += [
            TransformerSentenceEncoderLayer(d_model=att_d_model, nhead=att_nhead, 
                                            dim_feedforward=att_ff, norm_first=att_norm_first,
                                            num_layers=att_layers)
        ]

        # Create last convolution
        layers += [
            Snake1d(att_d_model),
            WNConv1d(att_d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*layers)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)
    

class DACDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 1536,
        strides: list = [8, 8, 4, 2],
        d_latent: int = 1024,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(d_latent, d_model, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks (from HiFi GAN)
        for stride in strides:
            layers += [DecoderBlock(d_model, d_model//2, stride)]
            d_model = d_model // 2

        # Add final conv layer
        layers += [
            Snake1d(d_model),
            WNConv1d(d_model, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DACDecoderTrans(nn.Module):
    def __init__(
        self,
        d_model: int = 1536,
        strides: list = [8, 8, 4, 2],
        d_latent: int = 1024,
        att_d_model: int = 512,
        att_nhead: int = 8,
        att_ff: int = 2048,
        att_norm_first: bool = False,
        att_layers: int = 1,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(d_latent, att_d_model, kernel_size=7, padding=3)]

        # Add attention layer
        layers += [
            TransformerSentenceEncoderLayer(d_model=att_d_model, nhead=att_nhead, 
                                            dim_feedforward=att_ff, norm_first=att_norm_first,
                                            num_layers=att_layers)
        ]

        # Add conv layer
        layers += [
            Snake1d(att_d_model),
            WNConv1d(att_d_model, d_model, kernel_size=7, padding=3)
        ]

        # Add upsampling + MRF blocks (from HiFi GAN)
        for stride in strides:
            layers += [DecoderBlock(d_model, d_model//2, stride)]
            d_model = d_model // 2

        # Add final conv layer
        layers += [
            Snake1d(d_model),
            WNConv1d(d_model, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    


class CodecMixin:
    """Truncated version of DAC CodecMixin
    """
    def get_delay(self):
        # Any number works here, delay is invariant to input length
        l_out = self.get_output_length(0)
        L = l_out

        layers = []
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                layers.append(layer)

        for layer in reversed(layers):
            d = layer.dilation[0]
            k = layer.kernel_size[0]
            s = layer.stride[0]

            if isinstance(layer, nn.ConvTranspose1d):
                L = ((L - d * (k - 1) - 1) / s) + 1
            elif isinstance(layer, nn.Conv1d):
                L = (L - 1) * s + d * (k - 1) + 1

            L = math.ceil(L)

        l_in = L

        return (l_in - l_out) // 2

    def get_output_length(self, input_length):
        L = input_length
        # Calculate output length
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d = layer.dilation[0]
                k = layer.kernel_size[0]
                s = layer.stride[0]

                if isinstance(layer, nn.Conv1d):
                    L = ((L - d * (k - 1) - 1) / s) + 1
                elif isinstance(layer, nn.ConvTranspose1d):
                    L = (L - 1) * s + d * (k - 1) + 1

                L = math.floor(L)
        return L