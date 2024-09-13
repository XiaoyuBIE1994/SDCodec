import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import julius # a little bit worse than soxr, but compatible with torch and cuda
from einops import rearrange
from collections import namedtuple

from ..modules import WNConv1d, WNConv2d

def get_window(window_type: str, window_length: int, device: str):
        if window_type == "average":
            window = torch.ones(window_length) / window_length
        elif window_type == "sqrt_hann":
            window = torch.hann_window(window_length).sqrt()
        else:
            win_fn = getattr(torch, f'{window_type}_window')
            window = win_fn(window_length)

        window = window.to(device)
        return window


class MSD(nn.Module):
    def __init__(self, rate: int = 1, sample_rate: int = 44100):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                WNConv1d(1, 16, 15, 1, padding=7, act=True),
                WNConv1d(16, 64, 41, 4, groups=4, padding=20, act=True),
                WNConv1d(64, 256, 41, 4, groups=16, padding=20, act=True),
                WNConv1d(256, 1024, 41, 4, groups=64, padding=20, act=True),
                WNConv1d(1024, 1024, 41, 4, groups=256, padding=20, act=True),
                WNConv1d(1024, 1024, 5, 1, padding=2, act=True),
            ]
        )
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        
        # julius resample 
        # https://adefossez.github.io/julius/julius/resample.html
        old_sr = sample_rate
        new_sr = sample_rate // rate
        gcd = math.gcd(old_sr, new_sr)
        old_sr = old_sr // gcd
        new_sr = new_sr // gcd
        self.resample = julius.ResampleFrac(old_sr=old_sr, new_sr=new_sr)

    def forward(self, x):
        x = self.resample(x)
        fmap = []

        for l in self.convs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class MPD(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0), act=True),
                WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0), act=True),
                WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0), act=True),
                WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0), act=True),
                WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0), act=True),
            ]
        )
        self.conv_post = WNConv2d(
            1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
        )

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
STFTParams = namedtuple(
    "STFTParams",
    ["window_length", "hop_length"],
)

STFTParams = namedtuple(
    "STFTParams",
    ["window_length", "hop_length", "window_type", "padding_type", "match_stride"],
)

class MRD(nn.Module):
    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
        bands: list = BANDS,
    ):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.stft_params = STFTParams(
            window_length=window_length,
            hop_length=int(window_length * hop_factor),
            window_type='hann',
            padding_type='reflect',
            match_stride=True,
        )

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch = 32
        convs = lambda: nn.ModuleList(
            [
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4), act=True),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4), act=True),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4), act=True),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4), act=True),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1), act=True),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        B, C, T = x.shape
        x = torch.stft(x.reshape(-1, T), 
                       n_fft=self.stft_params.window_length, 
                       hop_length=self.stft_params.hop_length, 
                       win_length=self.stft_params.window_length, 
                       window=torch.hann_window(self.stft_params.window_length).to(x.device),
                       pad_mode='reflect', center=True, onesided=True, return_complex=True)
        x = torch.view_as_real(x)
        x = rearrange(x, "n f t c -> n c t f")
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class Discriminator(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        rates: list = [],
        periods: list = [2, 3, 5, 7, 11],
        fft_sizes: list = [2048, 1024, 512],
        bands: list = BANDS,
    ):
        """Discriminator that combines multiple discriminators.

        Parameters
        ----------
        sample_rate : int, needed
            Sampling rate of audio in Hz
        rates : list, optional
            MSD, sampling rates (in Hz), by default []
            If empty, MSD is not used.
        periods : list, optional
            MPD, periods of samples, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            MRD, window sizes of the FFT, by default [2048, 1024, 512]
        bands : list, optional
            MRD, bands, by default `BANDS`
        """
        super().__init__()
        discs = []
        discs += [MSD(r, sample_rate=sample_rate) for r in rates]
        discs += [MPD(p) for p in periods]
        discs += [MRD(f, sample_rate=sample_rate, bands=bands) for f in fft_sizes]
        self.discriminators = nn.ModuleList(discs)

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        x = self.preprocess(x)
        fmaps = [d(x) for d in self.discriminators]
        return fmaps


if __name__ == "__main__":
    disc = Discriminator(sample_rate=44100,
                            rates=[],
                            periods=[2, 3, 5, 7, 11],
                            fft_sizes=[2048, 1024, 512],
                            bands=[[0.0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]]).to('cuda')
    x = torch.zeros(1, 1, 44100).to('cuda')

    total_params = sum(p.numel() for p in disc.parameters()) / 1e6
    print(f'Total params: {total_params:.2f} Mb')

    results = disc(x)
    print(len(results))

    for i, result in enumerate(results):
        print(f"disc{i}")
        for i, r in enumerate(result):
            print(r.shape, r.mean(), r.min(), r.max())
        print()
