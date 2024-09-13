
from typing import List, Callable
from collections import namedtuple
import numpy as np
from librosa.filters import mel

import torch
from torch import nn
from torch import Tensor

STFTParams = namedtuple(
    "STFTParams",
    ["window_length", "hop_length", "window_type", "padding_type"],
)


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


class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    window_type : str, optional
        Type of window to use, by default ``hann``.
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation modified from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        loss_fn: Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        window_type: str = "hann",
        padding_type: str = "reflect",
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                window_type=window_type,
                padding_type=padding_type
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.mag_weight = mag_weight
        self.log_weight = log_weight
        self.pow = pow


    def forward(self, est: Tensor, ref : Tensor):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        est : torch.Tensor [B, C, T]
            Estimate signal
        ref : torch.Tensor [B, C, T]
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        device = est.device
        if ref.device != est.device:
            est.to(device)

        assert est.shape == ref.shape, (
            'expected same shape, get est: {}, ref: {} instead'.format(est.shape, ref.shape)
        )
        assert len(est.shape) == len(ref.shape) == 3, (
            'expected BxCxN, get est: {}, ref: {} instead'.format(est.shape, ref.shape)
        )

        B, C, T = est.shape
        est = est.reshape(-1, T)
        ref = ref.reshape(-1, T)

        loss = 0.0
        for s in self.stft_params:
            est_spec = torch.stft(est, n_fft=s.window_length, hop_length=s.hop_length, win_length=s.window_length,         
                                  window=get_window(s.window_type, s.window_length, device),
                                  pad_mode=s.padding_type, center=True, onesided=True, return_complex=True)
            ref_spec = torch.stft(ref, n_fft=s.window_length, hop_length=s.hop_length, win_length=s.window_length,         
                                  window=get_window(s.window_type, s.window_length, device),
                                  pad_mode=s.padding_type, center=True, onesided=True, return_complex=True)

            loss += self.log_weight * self.loss_fn(
                est_spec.abs().clamp(self.clamp_eps).pow(self.pow).log10(),
                ref_spec.abs().clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(est_spec.abs(), ref_spec.abs())
        return loss


class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    window_type : str, optional
        Type of window to use, by default ``hann``.
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation modified from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        sr: int = 44100,
        n_mels: List[int] = [150, 80],
        mel_fmin: List[float] = [0.0, 0.0],
        mel_fmax: List[float] = [None, None],
        window_lengths: List[int] = [2048, 512],
        loss_fn: Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        window_type: str = "hann",
        padding_type: str = "reflect",
        match_stride: bool = False,
    ):
        assert len(n_mels) == len(window_lengths) == len(mel_fmin) == len(mel_fmax), \
            f'lengths are different, n_mels: {n_mels}, window_lengths: {window_lengths}, mel_fmin: {mel_fmin}, mel_fmax: {mel_fmax}'
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                window_type=window_type,
                padding_type=padding_type,
            )
            for w in window_lengths
        ]
        self.sr = sr
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, est: Tensor, ref : Tensor):
        """Computes multi-scale mel loss between an estimate and 
        a reference signal.

        Parameters
        ----------
        est : torch.Tensor [B, C, T]
            Estimate signal
        ref : torch.Tensor [B, C, T]
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale Mel loss.
        """
        device = est.device
        if ref.device != est.device:
            est.to(device)
        
        assert est.shape == ref.shape, (
            'expected same shape, get est: {}, ref: {} instead'.format(est.shape, ref.shape)
        )
        assert len(est.shape) == len(ref.shape) == 3, (
            'expected BxCxN, get est: {}, ref: {} instead'.format(est.shape, ref.shape)
        )

        B, C, T = est.shape
        est = est.reshape(-1, T)
        ref = ref.reshape(-1, T)

        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            est_spec = torch.stft(est, n_fft=s.window_length, hop_length=s.hop_length, win_length=s.window_length,         
                                  window=get_window(s.window_type, s.window_length, device),
                                  pad_mode=s.padding_type, center=True, onesided=True, return_complex=True)
            ref_spec = torch.stft(ref, n_fft=s.window_length, hop_length=s.hop_length, win_length=s.window_length,         
                                  window=get_window(s.window_type, s.window_length, device),
                                  pad_mode=s.padding_type, center=True, onesided=True, return_complex=True)
            
            # convert to mel
            est_mag = est_spec.abs()
            ref_mag = ref_spec.abs()

            mel_basis = mel(sr=self.sr, n_fft=s.window_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
            mel_basis = torch.from_numpy(mel_basis).to(device)

            est_mel = (est_mag.transpose(-1, -2) @ mel_basis.T).transpose(-1, -2)
            ref_mel = (ref_mag.transpose(-1, -2) @ mel_basis.T).transpose(-1, -2)


            loss += self.log_weight * self.loss_fn(
                est_mel.clamp(self.clamp_eps).pow(self.pow).log10(),
                ref_mel.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(est_mel, ref_mel)
            
        return loss