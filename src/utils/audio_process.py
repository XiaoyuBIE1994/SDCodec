import torch
import numpy as np
from torch import nn
import torchaudio
from collections import namedtuple
from einops import rearrange

def normalize_mean_var_np(wav, eps=1e-8, std=None):
    mean = np.mean(wav, axis=-1, keepdims=True)
    if std is None:
        std = np.std(wav, axis=-1, keepdims=True)
    return (wav - mean) / (std + eps)

def normalize_max_norm_np(wav):
    return wav /np.max(np.abs(wav), axis=-1, keepdims=True)


def normalize_mean_var(wav, eps=1e-8, std=None):
    mean = wav.mean(-1, keepdim=True)
    if std is None:
        std = wav.std(-1, keepdim=True)
    return (wav - mean) / (std + eps)

def normalize_max_norm(wav):
    return wav / wav.abs().max(-1, keepdim=True)

def db_to_gain(db):
    return np.power(10.0, db/20.0)



class VolumeNorm:
    """
    Volume normalization to a specific loudness [LUFS standard]
    """
    def __init__(self, sample_rate=16000):
        self.lufs_meter = torchaudio.transforms.Loudness(sample_rate)

    def __call__(self, signal, target_loudness=-30, var=0, return_gain=False):
        """
        signal: torch.Tensor [B, channels, L]
        """
        bs = signal.shape[0]
        # LUFS diff
        lufs_ref = self.lufs_meter(signal)
        lufs_target = (target_loudness + (torch.rand(bs) * 2 - 1) * var).to(lufs_ref.device)
        # db to gain
        gain = torch.exp((lufs_target - lufs_ref) * np.log(10) / 20) 
        gain[gain.isnan()] = 0 # zero gain for silent audio
        # norm
        signal *= gain[:, None, None]

        if return_gain:
            return signal, gain
        else:
            return signal


STFTParams = namedtuple(
    "STFTParams",
    ["window_length", "hop_length", "window_type", "padding_type"],
)
STFT_PARAMS = STFTParams(
                window_length=1024,
                hop_length=256,
                window_type="hann",
                padding_type="reflect",
            )


class WavSepMagNorm:
    """
    Normalize the separation results using the magnitude
    X_i = (|X_i| / sum_k<|X_i|> * |X_mix|) * exp(j arg<X_mix>)
    """
    def __init__(self):
        self.stft_params = STFT_PARAMS
    
    def __call__(self, mix, signal_sep):
        """
        Parameters
        ----------
        mix: torch.Tensor [B, 1, channels, L]
            Mixture signal
        signal: torch.Tensor [B, K, channels, L]
            Separation results without normalization
        Returns
        -------
        ret:  torch.Tensor [B, K, channels, L']
            Separation results
        """
        bs, K, channels, _ = signal_sep.shape
        mix = rearrange(mix, 'b k c l -> (b k c) l')
        signal_sep = rearrange(signal_sep, 'b k c l -> (b k c) l')

        mix_spec = torch.stft(mix, n_fft=self.stft_params.window_length, hop_length=self.stft_params.hop_length, 
                             win_length=self.stft_params.window_length,
                             window=torch.hann_window(self.stft_params.window_length, device=mix.device),
                             pad_mode=self.stft_params.padding_type, center=True, onesided=True, return_complex=True)
        signal_sep_spec = torch.stft(signal_sep, n_fft=self.stft_params.window_length, hop_length=self.stft_params.hop_length, 
                             win_length=self.stft_params.window_length,
                             window=torch.hann_window(self.stft_params.window_length, device=signal_sep.device),
                             pad_mode=self.stft_params.padding_type, center=True, onesided=True, return_complex=True)
        
        mix_spec = rearrange(mix_spec, '(b k c) n t -> b k c n t', k=1, c=channels)
        signal_sep_spec = rearrange(signal_sep_spec, '(b k c) n t -> b k c n t', k=K, c=channels)

        signal_sep_mag = signal_sep_spec.abs() # (B, K, C, N, T)
        ratio = signal_sep_mag / signal_sep_mag.sum(dim=1, keepdim=True)
        ret_spec = torch.polar(mix_spec.abs() * ratio, mix_spec.angle())

        ret_spec = rearrange(ret_spec, 'b k c n t -> (b k c) n t')
        ret = torch.istft(ret_spec, n_fft=self.stft_params.window_length, hop_length=self.stft_params.hop_length,
                          win_length=self.stft_params.window_length,
                          window=torch.hann_window(self.stft_params.window_length, device=mix.device),
                          center=True)
        ret = rearrange(ret, '(b k c) l -> b k c l', k=K, c=channels)

        return ret


if __name__ == '__main__':

    import torch
    sep_norm = WavSepMagNorm()
    mix = torch.randn(32, 1, 1, 16000)
    signal_sep = torch.randn(32, 3, 1, 16000)
    ret = sep_norm(mix, signal_sep)
    print(ret.shape)
    print(((mix[...,:ret.shape[-1]] - ret.sum(dim=1, keepdim=True)) < 1e-6).all())

