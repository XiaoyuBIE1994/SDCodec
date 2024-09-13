"""
ResDQ DAC-based Residual VQ-VAE model.
"""
import math
import numpy as np
from typing import List, Union, Tuple
from typing import Optional
import random
from omegaconf import DictConfig

import torch
from torch import nn
import torch.nn.functional as F
from accelerate.logging import get_logger

from ..modules import CodecMixin
from .. import modules

logger = get_logger(__name__)
                    
def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02) # default kaiming_uniform_
        nn.init.constant_(m.bias, 0) # default uniform_


class SDCodec(nn.Module, CodecMixin):
    """Source-aware Disentangled Neural Audio Codec.
    Args:
        
    """
    def __init__(
        self,
        sample_rate: int,
        latent_dim: int = None,
        tracks: List[str] = ['speech', 'music', 'sfx'],
        enc_params: DictConfig = {'name': 'DACEncoder'},
        dec_params: DictConfig = {'name': 'DACDecoder'},
        quant_params: DictConfig = {'name': 'DACDecoder'},
        pretrain: dict = {},
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.tracks = tracks
        self.enc_params = enc_params
        self.dec_params = dec_params
        self.quant_params = quant_params
        self.pretrain = pretrain


        if latent_dim is None:
            latent_dim = self.enc_params.d_model * (2 ** len(self.enc_params.strides))
        self.latent_dim = latent_dim
        self.hop_length = np.prod(self.enc_params.strides)
        

        # Define encoder and decoder
        enc_net = getattr(modules, self.enc_params.pop('name'))
        dec_net = getattr(modules, self.dec_params.pop('name'))
        self.encoder = enc_net(**self.enc_params)
        self.decoder = dec_net(**self.dec_params)

        # Define quantizer
        quant_net = getattr(modules, self.quant_params.pop('name'))
        self.quantizer = quant_net(tracks=self.tracks, **self.quant_params)

        # Init
        self.apply(init_weights)
        self.delay = self.get_delay()

        # Load pretrained
        load_pretrained = self.pretrain.get('load_pretrained', False)
        if load_pretrained:
            pretrained_dict = torch.load(load_pretrained)
            hyparam = pretrained_dict['metadata']['kwargs']
            ignore_modules = self.pretrain.get('ignore_modules', [])
            freeze_modues = self.pretrain.get('freeze_modules', [])
            
            is_match = self._check_hyparam(hyparam)
            if is_match:
                self._load_pretrained(pretrained_dict['state_dict'], ignore_modules)
                self._freeze(freeze_modues)
                logger.info('Pretrain models load success from {}'.format(load_pretrained))
                logger.info('-> modules ignored: {}'.format(ignore_modules))
                logger.info('-> modules freezed: {}'.format(freeze_modues))
            else:
                logger.info(f'Pretrain param do not match model, load pretrained failed...')
                logger.info('Pretrain params:')
                logger.info(hyparam)
                logger.info('Model params:')
                logger.info('Encoder: {}'.format(self.enc_params))
                logger.info('Decoder: {}'.format(self.dec_params))

    @property
    def device(self):
        """Gets the device the model is on by looking at the device of
        the first parameter. May not be valid if model is split across
        multiple devices.
        """
        return list(self.parameters())[0].device


    def _check_hyparam(self, hyparam):
        return (hyparam['encoder_dim'] == self.enc_params['d_model']) \
            and (hyparam['encoder_rates'] == self.enc_params['strides']) \
            and (hyparam['decoder_dim'] == self.dec_params['d_model']) \
            and (hyparam['sample_rate'] == self.sample_rate)


    def _load_pretrained(self, pretrained_state, ignored_modules=[]):
        own_state = self.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k.split('.')[0] not in ignored_modules}
        for k in own_state.keys():
            if k in pretrained_state:
                own_state[k] = pretrained_state[k]
            elif 'quantizer' not in ignored_modules:
                own_key_list = k.split('.')
                if own_key_list[0] == 'quantizer':
                    if own_key_list[1] == 'jitter_dict':
                        continue
                    elif own_key_list[1] == 'shared_rvq':
                        own_key_list.pop(1) # shared_rvq
                    else:
                        own_key_list.pop(1) # rvq_dict
                        own_key_list.pop(1) # (speech, music, sfx)
                    pretrained_key = '.'.join(own_key_list)
                    if pretrained_key not in pretrained_state:
                        print(k)
                        print(pretrained_key)
                        breakpoint()
                    own_state[k] = pretrained_state[pretrained_key]


    def _freeze(self, freeze_modues=[]):
        for module in freeze_modues:
            child = getattr(self, module)
            for param in child.parameters():
                param.requires_grad = False


    def preprocess(self, audio_data, sample_rate) -> torch.Tensor:
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = F.pad(audio_data, (0, right_pad))

        return audio_data


    def encode(self, audio_data: torch.Tensor) -> torch.Tensor:
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode

        Returns
        -------
        "feats" : Tensor[B x D x T]
            Continuous features before quantization
        """
        return self.encoder(audio_data)
    

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input

        Returns
        -------
        "audio" : Tensor[B x 1 x length]
            Decoded audio data.
        """
        return self.decoder(z)
    

    def quantize(
        self,
        feats: torch.Tensor,
        track: str = 'speech',
        n_quantizers: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        feats : Tensor[B x D x T]
            Continuous features before quantization
        track: str
            Specify which quantizer to be used
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        "z" : Tensor[B x D x T]
            Quantized continuous representation of input
        "codes" : Tensor[B x N x T]
            Codebook indices for each codebook
            (quantized discrete representation of input)
        "latents" : Tensor[B x N*D' x T]
            Projected latents (continuous representation of input before quantization)
        "vq/commitment_loss" : Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        "vq/codebook_loss" : Tensor[1]
            Codebook loss to update the codebook
        """
        assert track in self.tracks, 'f{track} not included in quantizer'
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            track, feats, n_quantizers
        )
        return z, codes, latents, commitment_loss, codebook_loss


    def forward(
        self,
        batch: Optional[dict],
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        """Model forward pass

        Parameters
        ----------
        batch : dict of Tensor[B x 1 x T]
            Batch input with audio data
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.
        Returns
        -------
        dict
            A dictionary with the following keys:
            "track/z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "track/codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "track/latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "ref" : Tensor[B x (K+1) x 1 x length]
                Decoded audio data.
            "recon" : Tensor[B x (K+1) x 1 x length]
                Decoded audio data.
        """
        
        # mix by masking 
        audio_data = batch['mix']
        bs, _, length = audio_data.shape
        valid_tracks = batch['valid_tracks']
        mask = [1 if t in valid_tracks else 0 for t in self.tracks]
        mask = torch.tensor(mask, device=audio_data.device)

        # preprocess, zero-padding to proper length
        audio_data = self.preprocess(audio_data, sample_rate)

        # encoder
        feats = self.encode(audio_data)

        # quantize
        dict_z = {}
        dict_commit = {}
        dict_cb = {}
        # for i, track in enumerate(self.tracks):
        for i, track in enumerate(self.tracks):
            # quantize
            z, codes, latents, commitment_loss, codebook_loss = self.quantize(
                feats, track, n_quantizers 
            )
            # ppl
            probs =  F.one_hot(codes.detach(), num_classes=self.quant_params.codebook_size[i]).float() # (B, N, T, num_class)
            avg_probs = probs.mean(dim=(0,2)) # (N, num_class)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)) # (N,)
            # save retsults
            batch[f'{track}/z'] = z
            batch[f'{track}/codes'] = codes
            batch[f'{track}/latents'] = latents
            batch[f'{track}/ppl'] = perplexity
            dict_z[track] = z
            dict_commit[track] = commitment_loss
            dict_cb[track] = codebook_loss

        # commit and codebook loss
        commit_loss = (torch.stack([dict_commit[t] for t in self.tracks]) * mask).sum() / len(valid_tracks)
        cb_loss = (torch.stack([dict_cb[t] for t in self.tracks]) * mask).sum() / len(valid_tracks)
        
        # re-mix and decode
        if batch['random_swap']:
            z_mix = torch.stack([dict_z[t][batch['shuffle_list'][t]] for t in self.tracks])
        else:
            z_mix = torch.stack([dict_z[t] for t in self.tracks])
        z_mix = (z_mix * mask[:, None, None, None]).sum(dim=0)
        x_remix = self.decode(z_mix) # (B, 1, T)

        # collect data
        sep_out = [self.decode(dict_z[t]) for t in valid_tracks]
        audio_recon = torch.stack([x_remix] + sep_out, dim=1) # (B, K, 1, T)

        batch['recon'] = audio_recon[..., :length]
        batch['length'] = length
        batch['vq/commitment_loss'] = commit_loss
        batch['vq/codebook_loss'] = cb_loss

        return batch


    def evaluate(
            self,
            input_audio: torch.Tensor,
            sample_rate: int = None,
            n_quantizers: int = None,
            output_tracks: list[str] = ['mix'],
    ) -> torch.Tensor:
        """Model evaluation
        Parameters
        ----------
        input_audio : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.
        output_tracks : List[str]
            List of track to return

        Returns
        -------
        output_audio : Tensor[B x K x T]
            Output audio with K tracks
        """
        assert all((t in self.tracks) or (t=='mix') for t in output_tracks); \
        "output tracks {} not included in model tracks {}".format(output_tracks, self.tracks)

        bs, _, length = input_audio.shape
        audio_data = self.preprocess(input_audio, sample_rate) # (B, 1, T)

        # encoder
        feats = self.encode(audio_data)
        
        # quantization
        latent_dict = {}
        for track in self.tracks:
            z, codes, latents, commitment_loss, codebook_loss = self.quantize(
                feats, track, n_quantizers
            )
            latent_dict[track] = z
        
        # decoder
        list_out = []
        for track in output_tracks:
            if track == 'mix':
                z_mix = torch.stack(list(latent_dict.values()), dim=0).sum(dim=0)
                x_out = self.decode(z_mix)
                list_out.append(x_out)
            else:
                x_out = self.decode(latent_dict[track])
                list_out.append(x_out)
        
        output_audio = torch.cat(list_out, dim=1)[...,:length]

        return output_audio



if __name__ == "__main__":
    from accelerate import Accelerator
    from omegaconf import OmegaConf
    import numpy as np
    from functools import partial
    
    accelerator = Accelerator()
    logger = get_logger(__name__)
    
    cfg = OmegaConf.load('/home/xbie/Code/ResDQ/config/model/sdcodec_16k.yaml')
    sample_rate = 16000
    tracks = ['speech', 'music', 'sfx']
    mask = [True, False, False]
    show_params = False

    model_name = cfg.pop('name')
    model = SDCodec(sample_rate=sample_rate, **cfg).to("cuda")

    if show_params:
        for n, m in model.named_modules():
            o = m.extra_repr()
            p = sum(p.numel() for p in m.parameters())
            fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
            setattr(m, "extra_repr", partial(fn, o=o, p=p))
        print(model)
    else:
        print("Total # of params: {:.2f} M".format(sum(p.numel() for p in model.parameters())/1e6))

    print('Show model device')
    print(model.encoder.block[0].weight.device)
    print(model.quantizer_dict['speech'].quantizers[0].in_proj.weight.device)


    # test on random input
    bs = 2
    length = 1
    batch = {}
    length = int(sample_rate * length)
    for t in tracks:
        batch[t] = torch.randn(bs, 1, length).to(model.device)
        batch[t].requires_grad_(True)
        batch[t].retain_grad()

    # Make a forward pass
    batch = model(batch, 
                  track_mask=mask,
                  random_swap=True)
    print("Input shape:", batch['ref'].shape)
    print("Output shape:", batch['recon'].shape)

    # Create gradient variable
    B, N,_, T = batch['recon'].shape
    model_out = batch['recon'].reshape(-1, 1, T)
    grad = torch.zeros_like(model_out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    model_out.backward(grad)

    # Check non-zero values
    gradmap = batch['speech'].grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")


    ## test on real data
    import pandas as pd
    import torchaudio

    idx = 0
    tsv_filepath = '/home/xbie/Data/dnr_v2/val_mini_2s.tsv'
    
    metadata = pd.read_csv(tsv_filepath)
    wav_info = metadata.iloc[idx]

    x_speech, sr = torchaudio.load(wav_info['speech'])
    x_music, sr = torchaudio.load(wav_info['music'])
    x_sfx, sr = torchaudio.load(wav_info['sfx'])
    x_mix, sr = torchaudio.load(wav_info['mix'])

    x_speech = x_speech[..., wav_info['trim30dBs']: wav_info['trim30dBe']].reshape(1, 1, -1).to(model.device)
    x_music = x_music[..., wav_info['trim30dBs']: wav_info['trim30dBe']].reshape(1, 1, -1).to(model.device)
    x_sfx = x_sfx[..., wav_info['trim30dBs']: wav_info['trim30dBe']].reshape(1, 1, -1).to(model.device)

    batch = {'speech': x_speech,
             'music': x_music,
             'sfx': x_sfx}
    
    batch = model(batch, 
                  track_mask=mask,
                  random_swap=True)
    
    print('Reference audio shape: {}'.format(batch['ref'].shape))
    print('Recon audio shape: {}'.format(batch['recon'].shape))
    
    # batch['ref'] = batch['ref'].detach().cpu()
    # batch['recon'] = batch['recon'].detach().cpu()
    
    # # for i, c in enumerate(['speech', 'music', 'sfx', 'mix']):
    # #     path = '{}_{}_refRaw.wav'.format(wav_info['id'], c)
    # #     torchaudio.save(path, batch['ref'][0, i], 44100)
    
    # # for i, c in enumerate(['speech', 'music', 'sfx', 'mix']):
    # #     path = '{}_{}_reconRaw.wav'.format(wav_info['id'], c)
    # #     torchaudio.save(path, batch['recon'][0, i], 44100)
    # # breakpoint()
    # breakpoint()