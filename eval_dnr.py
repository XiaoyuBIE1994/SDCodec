
import sys
import json
import argparse
import importlib
import math
import julius
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
from collections import namedtuple

from src import utils
from src.metrics import (
    VisqolMetric,
    SingleSrcNegSDR,
    MultiScaleSTFTLoss,
    MelSpectrogramLoss,
)

import torch
import torchaudio
from accelerate import Accelerator
accelerator = Accelerator()

parser = argparse.ArgumentParser(description='Generate manifest for audio dataset',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--ret-dir', type=str, default='output/debug', help='Training result directory')
parser.add_argument('--csv-path', type=str, default='./manifest/test.csv', help='csv file to test')
parser.add_argument('--data-sr', type=int, default=[44100], nargs='+', help='list of sampling rate in test files')
parser.add_argument('--length', type=int, default=10, help='audio length')
parser.add_argument('--visqol-mode', type=str, default='speech', choices=['speech', 'audio'], help='visqol mode')
parser.add_argument('--threshold', type=float, default=0.4, help='threshold of silence part to drop audio')
parser.add_argument('--fast', action='store_true', help='fast eval, disable visqol computation')

# parse
args = parser.parse_args()
ret_dir = Path(args.ret_dir)
csv_path = Path(args.csv_path)
length = args.length
visqol_mode = args.visqol_mode
threshold = args.threshold
use_visqol = not args.fast
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# read config
cfg_filepath = ret_dir / '.hydra' / 'config.yaml'
cfg = OmegaConf.load(cfg_filepath)
sample_rate = cfg.sampling_rate
chunk_len = sample_rate * length

# init julius resample
resample_pool = dict()
for sr in args.data_sr:
    old_sr = sr
    new_sr = sample_rate
    gcd = math.gcd(old_sr, new_sr)
    old_sr = old_sr // gcd
    new_sr = new_sr // gcd
    resample_pool[sr] = julius.ResampleFrac(old_sr=old_sr, new_sr=new_sr)

# import lib
model_name = cfg.model.pop('name')
module_path = str(ret_dir / 'backup_src' / 'models').replace('/', '.')
try:
    load_model = importlib.import_module(module_path)
    net_class = getattr(load_model, f'{model_name}')
    print('Load model from ckpt')
except:
    from src import models
    net_class = getattr(models, f'{model_name}')
    print('Load model from source code')

# load model and weigth
model_cfg = cfg.model
model = net_class(sample_rate=sample_rate, **model_cfg)
total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'Total params: {total_params:.2f} Mb')
print('Model sampling rate: {} Hz'.format(model.sample_rate))

ckpt_finalpath = ret_dir / 'ckpt_final' / 'ckpt_model_final.pth'
state_dict = torch.load(ckpt_finalpath, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(f'ckpt path: {ckpt_finalpath}')
print(f'Model weights load successfully...')

# prepare metrics
loss_cfg = cfg.training.loss
metric_stft = MultiScaleSTFTLoss(**loss_cfg.MultiScaleSTFTLoss)
metric_mel = MelSpectrogramLoss(**loss_cfg.MelSpectrogramLoss)
metric_sisdr = SingleSrcNegSDR(sdr_type='sisdr')
metric_visqol = VisqolMetric(mode=visqol_mode)

# prepare data transform
transform_cfg = cfg.training.transform
volume_norm = utils.VolumeNorm(sample_rate=sample_rate)
def _data_transform(batch, transform_cfg, valid_tracks=['speech'], norm_var=0):
    peak_norm = utils.db_to_gain(transform_cfg.peak_norm_db)
    mix_max_peak = torch.zeros_like(batch['speech'])[...,:1] # (bs, C, 1)

    # volume norm for each track
    for track in valid_tracks:
        batch[track] = volume_norm(signal=batch[track],
                                    target_loudness=transform_cfg.lufs_norm_db[track],
                                    var=norm_var)
        # peak value
        peak = batch[track].abs().max(dim=-1, keepdims=True)[0]
        mix_max_peak = torch.maximum(peak, mix_max_peak)
    
    # peak norm
    peak_gain = torch.ones_like(mix_max_peak) # (bs, C, 1)
    peak_gain[mix_max_peak > peak_norm] = peak_norm / mix_max_peak[mix_max_peak > peak_norm]
    
    # build mix
    batch['mix'] = torch.zeros_like(batch['speech'])
    for track in valid_tracks:
        batch[track] *= peak_gain
        batch['mix'] += batch[track]

    # mix volum norm
    batch['mix'], mix_gain = volume_norm(signal=batch['mix'],
                                        target_loudness=transform_cfg.lufs_norm_db['mix'],
                                        var=norm_var,
                                        return_gain=True)
    
    # norm each track
    for track in valid_tracks:
        batch[track] *= mix_gain[:, None, None]

    batch['valid_tracks'] = valid_tracks
    batch['random_swap'] = False

    return batch


# define mask separation
sep_norm = utils.WavSepMagNorm()

# define STFT params
STFTParams = namedtuple(
    "STFTParams",
    ["window_length", "hop_length", "window_type", "padding_type"],
)
stft_params = STFTParams(
                window_length=1024,
                hop_length=256,
                window_type="hann",
                padding_type="reflect",
            )

# run eval
tracks = model.tracks
print('Model tracks: {}'.format(tracks))
test_tracks = ['mix'] + [f'{t}_rec' for t in tracks] + [f'{t}_sep' for t in tracks] + [f'{t}_sep_mask' for t in tracks]
test_results = {t: {} for t in test_tracks}
metadata = pd.read_csv(csv_path)

for i in tqdm(range(len(metadata)), desc='Eval'):
# for i in tqdm(range(20), desc='Eval'):
    wav_info = metadata.iloc[i]
    audio_id = wav_info['id']
    start = wav_info['start']
    end = wav_info['end']
    batch = {}
    # read data
    for t in tracks:
        x, sr = torchaudio.load(wav_info[t])
        x = x.mean(dim=0)[..., start: end]
        if sr != sample_rate:
            x = resample_pool[sr](x)
        batch[t] = x
        audio_len = x.shape[-1]

    # clip audio
    for j, k in enumerate(range(0, audio_len-chunk_len+1, chunk_len)):
        clip_id = f'{audio_id}_{j}'
        eval_batch = {}
        mask = {}
        for t in tracks:
            audio_clip = batch[t][k:k+chunk_len]
            
            # silent audio detection
            audio_energy = torch.stft(audio_clip, n_fft=stft_params.window_length, hop_length=stft_params.hop_length, 
                             win_length=stft_params.window_length,
                             window=torch.hann_window(stft_params.window_length, device='cpu'),
                             pad_mode=stft_params.padding_type, center=True, onesided=True, return_complex=True).abs().sum(dim=0)
            count = sum(1 for item in audio_energy if item > 1e-6)
            silence_detect = count < threshold * len(audio_energy)
            mask[f'{t}_rec'] = silence_detect
            mask[f'{t}_sep'] = silence_detect
            mask[f'{t}_sep_mask'] = silence_detect
      
            eval_batch[t] = audio_clip.reshape(1,1,-1).to(device)
        
        mask['mix'] = all(mask.values())

        # data transform
        # eval_batch = _data_transform(eval_batch, transform_cfg=transform_cfg, valid_tracks=tracks, norm_var=0)
        eval_batch['mix'] = eval_batch['speech']+eval_batch['music']+eval_batch['sfx']
        eval_batch['valid_tracks'] = tracks
        eval_batch['random_swap'] = False
        
        # mixture forward
        with torch.no_grad():
            output_audio = model.evaluate(input_audio=eval_batch['mix'],
                                          output_tracks=['mix']+tracks)
            # eval_batch = model(eval_batch)
            # output_audio = eval_batch['recon'][:,:,0]

        # Eval mix reconstruction
        est = output_audio[:, 0].unsqueeze(1)
        ref = eval_batch['mix']
        test_results['mix'][clip_id] = {}
        if mask['mix']:
            test_results['mix'][clip_id]['stft'] = None
            test_results['mix'][clip_id]['mel'] = None
            test_results['mix'][clip_id]['sisdr'] = None
            if use_visqol:
                test_results['mix'][clip_id]['visqol'] = None
        else:
            test_results['mix'][clip_id]['stft'] = metric_stft(est=est, ref=ref).item()
            test_results['mix'][clip_id]['mel'] = metric_mel(est=est, ref=ref).item()
            test_results['mix'][clip_id]['sisdr'] = - metric_sisdr(est=est, ref=ref).item()
            if use_visqol:
                test_results['mix'][clip_id]['visqol'] = metric_visqol(est=est, ref=ref, sr=sample_rate)

        # Eval separation using synthesizer (decoder)
        for p, t in enumerate(tracks):
            est = output_audio[:,p+1].unsqueeze(1)
            ref = eval_batch[t]
            test_results[f'{t}_sep'][clip_id] = {}
            if mask[f'{t}_sep']:
                test_results[f'{t}_sep'][clip_id]['stft'] = None
                test_results[f'{t}_sep'][clip_id]['mel'] = None
                test_results[f'{t}_sep'][clip_id]['sisdr'] = None
                if use_visqol:
                    test_results[f'{t}_sep'][clip_id]['visqol'] = None
            else:
                test_results[f'{t}_sep'][clip_id]['stft'] = metric_stft(est=est, ref=ref).item()
                test_results[f'{t}_sep'][clip_id]['mel'] = metric_mel(est=est, ref=ref).item()
                test_results[f'{t}_sep'][clip_id]['sisdr'] = - metric_sisdr(est=est, ref=ref).item()
                if use_visqol:
                    test_results[f'{t}_sep'][clip_id]['visqol'] = metric_visqol(est=est, ref=ref, sr=sample_rate)
        
        # Eval separation using mask
        mix = eval_batch['mix'].unsqueeze(2)
        signal_sep = output_audio[:,1:].unsqueeze(2)
        all_sep_mask_norm = sep_norm(mix, signal_sep)
        for p, t in enumerate(tracks):
            est = all_sep_mask_norm[:,p]
            ref = eval_batch[t]
            ref = ref[...,:est.shape[-1]] # stft + istft. shorter
            # breakpoint()
            test_results[f'{t}_sep_mask'][clip_id] = {}
            if mask[f'{t}_sep_mask']:
                test_results[f'{t}_sep_mask'][clip_id]['stft'] = None
                test_results[f'{t}_sep_mask'][clip_id]['mel'] = None
                test_results[f'{t}_sep_mask'][clip_id]['sisdr'] = None
                if use_visqol:
                    test_results[f'{t}_sep'][clip_id]['visqol'] = None
            else:
                test_results[f'{t}_sep_mask'][clip_id]['stft'] = metric_stft(est=est, ref=ref).item()
                test_results[f'{t}_sep_mask'][clip_id]['mel'] = metric_mel(est=est, ref=ref).item()
                test_results[f'{t}_sep_mask'][clip_id]['sisdr'] = - metric_sisdr(est=est, ref=ref).item()
                if use_visqol:
                    test_results[f'{t}_sep_mask'][clip_id]['visqol'] = metric_visqol(est=est, ref=ref, sr=sample_rate)

        # Evaluate reconstruction of single track
        for p, t in enumerate(tracks):
            # single track forward
            with torch.no_grad():
                output_audio = model.evaluate(input_audio=eval_batch[t],
                                              output_tracks=[t])
            est = output_audio
            ref = eval_batch[t]
            test_results[f'{t}_rec'][clip_id] = {}
            if mask[f'{t}_rec']:
                test_results[f'{t}_rec'][clip_id]['stft'] = None
                test_results[f'{t}_rec'][clip_id]['mel'] = None
                test_results[f'{t}_rec'][clip_id]['sisdr'] = None
                if use_visqol:
                    test_results[f'{t}_rec'][clip_id]['visqol'] = None
            else:
                test_results[f'{t}_rec'][clip_id]['stft'] = metric_stft(est=est, ref=ref).item()
                test_results[f'{t}_rec'][clip_id]['mel'] = metric_mel(est=est, ref=ref).item()
                test_results[f'{t}_rec'][clip_id]['sisdr'] = - metric_sisdr(est=est, ref=ref).item()
                if use_visqol:
                    test_results[f'{t}_rec'][clip_id]['visqol'] = metric_visqol(est=est, ref=ref, sr=sample_rate)


test_results['summary'] = {}
for track in test_tracks:
    test_results['summary'][track] = {}
    list_stft = []
    list_mel = []
    list_sisdr = []
    if use_visqol:
        list_visqol = []

    for metrics in test_results[track].values():
        list_stft.append(metrics['stft'])
        list_mel.append(metrics['mel'])
        list_sisdr.append(metrics['sisdr'])
        if use_visqol:
            list_visqol.append(metrics['visqol'])

    np_stft = np.array([x for x in list_stft if x is not None])
    np_mel = np.array([x for x in list_mel if x is not None])
    np_sisdr = np.array([x for x in list_sisdr if x is not None])
    if use_visqol:
        np_visqol = np.array([x for x in list_visqol if x is not None])

    stft_m, stft_std = np.mean(np_stft), np.std(np_stft)
    mel_m, mel_std = np.mean(np_mel), np.std(np_mel)
    sisdr_m, sisdr_std = np.mean(np_sisdr), np.std(np_sisdr)
    if use_visqol:
        visqol_m, visqol_std = np.mean(np_visqol), np.std(np_visqol)

    print('='*80)
    print(f'{track}')
    print('Valid datapoint: {}/{}'.format(len(np_stft), len(list_stft)))
    print('Distance STFT: {:.2f} +/- {:.2f}'.format(stft_m, stft_std))
    print('Distance Mel: {:.2f} +/- {:.2f}'.format(mel_m, mel_std))
    print('SI-SDR: {:.2f} +/- {:.2f}'.format(sisdr_m, sisdr_std))
    if use_visqol:
        print('VisQOL: {:.2f} +/- {:.2f}'.format(visqol_m, visqol_std))

    test_results['summary'][track]['tot_seq'] = len(list_stft)
    test_results['summary'][track]['valid_seq'] = len(np_stft)
    test_results['summary'][track]['stft'] = {'mean': stft_m, 'std': stft_std}
    test_results['summary'][track]['mel'] = {'mean': mel_m, 'std': mel_std}
    test_results['summary'][track]['sisdr'] = {'mean': sisdr_m, 'std': sisdr_std}
    if use_visqol:
        test_results['summary'][track]['visqol'] = {'mean': visqol_m, 'std': visqol_std}


# save to json
json_filename = ret_dir / '{}_{}s.json'.format(csv_path.stem, length)
with open(json_filename, 'w') as f:
    json.dump(test_results, f, indent=1)