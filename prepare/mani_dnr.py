"""
Copyright (c) 2024 by Telecom-Paris
Authoried by Xiaoyu BIE (xiaoyu.bie@telecom-paris.fr)
License agreement in LICENSE.txt
"""
import os
import argparse
import librosa
import torchaudio
from tqdm import tqdm
from pathlib import Path
from collections import namedtuple
import torch

parser = argparse.ArgumentParser(description='Generate manifest for audio dataset',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data-dir', type=str, default='/home/xbie/Data/dnr_v2', help='Audio Dataset Path')
parser.add_argument('--out-dir', type=str, default='./manifest', help='Path to write manifest')
parser.add_argument('--ext', type=str, default='wav', choices=['wav', 'mp3', 'flac'], help='Audio format')

args = parser.parse_args()
data_dir = Path(args.data_dir)
out_dir = Path(args.out_dir)
ext = args.ext

train_dir = data_dir / 'tr' 
val_dir = data_dir / 'cv'
test_dir = data_dir / 'tt'
tracks = ['speech', 'music', 'sfx']

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

## train
for track in tracks:
    with open(out_dir / f'{track}_dnr.csv', 'w') as f:
        f.write('id,filepath,sr,length,start,end\n')
        for subdir in tqdm(sorted(train_dir.iterdir()), desc=f'train_{track}'):
            if subdir.is_dir():
                audio_id = subdir.name
                audio_filepath = subdir / f'{track}.{ext}'
                x, sr = torchaudio.load(audio_filepath)
                length = x.shape[-1]
                _, (trim30dBs,trim30dBe) = librosa.effects.trim(x.numpy(), top_db=30)
                line = '{},{},{},{},{},{}\n'.format(audio_id, audio_filepath, sr, length, trim30dBs, trim30dBe)
                f.write(line)


## val with no silence
length = 5
threshold = 0.5
with open(out_dir / 'val_dnr.csv', 'w') as f:
    f.write('id,mix,speech,music,sfx,sr,length,start,end\n')
    for subdir in tqdm(sorted(val_dir.iterdir()), desc='val'):
        if subdir.is_dir():
            audio_id = subdir.name
            mix_filepath = subdir / f'mix.{ext}'
            speech_filepath = subdir / f'speech.{ext}'
            music_filepath = subdir / f'music.{ext}'
            sfx_filepath = subdir / f'sfx.{ext}'

            x_mix, sr = torchaudio.load(mix_filepath)
            x_speech, _ = torchaudio.load(speech_filepath)
            x_music, _ = torchaudio.load(music_filepath)
            x_sfx, _ = torchaudio.load(sfx_filepath)

            chunk_len = sr * length
            _, (trim30dBs,trim30dBe) = librosa.effects.trim(x_mix.numpy(), top_db=30)
            for j, k in enumerate(range(trim30dBs, trim30dBe-chunk_len, chunk_len)):
                chunk_mix = x_mix[..., k:k+chunk_len]
                chunk_speech = x_speech[..., k:k+chunk_len]
                chunk_music = x_music[..., k:k+chunk_len]
                chunk_sfx = x_sfx[..., k:k+chunk_len]
            
                # detect silence length
                is_active = True
                for audio_clip in [chunk_mix, chunk_speech, chunk_music, chunk_sfx]:
                    audio_clip = audio_clip.reshape(-1)
                    audio_energy = torch.stft(audio_clip, n_fft=stft_params.window_length, 
                                            hop_length=stft_params.hop_length, win_length=stft_params.window_length,
                                            window=torch.hann_window(stft_params.window_length, device='cpu'),
                                            pad_mode=stft_params.padding_type, center=True, onesided=True, return_complex=True
                                    ).abs().sum(dim=0)
                    count = sum(1 for item in audio_energy if item > 1e-6)
                    if count < threshold * len(audio_energy):
                        is_active = False
                
                # save if no silence detect
                if is_active:
                    clip_id = f'{audio_id}_{j}'
                    start = k
                    end = k + chunk_len
                    line = '{},{},{},{},{},{},{},{},{}\n'.format(clip_id, mix_filepath, speech_filepath, music_filepath, sfx_filepath, \
                                                            sr, chunk_len, start, end)
                    f.write(line)

## test with no silence
length = 10
threshold = 0.5
with open(out_dir / 'test_dnr.csv', 'w') as f:
    f.write('id,mix,speech,music,sfx,sr,length,start,end\n')
    for subdir in tqdm(sorted(val_dir.iterdir()), desc='test'):
        if subdir.is_dir():
            audio_id = subdir.name
            mix_filepath = subdir / f'mix.{ext}'
            speech_filepath = subdir / f'speech.{ext}'
            music_filepath = subdir / f'music.{ext}'
            sfx_filepath = subdir / f'sfx.{ext}'

            x_mix, sr = torchaudio.load(mix_filepath)
            x_speech, _ = torchaudio.load(speech_filepath)
            x_music, _ = torchaudio.load(music_filepath)
            x_sfx, _ = torchaudio.load(sfx_filepath)

            chunk_len = sr * length
            _, (trim30dBs,trim30dBe) = librosa.effects.trim(x_mix.numpy(), top_db=30)
            for j, k in enumerate(range(trim30dBs, trim30dBe-chunk_len, chunk_len)):
                chunk_mix = x_mix[..., k:k+chunk_len]
                chunk_speech = x_speech[..., k:k+chunk_len]
                chunk_music = x_music[..., k:k+chunk_len]
                chunk_sfx = x_sfx[..., k:k+chunk_len]
            
                # detect silence length
                is_active = True
                for audio_clip in [chunk_mix, chunk_speech, chunk_music, chunk_sfx]:
                    audio_clip = audio_clip.reshape(-1)
                    audio_energy = torch.stft(audio_clip, n_fft=stft_params.window_length, 
                                            hop_length=stft_params.hop_length, win_length=stft_params.window_length,
                                            window=torch.hann_window(stft_params.window_length, device='cpu'),
                                            pad_mode=stft_params.padding_type, center=True, onesided=True, return_complex=True
                                    ).abs().sum(dim=0)
                    count = sum(1 for item in audio_energy if item > 1e-6)
                    if count < threshold * len(audio_energy):
                        is_active = False
                
                # save if no silence detect
                if is_active:
                    clip_id = f'{audio_id}_{j}'
                    start = k
                    end = k + chunk_len
                    line = '{},{},{},{},{},{},{},{},{}\n'.format(clip_id, mix_filepath, speech_filepath, music_filepath, sfx_filepath, \
                                                            sr, chunk_len, start, end)
                    f.write(line)