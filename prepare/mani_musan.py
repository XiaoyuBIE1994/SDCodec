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

parser = argparse.ArgumentParser(description='Generate manifest for audio dataset, DNS-challenge-5',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data-dir', type=str, default='/home/xbie/Data/Libri-light/', help='Audio Dataset Path')
parser.add_argument('--out-dir', type=str, default='./manifest', help='Path to write manifest')
parser.add_argument('--threshold', type=float, default=0.5, help='Remove audio files that are too short')
parser.add_argument('--ext', type=str, default='wav', choices=['wav', 'mp3', 'flac'], help='Audio format')
args = parser.parse_args()
data_dir = Path(args.data_dir)
out_dir = Path(args.out_dir)
threshold = args.threshold
ext = args.ext

# speech
audio_dir = data_dir / 'speech'
audio_manifest = out_dir / f'speech_musan.csv'
audio_len = 0
with open(audio_manifest, 'w') as f:
    f.write('id,filepath,sr,length,start,end\n')
    for audio_filepath in tqdm(list(audio_dir.glob(f'**/*.{ext}')), desc=f'speech'):
        audio_id = audio_filepath.stem
        x, sr = torchaudio.load(audio_filepath)
        length = x.shape[-1]
        _, (trim30dBs,trim30dBe) = librosa.effects.trim(x.numpy(), top_db=30)
        utt_len = (trim30dBe - trim30dBs) / sr
        if utt_len >= threshold:
            audio_len += length / sr
            line = '{},{},{},{},{},{}\n'.format(audio_id, audio_filepath, sr, length, trim30dBs, trim30dBe)
            f.write(line)
    print('Source: noise. audio len: {:.2f}h'.format(audio_len/3600))

# music
audio_dir = data_dir / 'music'
audio_manifest = out_dir / f'music_musan.csv'
audio_len = 0
with open(audio_manifest, 'w') as f:
    f.write('id,filepath,sr,length,start,end\n')
    for audio_filepath in tqdm(list(audio_dir.glob(f'**/*.{ext}')), desc=f'music'):
        audio_id = audio_filepath.stem
        x, sr = torchaudio.load(audio_filepath)
        length = x.shape[-1]
        _, (trim30dBs,trim30dBe) = librosa.effects.trim(x.numpy(), top_db=30)
        utt_len = (trim30dBe - trim30dBs) / sr
        if utt_len >= threshold:
            audio_len += length / sr
            line = '{},{},{},{},{},{}\n'.format(audio_id, audio_filepath, sr, length, trim30dBs, trim30dBe)
            f.write(line)
    print('Source: noise. audio len: {:.2f}h'.format(audio_len/3600))

# noise
audio_dir = data_dir / 'noise'
audio_manifest = out_dir / f'sfx_musan.csv'
audio_len = 0
with open(audio_manifest, 'w') as f:
    f.write('id,filepath,sr,length,start,end\n')
    for audio_filepath in tqdm(list(audio_dir.glob(f'**/*.{ext}')), desc=f'noise'):
        audio_id = audio_filepath.stem
        x, sr = torchaudio.load(audio_filepath)
        length = x.shape[-1]
        _, (trim30dBs,trim30dBe) = librosa.effects.trim(x.numpy(), top_db=30)
        utt_len = (trim30dBe - trim30dBs) / sr
        if utt_len >= threshold:
            audio_len += length / sr
            line = '{},{},{},{},{},{}\n'.format(audio_id, audio_filepath, sr, length, trim30dBs, trim30dBe)
            f.write(line)
    print('Source: noise. audio len: {:.2f}h'.format(audio_len/3600))