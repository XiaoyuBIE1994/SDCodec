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

parser = argparse.ArgumentParser(description='Generate manifest for audio dataset, MTG-Jamendo',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data-dir', type=str, default='/home/xbie/Data/mtg-jamendo/', help='Audio Dataset Path')
parser.add_argument('--partition', type=int, default=-1, help='Audio partition to create manifest')
parser.add_argument('--out-dir', type=str, default='./manifest', help='Path to write manifest')
parser.add_argument('--threshold', type=float, default=0.5, help='Remove audio files that are too short')
parser.add_argument('--ext', type=str, default='mp3', choices=['wav', 'mp3', 'wav'], help='Audio format')

args = parser.parse_args()
data_dir = Path(args.data_dir)
out_dir = Path(args.out_dir)
partition = args.partition
threshold = args.threshold
ext = args.ext

if partition == -1:
    audio_sources = list(range(100))
    audio_manifest = out_dir / f'music_jamendo.csv'
else:
    audio_sources = list(range(10*partition, 10*partition+10))
    audio_manifest = out_dir / f'music_jamendo_{partition}.csv'

with open(audio_manifest, 'w') as f:
    f.write('id,filepath,sr,length,start,end\n')
    audio_len = 0
    for audio_source in audio_sources:
        audio_dir = data_dir / f"{audio_source:0>2}"
        for audio_filepath in tqdm(list(audio_dir.glob(f'**/*.{ext}')), desc=f"jamendo_{audio_source:0>2}"):
            audio_id = audio_filepath.stem
            x, sr = torchaudio.load(audio_filepath)
            length = x.shape[-1]
            _, (trim30dBs,trim30dBe) = librosa.effects.trim(x.numpy(), top_db=30)
            utt_len = (trim30dBe - trim30dBs) / sr
            if utt_len >= threshold:
                audio_len += length / sr
                line = '{},{},{},{},{},{}\n'.format(audio_id, audio_filepath, sr, length, trim30dBs, trim30dBe)
                f.write(line)
    print('Source: music. audio len: {:.2f}h'.format(audio_len/3600))

