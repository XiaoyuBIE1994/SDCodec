"""
Copyright (c) 2024 by Telecom-Paris
Authoried by Xiaoyu BIE (xiaoyu.bie@telecom-paris.fr)
License agreement in LICENSE.txt
"""

import math
import numpy as np
import pandas as pd
import julius
from pathlib import Path
from omegaconf import DictConfig
from typing import List

import torch
import torchaudio
from torch.utils.data import Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__)



class DatasetAudioTrain(Dataset):
    def __init__(self,
        sample_rate: int,
        speech: List[str],
        music: List[str],
        sfx: List[str],
        n_examples: int = 10000000,
        chunk_size: float = 2.0,
        trim_silence: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        # init
        self.EPS = 1e-8
        self.sample_rate = sample_rate # target sampling rate
        self.length = n_examples # pseudo dataset length
        self.chunk_size = chunk_size # negative for entire sentence
        self.trim_silence = trim_silence
        
        # manifest
        self.csv_files = {}
        self.csv_files['speech'] = [Path(filepath) for filepath in speech]
        self.csv_files['music'] = [Path(filepath) for filepath in music]
        self.csv_files['sfx'] = [Path(filepath) for filepath in sfx]

        # check valid samples
        self.resample_pool = dict()
        self.metadata_dict = dict()
        self.lens_dict = dict()
        for track, files in self.csv_files.items():
            logger.info(f"Track: {track}")
            orig_utt, orig_len, drop_utt, drop_len = 0, 0, 0, 0
            metadata_list = []
            for tsv_filepath in files:
                if not tsv_filepath.is_file():
                    logger.error('No tsv file found in: {}'.format(tsv_filepath))
                    continue
                else:
                    logger.info(f'Manifest filepath: {tsv_filepath}')
                    metadata = pd.read_csv(tsv_filepath)
                    if self.trim_silence:
                        wav_lens = (metadata['end'] - metadata['start']) / metadata['sr']
                    else:
                        wav_lens = metadata['length'] / metadata['sr']
                    # remove wav files that too short
                    orig_utt += len(metadata)
                    drop_rows = []
                    for row_idx in range(len(wav_lens)):
                        orig_len += wav_lens[row_idx]
                        if wav_lens[row_idx] < self.chunk_size:
                            drop_rows.append(row_idx)
                            drop_utt += 1
                            drop_len += wav_lens[row_idx]
                        else:
                            # prepare julius resample
                            sr = int(metadata.at[row_idx, 'sr'])
                            if sr not in self.resample_pool.keys():
                                old_sr = sr
                                new_sr = self.sample_rate
                                gcd = math.gcd(old_sr, new_sr)
                                old_sr = old_sr // gcd
                                new_sr = new_sr // gcd
                                self.resample_pool[sr] = julius.ResampleFrac(old_sr=old_sr, new_sr=new_sr)

                    metadata = metadata.drop(drop_rows)
                    metadata_list.append(metadata)

            self.metadata_dict[track] = pd.concat(metadata_list, axis=0)
            self.lens_dict[track] = len(self.metadata_dict[track])
            
            logger.info("Drop {}/{} utterances ({:.2f}/{:.2f}h), shorter than {:.2f}s".format(
                drop_utt, orig_utt, drop_len / 3600, orig_len / 3600, self.chunk_size
            ))
            logger.info('Used data: {} utterances, ({:.2f} h)'.format(
                self.lens_dict[track], (orig_len-drop_len) / 3600
            ))

        logger.info('Resample pool: {}'.format(list(self.resample_pool.keys())))


    def __len__(self):
        return self.length # can be any number


    def __getitem__(self, idx:int):

        batch = {}
        for track in self.csv_files.keys():
            idx = np.random.randint(self.lens_dict[track])
            wav_info = self.metadata_dict[track].iloc[idx]
            chunk_len = int(wav_info['sr'] * self.chunk_size)

            # slice wav files
            if self.trim_silence: 
                start = np.random.randint(int(wav_info['start']), int(wav_info['end']) - chunk_len + 1)
            else:
                start = np.random.randint(0, int(wav_info['length']) - chunk_len + 1)

            # load file
            x, sr = torchaudio.load(wav_info['filepath'],
                                    frame_offset=start,
                                    num_frames=chunk_len)

            # single channel
            x = x.mean(dim=0, keepdim=True)

            # resample
            if sr != self.sample_rate:
                x = self.resample_pool[sr](x)

            batch[track] = x

        return batch



class DatasetAudioVal(Dataset):
    def __init__(self,
        sample_rate: int,
        tsv_filepath: str,
        chunk_size: float = 5.0,
        **kwargs
    ) -> None:
        super().__init__()

        # init
        self.EPS = 1e-8
        self.sample_rate = sample_rate # target sampling rate
        self.tsv_filepath = Path(tsv_filepath)
        self.chunk_size = chunk_size # negative for entire sentence
        self.resample_pool = dict()

        # read manifest tsv file
        if self.tsv_filepath.is_file():
            metadata = pd.read_csv(self.tsv_filepath)
            logger.info(f'Manifest filepath: {self.tsv_filepath}')
        else:
            logger.error('No tsv file found in: {}'.format(self.tsv_filepath))

        # audio lengths
        wav_lens = (metadata['end'] - metadata['start']) / metadata['sr']

        # remove wav files that too short
        orig_utt = len(metadata)
        orig_len, drop_utt, drop_len = 0, 0, 0
        drop_rows = []
        for row_idx in range(len(wav_lens)):
            orig_len += wav_lens[row_idx]
            if wav_lens[row_idx] < self.chunk_size:
                drop_rows.append(row_idx)
                drop_utt += 1
                drop_len += wav_lens[row_idx]
            else:
                # prepare julius resample
                sr = int(metadata.at[row_idx, 'sr'])
                if sr not in self.resample_pool.keys():
                    old_sr = sr
                    new_sr = self.sample_rate
                    gcd = math.gcd(old_sr, new_sr)
                    old_sr = old_sr // gcd
                    new_sr = new_sr // gcd
                    self.resample_pool[sr] = julius.ResampleFrac(old_sr=old_sr, new_sr=new_sr)

        logger.info("Drop {}/{} utts ({:.2f}/{:.2f}h), shorter than {:.2f}s".format(
            drop_utt, orig_utt, drop_len / 3600, orig_len / 3600, self.chunk_size
        ))
        logger.info('Actual data size: {} utterance, ({:.2f} h)'.format(
            orig_utt-drop_utt, (orig_len-drop_len) / 3600
        ))
        logger.info('Resample pool: {}'.format(list(self.resample_pool.keys())))

        self.metadata = metadata.drop(drop_rows)


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx:int):

        wav_info = self.metadata.iloc[idx]
        chunk_len = int(wav_info['sr'] * self.chunk_size)
        start = wav_info['start']
        
        # Load wav files and resample if needed
        batch = {}
        for track in ['mix', 'speech', 'music', 'sfx']:
            x, sr = torchaudio.load(wav_info[track],
                                    frame_offset=start,
                                    num_frames=chunk_len)
            x = x.mean(dim=0, keepdim=True)
            # resample
            if sr != self.sample_rate:
                x = self.resample_pool[sr](x)
            batch[track] = x
        
        return batch
    


if __name__ == '__main__':
    import time
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from omegaconf import OmegaConf
    accelerator = Accelerator()
    
    # train
    cfg = OmegaConf.create()
    cfg.sample_rate = 16000
    cfg.speech = [
        './manifest/speech_dnr.csv'
    ]
    cfg.music = [
        './manifest/music_dnr.csv'
    ]
    cfg.sfx = [
        './manifest/sfx_dnr.csv'
    ]
    cfg.general = [
        './manifest/mix_dnr.csv'
    ]
    cfg.chunk_size = 2.0
    cfg.trim_silence = False
    train_dataset = DatasetAudioTrain(**cfg)

    print('Train data: {}'.format(len(train_dataset)))

    idx = np.random.randint(train_dataset.__len__())
    data_ = train_dataset.__getitem__(idx)
    for k, v in data_.items():
        print('audio idx: {} audio: {}, length: {}'.format(idx, k, v.shape))

    # val
    cfg = OmegaConf.create()
    cfg.sample_rate = 16000
    cfg.tsv_filepath = './manifest/val.csv'
    cfg.chunk_size = 5.0
    val_dataset = DatasetAudioVal(**cfg)

    print('Validation data: {}'.format(len(val_dataset)))
    idx = np.random.randint(val_dataset.__len__())
    data_ = val_dataset.__getitem__(idx)
    for k, v in data_.items():
        print('audio idx: {} audio: {}, length: {}'.format(idx, k, v.shape))

    # test
    cfg = OmegaConf.create()
    cfg.sample_rate = 16000
    cfg.tsv_filepath = './manifest/test.csv'
    cfg.chunk_size = 10.0
    test_dataset = DatasetAudioVal(**cfg)

    print('Test data: {}'.format(len(test_dataset)))
    idx = np.random.randint(test_dataset.__len__())
    data_ = test_dataset.__getitem__(idx)
    for k, v in data_.items():
        print('audio idx: {} audio: {}, length: {}'.format(idx, k, v.shape))

    # dataloader
    train_dataset.length = 10000
    train_loader = DataLoader(dataset=train_dataset, 
                                batch_size=32, num_workers=8,
                                shuffle=True, drop_last=True)
    
    total_seq = 0
    start_time = time.time()
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        mix_audio = batch['speech']
        total_seq += mix_audio.shape[0]
        # print(mix_audio.shape)
        # breakpoint()

    elapsed_time = time.time() - start_time
    tpf = elapsed_time / total_seq
    tpb = elapsed_time / (i+1)

    print(f"Read pure data time cost {tpf:.3f}s per file")
    print(f"Read pure data time cost {tpb:.3f}s per batch")

