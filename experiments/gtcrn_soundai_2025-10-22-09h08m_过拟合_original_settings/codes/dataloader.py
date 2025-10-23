from random import random
import soundfile as sf
import librosa
import torch
from torch.utils import data
import numpy as np
import random
import os
os.path
# Resolve prepare_datasets path relative to this script for consistent usage

_script_dir = os.path.dirname(os.path.abspath(__file__))
TEST_VAL_DATABASE_TRAIN = os.path.join(_script_dir, 'prepare_datasets', 'soundai', 'train_noisy')
NOISY_DATABASE_TRAIN = TEST_VAL_DATABASE_TRAIN
NOISY_DATABASE_VALID = TEST_VAL_DATABASE_TRAIN
# NOISY_DATABASE_TRAIN = '/data/ssd0/xiaobin.rong/Datasets/DNS3/train_noisy'
# NOISY_DATABASE_VALID = '/data/ssd0/xiaobin.rong/Datasets/DNS3/dev_noisy'
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        fs=16000,
        length_in_seconds=8,
        num_data_tot=-1,
        num_data_per_epoch=10000,
        random_start_point=False,
        train=True
    ):
        if train:
            print("You are using this training data:", NOISY_DATABASE_TRAIN)
        else:
            print("You are using this validation data:", NOISY_DATABASE_VALID)
        self.noisy_database_train = sorted(librosa.util.find_files(NOISY_DATABASE_TRAIN, ext='wav'))[:num_data_tot]
        self.noisy_database_valid = sorted(librosa.util.find_files(NOISY_DATABASE_VALID, ext='wav'))
        self.L = int(length_in_seconds * fs)
        self.random_start_point = random_start_point
        self.fs = fs
        self.length_in_seconds = length_in_seconds
        self.num_data_per_epoch = num_data_per_epoch
        self.train = train
        
    def sample_data_per_epoch(self):
        self.noisy_data_train = random.sample(self.noisy_database_train, self.num_data_per_epoch)

    def __getitem__(self, idx):
        if self.train:
            noisy_list = self.noisy_data_train
        else:
            noisy_list = self.noisy_database_valid

        if self.random_start_point:
            Begin_S = int(np.random.uniform(0, 10 - self.length_in_seconds)) * self.fs
            noisy, _ = sf.read(noisy_list[idx], dtype='float32',start= Begin_S,stop = Begin_S + self.L)
            clean, _ = sf.read(noisy_list[idx].replace('noisy', 'clean'), dtype='float32',start=Begin_S, stop=Begin_S + self.L)

        else:
            noisy, _ = sf.read(noisy_list[idx], dtype='float32',start= 0, stop = self.L) 
            clean, _ = sf.read(noisy_list[idx].replace('noisy', 'clean'), dtype='float32', start=0, stop=self.L)

        return noisy, clean

    def __len__(self):
        if self.train:
            return self.num_data_per_epoch
        else:
            return len(self.noisy_database_valid)
        
class DNS3Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        fs=16000,
        length_in_seconds=8,
        num_data_tot=None,
        num_data_per_epoch=40000,
        random_start_point=False,
        train=True
    ):
        if train:
            print("You are using this DNS3 training data:", NOISY_DATABASE_TRAIN)
        else:
            print("You are using this DNS3 validation data:", NOISY_DATABASE_VALID)
        all_train_files = sorted(librosa.util.find_files(NOISY_DATABASE_TRAIN, ext='wav'))
        if num_data_tot is None:
            self.noisy_database_train = all_train_files
        else:
            self.noisy_database_train = all_train_files[:num_data_tot]
        self.noisy_database_valid = sorted(librosa.util.find_files(NOISY_DATABASE_VALID, ext='wav'))
        self.L = int(length_in_seconds * fs)
        self.random_start_point = random_start_point
        self.fs = fs
        self.length_in_seconds = length_in_seconds
        self.num_data_per_epoch = num_data_per_epoch
        self.train = train
        
    def sample_data_per_epoch(self):
        population = self.noisy_database_train
        if len(population) == 0:
            raise ValueError(f'No audio files found in {NOISY_DATABASE_TRAIN}')
        if len(population) >= self.num_data_per_epoch:
            # sample without replacement
            self.noisy_data_train = random.sample(population, self.num_data_per_epoch)
        else:
            # sample with replacement to avoid ValueError when population < required
            self.noisy_data_train = [random.choice(population) for _ in range(self.num_data_per_epoch)]

    def _safe_load(self, path, start, stop):
        """Robustly load audio segment [start:stop] (frames). Returns a 1-D float32 numpy array of length (stop-start).

        Tries soundfile, then librosa, and falls back to zeros if both fail. Pads or trims to exact length.
        """
        expected_len = int(stop - start)
        # check path exists
        if not os.path.exists(path):
            # try to be tolerant: if path contains 'noisy' try replacing with 'clean' and vice versa
            alt_path = None
            if 'noisy' in path:
                alt_path = path.replace('noisy', 'clean')
            elif 'clean' in path:
                alt_path = path.replace('clean', 'noisy')
            if alt_path and os.path.exists(alt_path):
                path = alt_path
            else:
                print(f"Warning: file not found: {path} (and no alternative). Returning zeros.", flush=True)
                return np.zeros(expected_len, dtype='float32')

        # try soundfile first
        try:
            data, sr = sf.read(path, dtype='float32', start=int(start), stop=int(stop))
            # ensure mono
            if data.ndim > 1:
                data = data[:, 0]
            # pad or trim
            if len(data) < expected_len:
                pad = np.zeros(expected_len - len(data), dtype='float32')
                data = np.concatenate([data, pad])
            elif len(data) > expected_len:
                data = data[:expected_len]
            return data.astype('float32')
        except Exception as e:
            # try librosa as fallback with offset/duration
            try:
                offset = float(start) / float(self.fs)
                duration = float(expected_len) / float(self.fs)
                y, sr = librosa.load(path, sr=self.fs, mono=True, offset=offset, duration=duration)
                if len(y) < expected_len:
                    pad = np.zeros(expected_len - len(y), dtype='float32')
                    y = np.concatenate([y, pad])
                elif len(y) > expected_len:
                    y = y[:expected_len]
                return y.astype('float32')
            except Exception as e2:
                print(f"Warning: failed to load {path} with soundfile ({e}) and librosa ({e2}). Returning zeros.", flush=True)
                return np.zeros(expected_len, dtype='float32')

    def __getitem__(self, idx):
        if self.train:
            noisy_list = self.noisy_data_train
        else:
            noisy_list = self.noisy_database_valid

        if self.random_start_point:
            Begin_S = int(np.random.uniform(0, 10 - self.length_in_seconds)) * self.fs
            noisy = self._safe_load(noisy_list[idx], start=Begin_S, stop=Begin_S + self.L)
            clean_path = noisy_list[idx].replace('noisy', 'clean')
            clean = self._safe_load(clean_path, start=Begin_S, stop=Begin_S + self.L)

        else:
            noisy = self._safe_load(noisy_list[idx], start=0, stop=self.L)
            clean_path = noisy_list[idx].replace('noisy', 'clean')
            clean = self._safe_load(clean_path, start=0, stop=self.L)

        return noisy, clean

    def __len__(self):
        if self.train:
            return self.num_data_per_epoch
        else:
            return len(self.noisy_database_valid)


if __name__=='__main__':
    from tqdm import tqdm 
    from omegaconf import OmegaConf
    
    config = OmegaConf.load('configs/cfg_train.yaml')

        
    train_dataset = DNS3Dataset(**config['train_dataset'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])
    train_dataloader.dataset.sample_data_per_epoch()

    validation_dataset = DNS3Dataset(**config['validation_dataset'])
    validation_dataloader = data.DataLoader(validation_dataset, **config['validation_dataloader'])

    print(len(train_dataloader), len(validation_dataloader))

    for noisy, clean in tqdm(train_dataloader):
        print(noisy.shape, clean.shape)
        break
        # pass

    for noisy, clean in tqdm(validation_dataloader):
        print(noisy.shape, clean.shape)
        break
        # pass
