## Author: Beiming Cao
import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import librosa


class CMU_ARCTIC_VC(Dataset):
    def __init__(self, data_path, id_list, src_spk, tar_spk, transforms=None):

        '''
        data_path: path of 'cmu_arctic' folder
        id_list:   file_ids for this dataset, e.g., ['arctic_a0001', 'arctic_a0002']
        src_spk: source speaker, e.g., 'aew'
        tar_spk: target speaker, e.g., 'ahw'
        transform: a list of composed function to transform the data, e.g. feature extraction, change_sampling_rate
        
        '''
        self.data_path = data_path
        self.src_spk = src_spk
        self.tar_spk = tar_spk
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        file_id, EMA, WAV = self.data[idx]     
        if self.transforms is not None:
            EMA, WAV = self.transforms(EMA, WAV)         
        return (file_id, EMA, WAV)





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/VC_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')

    

    args = parser.parse_args()
    data_processing(args)
