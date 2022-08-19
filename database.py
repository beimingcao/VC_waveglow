## Author: Beiming Cao
import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import librosa


class CMU_ARCTIC_VC(Dataset):
    def __init__(self, data_path, spk_list, src_spk, tar_spk, transforms=None):

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

    def compute_mean_std(self):
        idx = 0
        for src, tar in self.data:                        
            if idx == 0:
                src_all, tar_all = src, tar
            else:
                src_all, tar_all = torch.cat((src_all, src), axis = 0), torch.cat((tar_all, tar), axis = 0)
            idx += 1                        
        src_mean, src_std = torch.std_mean(src_all, axis = 0)
        tar_mean, tar_std = torch.std_mean(tar_all, axis = 0)
        return src_mean, src_std, tar_mean, tar_std 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        src, tar = self.data[idx]     
        if self.transforms is not None:
            src, tar = self.transforms(src, tar)         
        return (src, tar)





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/VC_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')

    

    args = parser.parse_args()
    data_processing(args)
