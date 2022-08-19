# Author: Beiming Cao

import time
import yaml
import os
import torch
import pickle
from database import HaskinsData_ATS
from torch.utils.data import Dataset, DataLoader
from utils.transforms import Pair_Transform_Compose
from utils.IO_func import read_file_list, load_binary_file, array_to_binary_file, load_Haskins_ATS_data
from utils.utils import prepare_Haskins_lists
from shutil import copyfile
from utils.transforms import padding_end, apply_EMA_MVN, zero_padding_end
import random

### Fix the randomness for reproduction

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

'''
Further separate data into src-tar pairs, apply z-scores
'''

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/VC_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')
    args = parser.parse_args()
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)

    data_path = config['corpus']['path']
    src_spk_list = config['data_setup']['source_spk_list']
    tar_spk_list = config['data_setup']['target_spk_list']

    train_transforms = []
    valid_transforms = []
    test_transforms = []

    exp_train_lists, exp_valid_lists, exp_test_lists = prepare_Haskins_lists(args)

    for i in range(len(exp_test_lists)):
  #      CV = 'CV' + format(i, '02d')
        CV = exp_test_lists[i][0][:3]
        CV_data_dir = os.path.join(prepared_data_CV_path, CV)
        if not os.path.exists(CV_data_dir):
            os.makedirs(CV_data_dir)

        train_list = exp_train_lists[i]
        valid_list = exp_valid_lists[i]
        test_list = exp_test_lists[i]

        train_dataset = HaskinsData_ATS(prepared_data_path, train_list, ema_dim)
        if MVN == True:
            EMA_mean, EMA_std = train_dataset.compute_ema_mean_std()
            train_transforms.append(apply_EMA_MVN(EMA_mean, EMA_std))
            valid_transforms.append(apply_EMA_MVN(EMA_mean, EMA_std))
            test_transforms.append(apply_EMA_MVN(EMA_mean, EMA_std))

        if batch_size > 1:
            valid_dataset = HaskinsData_ATS(prepared_data_path, valid_list, ema_dim)
  #          max_len = max(train_dataset.find_max_len(), valid_dataset.find_max_len())
            max_len = 340
  #          train_transforms.append(padding_end(max_len))
  #          valid_transforms.append(padding_end(max_len))    
            train_transforms.append(zero_padding_end(max_len))
            valid_transforms.append(zero_padding_end(max_len))

        train_dataset = HaskinsData_ATS(prepared_data_path, train_list, ema_dim, transforms = Pair_Transform_Compose(train_transforms))
        valid_dataset = HaskinsData_ATS(prepared_data_path, valid_list, ema_dim, transforms = Pair_Transform_Compose(valid_transforms))
        test_dataset = HaskinsData_ATS(prepared_data_path, test_list, ema_dim, transforms = Pair_Transform_Compose(test_transforms))

        train_pkl_path = os.path.join(CV_data_dir, 'train_data.pkl')
        tr = open(train_pkl_path, 'wb')
        pickle.dump(train_dataset, tr)
        valid_pkl_path = os.path.join(CV_data_dir, 'valid_data.pkl')
        va = open(valid_pkl_path, 'wb')
        pickle.dump(valid_dataset, va)
        test_pkl_path = os.path.join(CV_data_dir, 'test_data.pkl')
        te = open(test_pkl_path, 'wb')
        pickle.dump(test_dataset, te)
