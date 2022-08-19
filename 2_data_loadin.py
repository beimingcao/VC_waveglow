# Author: Beiming Cao

import time
import yaml
import os
import glob
import torch
import pickle
from utils.database import CMU_ARCTIC_VC
from torch.utils.data import Dataset, DataLoader
from utils.transforms import Pair_Transform_Compose

from shutil import copyfile
from utils.transforms import apply_DTW, apply_src_MVN, apply_tar_MVN, apply_delta_deltadelta_Src_Tar
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
Further separate data into src-tar pairs, apply z-scores, DTW, train test split
'''

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/VC_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')
    args = parser.parse_args()
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)

    pair_transforms = [] # transforms applied to src-tar pairs

    ############### Apply DTW alignment ######
    pair_transforms.append(apply_DTW())


    data_path = config['corpus']['path']
    src_spk_list = config['data_setup']['source_spk_list']
    tar_spk_list = config['data_setup']['target_spk_list']
 
    train_val_test_ratio = config['training_setup']['train_val_test_ratio']  # [train, val, test]

    input_norm = config['training_setup']['normalization']['input']
    output_norm = config['training_setup']['normalization']['output']

    pt_data_path = os.path.join(args.buff_dir, 'data')

    for tar_spk in tar_spk_list:
        src_spk_unique = src_spk_list.copy()
        src_spk_unique.remove(tar_spk)
        for src_spk in src_spk_unique:
            sub_exp_id = src_spk + '_vc_' + tar_spk
            sub_exp_folder = os.path.join(args.buff_dir, sub_exp_id)
            if not os.path.exists(sub_exp_folder):
                os.makedirs(sub_exp_folder)

            src_pt_folder = os.path.join(pt_data_path, src_spk)
            tar_pt_folder = os.path.join(pt_data_path, tar_spk)

            src_pt_list, tar_pt_list = glob.glob(src_pt_folder + '/*.pt'), glob.glob(tar_pt_folder + '/*.pt')
            src_id_list = [os.path.basename(x)[:-3] for x in src_pt_list]
            tar_id_list = [os.path.basename(x)[:-3] for x in tar_pt_list]

            #### find common parallel samples from src and tar speakers
            
            common_id_list = []
            for s_id in src_id_list:
                if s_id in tar_id_list:
                    common_id_list.append(s_id)
            print('Speaker ' + src_spk + ' and ' + tar_spk + ' has ' + str(len(common_id_list)) + ' parallel samples')

            sample_num = len(common_id_list)

            train_id_list = common_id_list[0:int(sample_num*train_val_test_ratio[0])]
            valid_id_list = common_id_list[int(sample_num*train_val_test_ratio[0]):int(sample_num*(train_val_test_ratio[0]+train_val_test_ratio[1]))]
            test_id_list = common_id_list[int(sample_num*(train_val_test_ratio[0]+train_val_test_ratio[1])):sample_num]
  
            train_transforms = pair_transforms
            valid_transforms = pair_transforms
            test_transforms = pair_transforms

            ### apply_normalization

            if input_norm == True or output_norm == True:
                train_dataset = CMU_ARCTIC_VC(pt_data_path, train_id_list, src_spk, tar_spk)
                src_mean, src_std, tar_mean, tar_std  = train_dataset.compute_mean_std()

                if input_norm == True:              
                    train_transforms.append(apply_src_MVN(src_mean, src_std))
                    valid_transforms.append(apply_src_MVN(src_mean, src_std))
                    test_transforms.append(apply_src_MVN(src_mean, src_std))
                if output_norm == True:
                    train_transforms.append(apply_tar_MVN(tar_mean, tar_std))
                    valid_transforms.append(apply_tar_MVN(tar_mean, tar_std))
                    torch.save(tar_mean, os.path.join(sub_exp_folder, 'tar_mean.pt'))
                    torch.save(tar_std, os.path.join(sub_exp_folder, 'tar_std.pt'))

            train_dataset = CMU_ARCTIC_VC(pt_data_path, train_id_list, src_spk, tar_spk, transforms=Pair_Transform_Compose(train_transforms))
            valid_dataset = CMU_ARCTIC_VC(pt_data_path, valid_id_list, src_spk, tar_spk, transforms=Pair_Transform_Compose(valid_transforms))
            test_dataset = CMU_ARCTIC_VC(pt_data_path, test_id_list, src_spk, tar_spk, transforms=Pair_Transform_Compose(test_transforms))    

            train_pkl_path = os.path.join(sub_exp_folder, 'train_data.pkl')
            tr = open(train_pkl_path, 'wb')
            pickle.dump(train_dataset, tr)
            valid_pkl_path = os.path.join(sub_exp_folder, 'valid_data.pkl')
            va = open(valid_pkl_path, 'wb')
            pickle.dump(valid_dataset, va)
            test_pkl_path = os.path.join(sub_exp_folder, 'test_data.pkl')
            te = open(test_pkl_path, 'wb')
            pickle.dump(test_dataset, te)
