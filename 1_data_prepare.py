## Author: Beiming Cao
import time
import yaml
import os
import torch
from utils.IO_func import read_file_list
from shutil import copyfile
from utils.transforms import Transform_Compose, Pair_Transform_Compose
from utils.transforms import pair_change_wav_sampling_rate, apply_delta_deltadelta_Src, apply_delta_deltadelta_Tar, apply_delta_deltadelta_Src_Tar, pair_wav2melspec, apply_DTW
from scipy.io.wavfile import read
from database import CMU_ARCTIC_VC

def data_processing(args):

    '''
    Load in data from all speakers involved, apply feature extraction, 
    save them into .pt files in the current_exp folder.

    '''

    config_path = args.conf_dir       
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    out_folder = os.path.join(args.buff_dir, 'data')
    
    transforms = [] # default transforms list, setup all the transforms before feeding data in

    #### CMU_arctic sampling rate is 16000, but to match the Waveglow vocoder, change it to 22050 here.
    #### 16000 works to but needs to change other feature setups, and quality might be lower

    fs_change = change_wav_sampling_rate(16000, 22050)
    transform.append(fs_change)
    #####################################################

    data_path = config['corpus']['path']
    fileset_path = os.path.join(data_path, 'filesets')
    src_spk_list = config['data_setup']['src_spk_list']
    tar_spk_list = config['data_setup']['tar_spk_list']

    #### Apply delta to input or/and output acoustic features

  #  input_delta = config['training_setup']['delta']['input']
  #  output_delta = config['training_setup']['delta']['output']

  #  if input_delta == True:
  #      transform.append(apply_delta_deltadelta_Src)
  #  if output_delta == True:
  #      transform.append(apply_delta_deltadelta_Tar)

    ############### Apply DTW alignment ######
  #  transforms.append(apply_DTW())


    ################ Acoustic feature extraction #################
    sampling_rate = config['acoustic_feature']['sampling_rate']
    filter_length = config['acoustic_feature']['filter_length']
    hop_length = config['acoustic_feature']['hop_length']
    win_length = config['acoustic_feature']['win_length']
    n_mel_channels = config['acoustic_feature']['n_mel_channels']
    mel_fmin = config['acoustic_feature']['mel_fmin']
    mel_fmax = config['acoustic_feature']['mel_fmax']

    transforms.append(wav2melspec(sampling_rate, filter_length, hop_length, win_length, 
                 n_mel_channels, mel_fmin, mel_fmax))


    ############### Compose all the transforms #####
    transforms_all = Transform_Compose(transforms)

    ############### Load in wav from all speakers involved ###

    all_SPK = list(set(src_spk_list, tar_spk_list))
    
    for SPK in all_list:
        out_folder_SPK = os.path.join(out_folder, SPK)
        if not os.path.exists(out_folder_SPK):
            os.makedirs(out_folder_SPK)

        fileset_path_SPK = os.path.join(fileset_path, SPK)
        print(fileset_path_SPK)

        for file_id in file_id_list:
            data_path_spk = os.path.join(data_path, file_id[:3])
            mat_path = os.path.join(data_path_spk, 'data/'+ file_id + '.mat')
            EMA, WAV, fs_ema, fs_wav = load_Haskins_ATS_data(mat_path, file_id, sel_sensors, sel_dim)
            EMA, WAV = transforms_all(EMA, WAV) 

            WAV_out_dir = os.path.join(out_folder_SPK, file_id + '.pt')

            torch.save(WAV, WAV_out_dir)
 
            

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/VC_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')

    args = parser.parse_args()
    data_processing(args)
