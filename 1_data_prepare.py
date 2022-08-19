## Author: Beiming Cao
import time
import yaml
import os
import glob
import torch
from utils.IO_func import read_file_list
from shutil import copyfile
from utils.transforms import Transform_Compose, Pair_Transform_Compose
from utils.transforms import change_wav_sampling_rate, apply_delta_deltadelta_Src, apply_delta_deltadelta_Tar, apply_delta_deltadelta_Src_Tar, wav2melspec, apply_DTW
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
    transforms.append(fs_change)
    #####################################################

    data_path = config['corpus']['path']
    src_spk_list = config['data_setup']['source_spk_list']
    tar_spk_list = config['data_setup']['target_spk_list']

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

    all_SPK = list(set(src_spk_list + tar_spk_list))
    
    for SPK in all_SPK:
        out_folder_SPK = os.path.join(out_folder, SPK)
        if not os.path.exists(out_folder_SPK):
            os.makedirs(out_folder_SPK)

        SPK_folder = os.path.join(data_path, 'cmu_us_' + SPK + '_arctic')
        SPK_wav_folder = os.path.join(SPK_folder, 'wav')
        SPK_wav_list = glob.glob(SPK_wav_folder + '/*.wav')        

        for wav_path in SPK_wav_list:
            wav_id = os.path.basename(wav_path)[:-4]      
            fs, wav = read(wav_path)
            wav_pt = transforms_all(wav) 

            wav_out_dir = os.path.join(out_folder_SPK, wav_id + '.pt')
            torch.save(wav_pt, wav_out_dir)
 
            

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/VC_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')

    args = parser.parse_args()
    data_processing(args)
