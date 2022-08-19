import time
import yaml
import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from models import MyLSTM
from models import RegressionLoss
from models import save_model
from utils.measures import MCD
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import ndimage

def test_LSTM(args):
    config_path = args.conf_dir       
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    data_path = config['corpus']['path']

    output_norm = config['training_setup']['normalization']['output']
    src_spk_list = config['data_setup']['source_spk_list']
    tar_spk_list = config['data_setup']['target_spk_list']
    D_in = config['acoustic_feature']['n_mel_channels']
    D_out = config['acoustic_feature']['n_mel_channels']

    hidden_size = config['NN_setup']['hidden_size']
    num_layers = config['NN_setup']['layer_num']
    batch_size = config['NN_setup']['batch_size']
    save_output = config['testing_setup']['save_output']
    synthesis_samples = config['testing_setup']['synthesis_samples']
    metric = MCD()

    for tar_spk in tar_spk_list:
        src_spk_unique = src_spk_list.copy()
        src_spk_unique.remove(tar_spk)
        for src_spk in src_spk_unique:
            sub_exp_id = src_spk + '_vc_' + tar_spk
            sub_exp_folder = os.path.join(args.buff_dir, sub_exp_id)

            te = open(os.path.join(sub_exp_folder, 'test_data.pkl'), 'rb')        
            test_dataset = pickle.load(te)

            test_data = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False, drop_last=False)
            test_out_folder = os.path.join(sub_exp_folder, 'testing')
            if not os.path.exists(test_out_folder):
                os.makedirs(test_out_folder)
   

            model_path = os.path.join(sub_exp_folder, sub_exp_id + '_lstm')
            model = MyLSTM(D_in, hidden_size, D_out, num_layers)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()

            acc_vals = []
            for file_id, x, y in test_data:
                x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)

                h, c = model.init_hidden(x)
                with torch.no_grad():
                    y_head = model(x, h, c)

                y_pt = y_head.squeeze(0).T

                if save_output == True:
                    outpath = os.path.join(sub_exp_folder, 'pred_pt')
                    if not os.path.exists(outpath):
                        os.makedirs(outpath)
                    if output_norm == True:
                        out_mean = torch.load(os.path.join(sub_exp_folder, 'tar_mean.pt'))
                        out_std = torch.load(os.path.join(sub_exp_folder, 'tar_std.pt'))
                        y_pt_T = y_pt.T * out_std + out_mean
                        y_pt = y_pt_T.T
                    torch.save(y_pt, os.path.join(outpath, file_id[0] + '.pt'))

                acc_vals.append(metric(y.squeeze(0), y_head.squeeze(0)))
            avg_vacc = sum(acc_vals) / len(acc_vals)

            results_out_folder = os.path.join(sub_exp_folder, 'RESULTS')
            if not os.path.exists(results_out_folder):
                os.makedirs(results_out_folder)

            results = os.path.join(sub_exp_id + '_results.txt')
            with open(results, 'w') as r:
                print('MCD = %0.3f' % avg_vacc, file = r)
            r.close()

       



if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/VC_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')
    args = parser.parse_args()
    test_LSTM(args)
