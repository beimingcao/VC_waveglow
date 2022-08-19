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
from torch.optim.lr_scheduler import StepLR
import random

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def train_LSTM(args):

  #  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cpu'

    config_path = args.conf_dir       
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    data_path = config['corpus']['path']
    src_spk_list = config['data_setup']['source_spk_list']
    tar_spk_list = config['data_setup']['target_spk_list']
    D_in = config['acoustic_feature']['n_mel_channels']
    D_out = config['acoustic_feature']['n_mel_channels']

    hidden_size = config['NN_setup']['hidden_size']
    num_layers = config['NN_setup']['layer_num']
    batch_size = config['NN_setup']['batch_size']

    learning_rate = config['NN_setup']['learning_rate']
    weight_decay = config['NN_setup']['weight_decay']
    num_epoch = config['NN_setup']['num_epoch']

    for tar_spk in tar_spk_list:
        src_spk_unique = src_spk_list.copy()
        src_spk_unique.remove(tar_spk)
        for src_spk in src_spk_unique:
            sub_exp_id = src_spk + '_vc_' + tar_spk
            sub_exp_folder = os.path.join(args.buff_dir, sub_exp_id)

            tr = open(os.path.join(sub_exp_folder, 'train_data.pkl'), 'rb') 
            va = open(os.path.join(sub_exp_folder, 'valid_data.pkl'), 'rb')        
            train_dataset, valid_dataset = pickle.load(tr), pickle.load(va)

            model = MyLSTM(D_in, hidden_size, D_out, num_layers)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.6)
            loss_func = RegressionLoss()
            metric = MCD()
            model.to(device)

            results = os.path.join(sub_exp_folder, sub_exp_id + '_train.txt')

            train_data = DataLoader(train_dataset, num_workers=0, batch_size=1, shuffle=True, drop_last=False)
            valid_data = DataLoader(valid_dataset, num_workers=0, batch_size=1, shuffle=True, drop_last=False)

            with open(results, 'w') as r:
                for epoch in range(num_epoch):
                    model.train()
                    acc_vals = []
                    for x, y in train_data:
                        x, y = x.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)
                        h, c = model.init_hidden(x)
                        h, c = h.to(device), c.to(device)
                        y_head = model(x, h, c)

                        loss_val = loss_func(y_head, y)
                        acc_val = metric(y_head.squeeze(0), y.squeeze(0))
                        acc_vals.append(acc_val)

                        optimizer.zero_grad()
                        loss_val.backward()
                        optimizer.step()    
                    avg_acc = sum(acc_vals) / len(acc_vals)

                    model.eval()
                    acc_vals = []
                    for x, y in valid_data:
                        x, y = x.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)
                        h, c = model.init_hidden(x)
                        h, c = h.to(device), c.to(device)
                        acc_vals.append(metric(model(x, h, c).squeeze(0), y.squeeze(0)))
                    scheduler.step()
                    avg_vacc = sum(acc_vals) / len(acc_vals)

                    print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc))
                    print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc), file = r)

                    model_out_folder = os.path.join(sub_exp_folder, 'trained_models')
                    save_model(model, os.path.join(model_out_folder, sub_exp_id + '_lstm'))
            r.close()
            print('Training for ' + sub_exp_id + ' is done.')




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/VC_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')
    args = parser.parse_args()
    train_LSTM(args)
   
