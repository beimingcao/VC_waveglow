import time
import yaml
import os
from shutil import copyfile

def setup(args):

    ## exp_path: a folder to save all previous experiment
    ## buff_dir: a folder to save current experiment, contain will be copy to exp_path, if "new_experiment==True" in run.sh

    exp_path = args.exp_dir
    buff_path = args.buff_dir   
    config_path = args.conf_dir

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    if not os.path.exists(buff_path):
        os.makedirs(buff_path)

    ## copy conf to backup the current configuration 

    conf_dst = os.path.join(buff_path, os.path.basename(config_path))        
    copyfile(config_path, conf_dst)

    ## copy models.py to backup the current model 
    
    model_dst = os.path.join(buff_path, 'models.py')
    copyfile('models.py', model_dst)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/VC_conf.yaml')
    parser.add_argument('--exp_dir', default = 'experiments')
    parser.add_argument('--buff_dir', default = 'current_exp')

    args = parser.parse_args()
    setup(args)
