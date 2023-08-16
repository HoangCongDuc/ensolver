import argparse
import glob
import os
import torch


parser = argparse.ArgumentParser("Collect the best checkpoints in workdirs into the same folder and remove ema in each checkpoint")
parser.add_argument('work_dir')
parser.add_argument('dst_dir')

if __name__ == '__main__':
    args = parser.parse_args()
    work_dir=  args.work_dir
    dst_dir = args.dst_dir
    
    ckpts = glob.glob(f'{work_dir}/**/*best*')
    for i, ckpt_file in enumerate(ckpts):
        ckpt_data = torch.load(ckpt_file)
        ckpt_data['state_dict'] = {k: v for k, v in ckpt_data['state_dict'].items() if 'ema' not in k}
        save_path = os.path.join(dst_dir, f'model{i}.pth')
        torch.save(ckpt_data, save_path)