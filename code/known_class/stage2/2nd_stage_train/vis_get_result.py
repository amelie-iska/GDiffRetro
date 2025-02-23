import os
import subprocess
import argparse
import csv
import torch
import numpy as np
from src.visualizer import visualize_chain, load_molecule_xyz

parser = argparse.ArgumentParser()
parser.add_argument('--save_result', action='store', type=str)
parser.add_argument('--eval_result', action='store', type=str)
parser.add_argument('--used_path', type=str, default='./sample-small/uspto_final_test/sampled_size/uspto_size_gnnbest_epoch=199/latent_epoch=294')
parser.add_argument('--n_samples', type=int)
args = parser.parse_args()

def main():
    used_path = args.used_path
    result_path = used_path + '/result.txt'
    error_uuid_path = used_path + '/error_uuid.txt'
    path_sample = used_path
    n_samples = args.n_samples

    if not os.path.exists(result_path):
        open(result_path, 'w').close()

    if not os.path.exists(error_uuid_path):
        open(error_uuid_path, 'w').close()

    idx_name = 0
    num_files = len(os.listdir(path_sample))
    attributes = ["uuid"] + list(map(str, list(range(1, args.n_samples + 2))))
    with open(result_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(attributes)

    for name in os.listdir(path_sample):

        idx_name += 1        
        if name == 'result.txt' or name == 'error_uuid.txt' or name == 'merged_result.txt' or name == 'merged_error_uuid.txt':
            continue

        name = int(name)
        path_sample_name = path_sample + f"/{name}/"
        for xyz_name in os.listdir(path_sample_name):
            if xyz_name.endswith('.xyz') :
                if not os.path.exists(path_sample_name + xyz_name.split(".")[0] + ".smi"):
                    command = "obabel -ixyz " + path_sample_name + xyz_name + " -osmi -O " + path_sample_name + \
                              xyz_name.split(".")[0] + ".smi"
                    subprocess.run(command, shell=True)
        res = []
        res.append(str(name))
        for i in range(n_samples):
            with open(path_sample_name + str(i) + '_.smi') as f:
                smiles = f.read().split('\t')[0]
                res.append(smiles)

        with open(path_sample_name + 'true_.smi') as f:
            smiles = f.read().split('\t')[0]
            res.append(smiles)

        with open(result_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(res)
    
if __name__ == "__main__":
    main()
