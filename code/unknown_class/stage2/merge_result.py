import pandas as pd
import glob
from src.utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--used_path', type=str, default='./sample-small/uspto_final_test/sampled_size/uspto_size_gnnbest_epoch=199/latent_epoch=294')
parser.add_argument('--n_samples', default=300, type=int)
args = parser.parse_args()

def main():
    files = glob.glob(args.used_path + '/part_*/result.txt')
    dataframes = [pd.read_csv(file) for file in files]
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv('result.csv', index=False)

    current_path = '.'
    path_csv_file_test = "./dataset_save_test/uspto_final_test_table.csv" 
    n_samples = args.n_samples

    result_path = current_path + '/result.csv'
    save_merged_result_path = current_path + '/merged_result.csv'
    uspto_final_test_table = pd.read_csv(path_csv_file_test)

    merge_res(current_path, n_samples, result_path, save_merged_result_path, uspto_final_test_table)

if __name__ == "__main__":
    main()
