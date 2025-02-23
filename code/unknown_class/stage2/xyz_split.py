import os
import shutil
import argparse

def split_folders(parent_folder, num_parts):
    subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
    pp_folder = os.path.dirname(parent_folder)
    folders_per_part = len(subfolders) // num_parts

    for i in range(num_parts):
        part_folder = os.path.join(pp_folder, f'part_{i + 1}')
        os.makedirs(part_folder, exist_ok=True)

        start_index = i * folders_per_part
        end_index = (i + 1) * folders_per_part if i < num_parts - 1 else None

        for folder_name in subfolders[start_index:end_index]:
            source_path = os.path.join(parent_folder, folder_name)
            destination_path = os.path.join(part_folder, folder_name)
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Result Statistics for Stage 1.')
    parser.add_argument('--sample_path', type=str, default="./sample/",
                        help='Path to sampled xyz dir')
    parser.add_argument('--num_parts', type=int, default=32,
                        help='Path to sampled xyz dir')
    args = parser.parse_args()
    parent_folder = args.sample_path
    num_parts = args.num_parts
    split_folders(parent_folder, num_parts)