import os
import pandas as pd
from tqdm import tqdm
import h5py
import numpy as np

def open_hdf5_file(filename):
    def read_group(group):
        data = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):  # If it's a group, recurse
                data[key] = read_group(item)
            elif isinstance(item, h5py.Dataset):  # If it's a dataset, read it
                data[key] = item[()]
        return data

    with h5py.File(filename, "r") as f:
        data = read_group(f) 
    return data


def open_all_hdf5_file(folder_path):
    all_files = os.listdir(folder_path)
    dictionary = {}
    for i in range(len(all_files)):
        file = all_files[i]
        if file.endswith('.h5'):
            dictionary[i] = open_hdf5_file(os.path.join(folder_path, file))
        else:
            None
    return dictionary

def open_dat_files(folder_path: str, flag: str):
    dataframes = []
    
    # Iterate over all files in the folder
    for file in tqdm(os.listdir(folder_path)):
        # Process only .dat files
        if file.endswith('.dat') or file.endswith('.log'):
            dat_file = file
            header_file = file.replace('.dat', '.txt', '.log')
            header_path = os.path.join(folder_path, header_file)
            
            # Check if the corresponding header (.txt) file exists
            if not os.path.exists(header_path):
                print(f"Header file {header_file} not found for {dat_file} skipping.")
                continue
            
            # Read the header file to obtain column names
            try:
                with open(header_path, 'r') as hf:
                    header_line = hf.readline().strip()
                    headers = header_line.split()  # Assumes headers are space-separated
            except Exception as e:
                print(f"Error reading header file {header_file}: {e}")
                continue
            
            # Read the .dat file using the header names
            dat_path = os.path.join(folder_path, dat_file)
            try:
                df = pd.read_csv(dat_path, delimiter=r'\s+', names=headers, header=None)
                dataframes.append(df)
                print(f"Loaded file: {dat_file} with headers: {headers}")
            except Exception as e:
                print(f"Error loading {dat_file}: {e}")
    
    print(f"Number of loaded files: {len(dataframes)}")
    return dataframes