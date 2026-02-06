import ring
from ring.extras import dataloader_torch
import os
import numpy as np
import pickle

def load_existing_data(data_path: str):
    data_files = dataloader_torch.FolderOfFilesDataset(data_path)
    print(f'The dataset contains {data_files.N} pickle files of motion data')
    loaded_data = [ring.utils.pickle_load(i) for i in data_files.files]
    return loaded_data

# data = load_existing_data(r'../4_seg')
# data = load_existing_data(r'../EMBC_KC')

def load_debug_files(data_folder: str, timestamp: str):
    debug_files = ['debug_X_train','debug_X_val','debug_Y_train','debug_Y_val']
    # debug_file_paths = [os.path.join(data_folder, f'{debug_file}_{timestamp}.npy') for debug_file in debug_files]
    debug_files_dict = {}
    for debug_file in debug_files:
        debug_files_dict[debug_file] = np.load(os.path.join(data_folder, f'{debug_file}_{timestamp}.npy'))
    return debug_files_dict

# debug_files = load_debug_files(r'C:\Users\Timo_Kuhlgatz\Seafile\Meine Bibliotheken\02_projects\04_IMU_Motion_capture\03_RING\ring_debug', '2025-12-17_17-58-10')