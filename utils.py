import ring
from ring.extras import dataloader_torch

def load_existing_data(data_path: str):
    data_files = dataloader_torch.FolderOfFilesDataset(data_path)
    print(f'The dataset contains {data_files.N} pickle files of motion data')
    loaded_data = [ring.utils.pickle_load(i) for i in data_files.files]
    return loaded_data

# data = load_existing_data(r'../4_seg')