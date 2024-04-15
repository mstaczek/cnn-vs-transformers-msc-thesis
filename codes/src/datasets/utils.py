import torch
import numpy as np
import random
from src.datasets.imagenette2_class_to_id_mapping import imagenette2_id_to_class_mapping, imagenette2_class_to_id_mapping

def get_random_seed_dataloader_params() -> dict:
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(0)
    dataloader_random_seed_params = {
        'generator': g,
        'worker_init_fn': seed_worker,
    }
    return dataloader_random_seed_params

def get_classes_id_to_name_mapping(dataset_name):
    if dataset_name == 'imagenette2':
        return imagenette2_id_to_class_mapping
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}')
    

def get_classes_name_to_id_mapping(dataset_name):
    if dataset_name == 'imagenette2':
        return imagenette2_class_to_id_mapping
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}')