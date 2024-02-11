import torch
import numpy as np
import random

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