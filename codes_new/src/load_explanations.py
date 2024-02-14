from src.utils import create_classes_from_strings
from tqdm import tqdm
import os
import torch

def load_explanations(dataset_name: str, model_name: str, explanation_name: str, root_images=None, 
                     root_explanations=None, **kwargs):
    """
        defaults paths to '../datasets/imagenette2/train' and '../explanations'
    """

    dataset_manager, model, explanation_method = create_classes_from_strings(model_name, dataset_name, explanation_name, root_images, root_explanations)

    dataloader_explanations = dataset_manager.get_explanations_dataloader(model, explanation_method, **kwargs)
    
    explanations = []
    paths = []
    labels = []

    for explanations_batch, labels_batch, paths_batch in tqdm(dataloader_explanations, desc='Loading explanations'):
        explanations.append(explanations_batch)
        paths.append(paths_batch)
        labels.append(labels_batch)

    explanations = torch.cat(explanations, dim=0)
    paths = [item for sublist in paths for item in sublist]
    labels = torch.cat(labels, dim=0)
    
    loaded_explanations = {'explanations': explanations, 'paths': paths, 'labels': labels, 
                           'model_name': model.name, 'explanation_name': explanation_method.name}
    return loaded_explanations