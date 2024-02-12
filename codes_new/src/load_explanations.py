from src.strings_to_classes_mappings import datasets_mapping, models_mapping, explanations_mapping
from tqdm import tqdm
import os
import torch

def load_explanations(dataset_name: str, model_name: str, explanation_name: str, root_images=None, 
                     root_explanations=None, **kwargs):
    """
        defaults paths to '../datasets/imagenette2/train' and '../explanations'
    """
    if root_images is None:
        root_images = os.path.join(os.pardir, 'datasets', 'imagenette2', 'train')
    if root_explanations is None:
        root_explanations = os.path.join(os.pardir, 'explanations')

    model = models_mapping[model_name]()
    dataset_manager = datasets_mapping[dataset_name](root_images, root_explanations)
    explanation_method = explanations_mapping[explanation_name]()

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