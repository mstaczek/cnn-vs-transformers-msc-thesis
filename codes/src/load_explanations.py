from src.utils import create_classes_from_strings, assert_class_names_are_defined
from tqdm import tqdm
import os
import torch
import pandas as pd


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

def load_explanations_of_many_models(dataset_name: str, model_names: list[str], explanation_name: str, root_images=None, 
                     root_explanations=None, unify_lengths=True, **kwargs):
    """
        defaults paths to '../datasets/imagenette2/train' and '../explanations'
    """
    for model_name in model_names:
        assert_class_names_are_defined(model_name=model_name)

    loaded_data_list = []
    for model_name in model_names:
        loaded_data = load_explanations(dataset_name, model_name, explanation_name, root_images, root_explanations, **kwargs)
        loaded_data_list.append(loaded_data)

    if not unify_lengths:
        return loaded_data_list
    
    paths_list_filtered = _get_paths_common_to_all_models(loaded_data_list)
    
    filtered_loaded_data_list = []
    for loaded_data in loaded_data_list:
        filtered_loaded_data = _filter_loaded_data_by_image_paths(loaded_data, paths_list_filtered)
        filtered_loaded_data_list.append(filtered_loaded_data)

    print(f'Loaded {len(paths_list_filtered)} common explanations for each model.')

    return filtered_loaded_data_list

def _get_paths_common_to_all_models(loaded_data_list: list[dict]):
    paths_list = [loaded_data['paths'] for loaded_data in loaded_data_list]
    paths_list = [path for sublist in paths_list for path in sublist]
    paths_list = [_trim_image_path(path) for path in paths_list] 

    # get a list paths that are present for all models
    paths_dataframe = pd.DataFrame(paths_list, columns=['path'])
    paths_dataframe = paths_dataframe.groupby('path').size().reset_index(name='counts')
    paths_list_filtered = paths_dataframe[paths_dataframe['counts'] == len(loaded_data_list)][['path']].values.tolist()
    paths_list_filtered = [path[0] for path in paths_list_filtered]
    return paths_list_filtered

def _filter_loaded_data_by_image_paths(loaded_data: dict, paths_to_leave_list: list[str]) -> dict:
    indices_to_leave = [i for i, path in enumerate(loaded_data['paths']) if _trim_image_path(path) in paths_to_leave_list]
    filtered_loaded_data = {'explanations': loaded_data['explanations'][indices_to_leave], 
                            'paths': [loaded_data['paths'][i] for i in indices_to_leave], 
                            'labels': loaded_data['labels'][indices_to_leave], 
                            'model_name': loaded_data['model_name'], 
                            'explanation_name': loaded_data['explanation_name']}
    return filtered_loaded_data

def _trim_image_path(path: str):
    """a/b/c/d/e.jpg -> d/e.jpg - leavly only filename and its parent directory name"""
    return os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))