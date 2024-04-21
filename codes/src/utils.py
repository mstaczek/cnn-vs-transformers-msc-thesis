from src.strings_to_classes_mappings import datasets_mapping, models_mapping, explanations_mapping
import os
import torch
import numpy as np
import random

def create_classes_from_strings(model_name: str, dataset_name: str, explanation_name: str, root_images=None, root_explanations=None, device: str='cpu'):
    if root_images is None:
        root_images = os.path.join(os.pardir, 'datasets', 'imagenette2', 'train')
    if root_explanations is None:
        root_explanations = os.path.join(os.pardir, 'explanations')

    assert_class_names_are_defined(dataset_name, model_name, explanation_name)

    dataset_manager = datasets_mapping[dataset_name](root_images, root_explanations)
    model = models_mapping[model_name]()
    explanation = explanations_mapping[explanation_name](device)
    
    return dataset_manager, model, explanation

def assert_class_names_are_defined(dataset_name=None, model_name=None, explanation_name=None):
    if dataset_name is not None:
        assert dataset_name in datasets_mapping, f"Dataset name {dataset_name} not found in datasets_mapping, available datasets: {datasets_mapping}"
    if model_name is not None:
        assert model_name in models_mapping, f"Model name {model_name} not found in models_mapping, available models: {models_mapping}"
    if explanation_name is not None:
        assert explanation_name in explanations_mapping, f"Explanation name {explanation_name} not found in explanations_mapping, available explanations: {explanations_mapping}"

def set_seed(seed: int = 912723012):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

models_to_architecture_mapping = {
    'DeiT_S'          : 'transformer',
    'DeiT_T'          : 'transformer',
    'DenseNet121'     : 'CNN',
    'EfficientNet_B3' : 'CNN',
    'EfficientNet_B4' : 'CNN',
    'ConvNeXtV2_Nano' : 'CNN',
    'PyramidViT_V2_B2': 'transformer',
    'MobileNetV3'     : 'CNN',
    'Swin_T'          : 'transformer',
    'ResNet18'        : 'CNN',
    'ResNet50'        : 'CNN',
    'ResNeXt50'       : 'CNN',
    'Res2Net50'       : 'CNN',
    'ViT_B_32'        : 'transformer',
}