from torch.utils.data import DataLoader
import torch
from src.models import Model
from src.explanations import Explanation
import os
import numpy as np
from torchvision.datasets import DatasetFolder, ImageFolder
from src.datasets.utils import get_random_seed_dataloader_params, get_classes_id_to_name_mapping, get_classes_name_to_id_mapping

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (img, label ,path)

class NpyFolderWithPaths(DatasetFolder):
    def __init__(self, root):
        super(NpyFolderWithPaths, self).__init__(root, np.load, extensions=('.npy',))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        return sample, target, path
    
class NpyFolderWithPathsAndPredictions(NpyFolderWithPaths):
    def __init__(self, root, classes_to_id_mapping):
        super(NpyFolderWithPathsAndPredictions, self).__init__(root)
        self.classes_to_id_mapping = classes_to_id_mapping

    def _load_models_prediction(self, path):
        txt_path = os.path.splitext(path)[0] + '.txt'
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                model_prediction = f.read()
        else:
            model_prediction = -1
        return model_prediction

    def __getitem__(self, index):
        sample, target, path = super(NpyFolderWithPathsAndPredictions, self).__getitem__(index)
        model_prediction_class_name = self._load_models_prediction(path)
        model_prediction_class_id = self.classes_to_id_mapping[model_prediction_class_name] if model_prediction_class_name in self.classes_to_id_mapping else -1
        return sample, target, model_prediction_class_id, path

class DatasetManager:
    def __init__(self, name: str, root_images: str, root_explanations: str):
        self.name = name
        self.root_images = root_images
        self.root_explanations = root_explanations
        self.dataloader_random_seed_params = get_random_seed_dataloader_params()
        self.mapping_classes_to_id = get_classes_name_to_id_mapping(self.name)
        self.mapping_id_to_class_name = get_classes_id_to_name_mapping(self.name)

    def _get_explanations_root(self, model: Model, explanation: Explanation):
        return os.path.join(self.root_explanations, self.name, explanation.name, model.name)

    def _get_transforms(self):
        raise NotImplementedError

    def get_dataloader(self, model: Model, **kwargs) -> DataLoader:
        transforms = self._get_transforms(model)
        dataset = ImageFolderWithPaths(root=self.root_images, transform=transforms)
        dataloader = DataLoader(dataset, **self.dataloader_random_seed_params, **kwargs)
        return dataloader

    def save_explanations(self, model: Model, explanation_method: Explanation, explanations: torch.Tensor, 
                          paths: list, predictions: list[int]):
        
        path_to_explanations = self._get_explanations_root(model, explanation_method)
        real_class_names = [os.path.split(os.path.dirname(p))[1] for p in paths]

        for explanation, prediction, path_to_image, real_class_name in \
                zip(explanations, predictions, paths, real_class_names):
            output_dir = os.path.join(path_to_explanations, real_class_name)
            image_name = os.path.basename(path_to_image)
            image_name_without_extension = os.path.splitext(image_name)[0]
            output_path_npy = os.path.join(output_dir, image_name_without_extension + '.npy')
            output_path_img = os.path.join(output_dir, image_name_without_extension + '.png')
            output_path_prediction = os.path.join(output_dir, image_name_without_extension + '.txt')
            os.makedirs(output_dir, exist_ok=True)

            np.save(output_path_npy, explanation)
            explanation_method.save(path_to_image, output_path_img, explanation)
            with open(output_path_prediction, 'w') as f:
                prediction_class_name = self.mapping_id_to_class_name[prediction]
                f.write(prediction_class_name)
    
    def get_explanations_dataloader(self, model: Model, explanation_method: Explanation, **kwargs) -> DataLoader:
        path_to_explanations = self._get_explanations_root(model, explanation_method)
        dataset = NpyFolderWithPathsAndPredictions(path_to_explanations, self.mapping_classes_to_id)
        dataloader = DataLoader(dataset, **self.dataloader_random_seed_params, **kwargs)
        return dataloader
            
    def get_computed_explanations_list(self, model: Model, explanation_method: Explanation):
        path_to_explanations = self._get_explanations_root(model, explanation_method)
        dataset = NpyFolderWithPathsAndPredictions(path_to_explanations, self.mapping_classes_to_id)
        return dataset.samples