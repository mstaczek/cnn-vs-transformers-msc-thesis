from torch.utils.data import DataLoader
import torch
from src.models.Model import Model
from src.explanations.Explanation import Explanation
import os
import numpy as np
from torchvision.datasets import DatasetFolder, ImageFolder
from src.datasets.utils import get_random_seed_dataloader_params

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

class DatasetManager:
    def __init__(self, name: str, root_images: str, root_explanations: str):
        self.name = name
        self.root_images = root_images
        self.root_explanations = root_explanations
        self.dataloader_random_seed_params = get_random_seed_dataloader_params()

    def _get_explanations_root(self, model: Model, explanation: Explanation):
        return os.path.join(self.root_explanations, self.name, explanation.name, model.name)

    def _get_transforms(self):
        raise NotImplementedError

    def get_dataloader(self, **kwargs) -> DataLoader:
        transforms = self._get_transforms()
        dataset = ImageFolderWithPaths(root=self.root_images, transform=transforms)
        dataloader = DataLoader(dataset, **self.dataloader_random_seed_params, **kwargs)
        return dataloader

    def save_explanations(self, model: Model, explanation_method: Explanation, explanations: torch.Tensor, 
                          paths: list):
        
        path_to_explanations = self._get_explanations_root(model, explanation_method)
        real_class_names = [os.path.split(os.path.dirname(p))[1] for p in paths]

        for explanation, path_to_image, real_class_name in zip(explanations, paths, real_class_names):
            output_dir = os.path.join(path_to_explanations, real_class_name)
            image_name = os.path.basename(path_to_image)
            image_name_without_extension = os.path.splitext(image_name)[0]
            output_path_npy = os.path.join(output_dir, image_name_without_extension + '.npy')
            output_path_img = os.path.join(output_dir, image_name_without_extension + '.png')
            os.makedirs(output_dir, exist_ok=True)

            np.save(output_path_npy, explanation)
            explanation_method.save(path_to_image, output_path_img, explanation)
    
    def get_explanations_dataloader(self, model: Model, explanation_method: Explanation, **kwargs) -> DataLoader:
        path_to_explanations = self._get_explanations_root(model, explanation_method)
        dataset = NpyFolderWithPaths(path_to_explanations)
        dataloader = DataLoader(dataset, **self.dataloader_random_seed_params, **kwargs)
        return dataloader
            
    def get_computed_explanations_list(self, model: Model, explanation_method: Explanation):
        path_to_explanations = self._get_explanations_root(model, explanation_method)
        dataset = NpyFolderWithPaths(path_to_explanations)
        return dataset.samples