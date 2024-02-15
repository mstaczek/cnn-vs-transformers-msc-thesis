import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel

torch.manual_seed(0)

# ImageFolder that also returns paths
class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (img, label ,path)

def dataset_load_imagenette(path, batch_size=64, shuffle=True, **kwargs):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.461], std=[0.219])
    ])
    dataset = ImageFolderWithPaths(root=path, transform=transform)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            generator=g, worker_init_fn=seed_worker, **kwargs)
    return dataloader

# DatasetFolder that loads npy files and also returns paths
class NpyFolder(DatasetFolder):
    def __init__(self, root):
        super(NpyFolder, self).__init__(root, np.load, extensions=('.npy',))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        return sample, target, path

def dataset_load_explanations(path, batch_size=64, shuffle=False, **kwargs):
    dataset = NpyFolder(path)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            generator=g, worker_init_fn=seed_worker, **kwargs)
    return dataloader

def save_explanations(explanations, paths, y, output_root, dataloader):

    def create_output_path(path_output_explanations, path_to_original_image, dataloader, original_class_id):
        class_name = dataloader.dataset.classes[original_class_id]
        image_name_without_extension = os.path.splitext(os.path.basename(path_to_original_image))[0]
        output_dir = os.path.join(path_output_explanations, class_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, image_name_without_extension + '.npy')
        return output_path
    
    for explanation, path_to_image, class_id in zip(explanations, paths, y):
        output_path = create_output_path(output_root, path_to_image, dataloader, class_id)
        np.save(output_path, explanation)

def compute_gradcam_explanations_for_model_and_dataset(model, gradcam_explanation_method, dataloader, path_output_explanations, number_of_images=16):
    counter = 0
    for batch in tqdm(dataloader, total=int(np.ceil(number_of_images / dataloader.batch_size))):
        x, y, paths = batch
        if counter + len(x) >= number_of_images:
            x = x[:number_of_images - counter]
            y = y[:number_of_images - counter]
            paths = paths[:number_of_images - counter]
        explanations = gradcam_explanation_method(input_tensor=x)
        save_explanations(explanations, paths, y, path_output_explanations, dataloader)
        counter += len(x)
        if counter >= number_of_images:
            break

def assure_explanations_of_same_images_are_compared(dataloader_ex1, dataloader_ex2):
    try:
        for batch1, batch2 in zip(dataloader_ex1, dataloader_ex2):
            ex1, y1, paths1 = batch1
            ex2, y2, paths2 = batch2
            assert y1.shape == y2.shape, 'batch sizes differ'
            assert (y1 == y2).all(), 'labels differ'
            for path1, path2 in zip(paths1, paths2):
                assert os.path.basename(path1) == os.path.basename(path2)
    except AssertionError as e:
        print(f'Dataloaders are not aligned: {e}')
        return False
    return True

def similarity_metric_rbf(explanation1, explanation2, batched=True):
    if batched:
        return rbf_kernel(explanation1.reshape(explanation1.shape[0], -1), explanation2.reshape(explanation2.shape[0], -1)).diagonal()
    else:
        return rbf_kernel(explanation1.reshape(1, -1), explanation2.reshape(1, -1))
    
def compute_models_similarity(explanationloader1, explanationloader2, similarity_metric):
    assure_explanations_of_same_images_are_compared(explanationloader1, explanationloader2)
    similarities = []
    for batch1, batch2 in zip(explanationloader1, explanationloader2):
        explanations1, y1, paths1 = batch1
        explanations2, y2, paths2 = batch2
        similarities.append(similarity_metric(explanations1, explanations2))
    return np.mean(similarities)