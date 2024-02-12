from src.strings_to_classes_mappings import datasets_mapping, models_mapping, explanations_mapping
from tqdm import tqdm
import os

def compute_explanations(dataset_name: str, model_name: str, explanation_name: str, root_images=None, 
                         root_explanations=None, number_of_batches_to_process=None, **kwargs):
    """
        defaults paths to '../datasets/imagenette2/train' and '../explanations'
    """
    if root_images is None:
        root_images = os.path.join(os.pardir, 'datasets', 'imagenette2', 'train')
    if root_explanations is None:
        root_explanations = os.path.join(os.pardir, 'explanations')

    model = models_mapping[model_name]()
    dataset_manager = datasets_mapping[dataset_name](root_images, root_explanations)
    explanation = explanations_mapping[explanation_name]()

    dataloader = dataset_manager.get_dataloader(**kwargs)

    processed_images = 0
    if number_of_batches_to_process is None:
        number_of_batches_to_process = len(dataloader)
    number_of_batches_to_process = min(number_of_batches_to_process, len(dataloader))
    images_to_process_limit = min(len(dataloader.dataset), number_of_batches_to_process * dataloader.batch_size)

    pbar = tqdm(dataloader, desc='Computing explanations for batches', total=number_of_batches_to_process)
   
    for images, labels, paths in pbar:

        if processed_images >= images_to_process_limit:
            break
        
        explanations = explanation.compute_explanation(model, images)
        dataset_manager.save_explanations(model, explanation, explanations, paths)

        processed_images += len(images)