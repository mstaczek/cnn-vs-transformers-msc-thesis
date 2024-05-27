from src.utils import create_classes_from_strings
from tqdm import tqdm


def compute_explanations(dataset_name: str, model_name: str, explanation_name: str, root_images=None, 
                         root_explanations=None, number_of_batches_to_process=None, device='cpu', 
                         models_weigths_pretrained='imagenet', root_trained_models=None, **kwargs):
    """
        defaults paths are '../datasets/imagenette2/train' and '../explanations'
        pretrained models are loaded by default from f'../trained_models/{self.root_trained_models}/{self.model_name}/', 
            otherwise from `root_trained_models + '/{self.model_name}/'` for each model. Fallback is imagenet pretrained weights.
    """

    dataset_manager, model, explanation = create_classes_from_strings(model_name, dataset_name, explanation_name, 
                                                                      root_images, root_explanations, device,
                                                                      models_weigths_pretrained, root_trained_models)

    dataloader = dataset_manager.get_dataloader(model=model, **kwargs)

    explanation.before_computing_explanations(model)

    processed_images = 0
    if number_of_batches_to_process is None:
        number_of_batches_to_process = len(dataloader)
    number_of_batches_to_process = min(number_of_batches_to_process, len(dataloader))
    images_to_process_limit = min(len(dataloader.dataset), number_of_batches_to_process * dataloader.batch_size)

    pbar = tqdm(dataloader, desc='Computing explanations for batches', total=number_of_batches_to_process)
   
    for images, labels, paths in pbar:

        if processed_images >= images_to_process_limit:
            break
        
        explanations, predicted_labels = explanation.compute_explanation(model, images)
        dataset_manager.save_explanations(model, explanation, explanations, paths, predicted_labels)

        processed_images += len(images)

    explanation.after_computing_explanations(model)