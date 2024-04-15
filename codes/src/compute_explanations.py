from src.utils import create_classes_from_strings
from tqdm import tqdm


def compute_explanations(dataset_name: str, model_name: str, explanation_name: str, root_images=None, 
                         root_explanations=None, number_of_batches_to_process=None, device='cpu', **kwargs):
    """
        defaults paths to '../datasets/imagenette2/train' and '../explanations'
    """

    dataset_manager, model, explanation = create_classes_from_strings(model_name, dataset_name, explanation_name, root_images, root_explanations, device)

    dataloader = dataset_manager.get_dataloader(**kwargs)

    processed_images = 0
    if number_of_batches_to_process is None:
        number_of_batches_to_process = len(dataloader)
    number_of_batches_to_process = min(number_of_batches_to_process, len(dataloader))
    images_to_process_limit = min(len(dataloader.dataset), number_of_batches_to_process * dataloader.batch_size)
   
    model.make_sure_is_initialized()

    pbar = tqdm(dataloader, desc='Computing explanations for batches', total=number_of_batches_to_process)
   
    for images, labels, paths in pbar:

        if processed_images >= images_to_process_limit:
            break
        
        explanations = explanation.compute_explanation(model, images)
        predicted_labels = model(images).cpu().detach().numpy().argmax(axis=1).tolist()
        dataset_manager.save_explanations(model, explanation, explanations, paths, predicted_labels)

        processed_images += len(images)