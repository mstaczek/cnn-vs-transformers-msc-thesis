from src.models import Model
import torch

class Explanation:
    def __init__(self, name: str, device: str = 'cpu'):
        self.name = name
        self.model_name = None
        self.device = device

    def before_computing_explanations(self, model: Model):
        model.make_sure_is_initialized()
        model.move_to_device(self.device)
        self._update_model_name(model)

    def compute_explanation(self, model: Model, images: torch.Tensor):
        explanations = self._compute_explanation(model, images)
        predicted_labels = model(images.to(self.device)).cpu().detach().numpy().argmax(axis=1).tolist()
        return explanations, predicted_labels
    
    def _update_model_name(self, model: Model):
        if self.model_name != model.name:
            self.model_name = model.name
        self._update_explanation_method(model)

    def _update_explanation_method(self, model: Model):
        raise NotImplementedError
    
    def _compute_explanation(self, model: Model, images: torch.Tensor) -> torch.Tensor:
        # internally, handles both gpu and cpu by using self.device. Model is already on self.device
        # return size Batch x H x W
        raise NotImplementedError
    
    def save(self, path_image: str, path_explanation: str, explanation: torch.Tensor):
        raise NotImplementedError
    
    def after_computing_explanations(self, model):
        model.move_to_device('cpu')
        torch.cuda.empty_cache()