from src.models.Model import Model
import torch

class Explanation:
    def __init__(self, name: str):
        self.name = name
        self.model_name = None

    def compute_explanation(self, model: Model, images: torch.Tensor) -> torch.Tensor:
        model.make_sure_is_initialized()
        self._update_model_name(model)
        return self._compute_explanation(model, images)

    def _update_model_name(self, model: Model):
        if self.model_name != model.name:
            self.model_name = model.name
            self._update_explanation_method(model)

    def _update_explanation_method(self, model: Model):
        raise NotImplementedError
    
    def _compute_explanation(self, model: Model, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def save(self, path_image: str, path_explanation: str, explanation: torch.Tensor):
        raise NotImplementedError