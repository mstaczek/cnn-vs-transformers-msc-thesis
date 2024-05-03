from src.explanations import Explanation
from src.models import Model
import torch
import numpy as np
from captum.attr import IntegratedGradients


class ExplanationIntegratedGradients(Explanation):
    N_STEPS = 50 # default 50

    def __init__(self, device: str = 'cpu'):
        super().__init__("IntegratedGradients", device)

    def _compute_explanation(self, model: Model, images: torch.Tensor) -> torch.Tensor:
        targets = model(images.to(self.device)).max(1).indices
        attributions = self.integrated_gradients_explanation_method.attribute(images.to(self.device), 
                            target=targets, n_steps=self.N_STEPS, internal_batch_size=32) 
        
        results = attributions.cpu().detach().numpy().sum(axis=1)
        explanations = (results - np.min(results)) / (np.max(results) - np.min(results))
        return explanations

    def _update_explanation_method(self, model: Model):        
        self.integrated_gradients_explanation_method = IntegratedGradients(model.get_model()) 
        