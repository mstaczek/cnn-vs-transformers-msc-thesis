from src.explanations import Explanation
from src.models import Model
import torch
import numpy as np
from captum.attr import IntegratedGradients


class ExplanationIntegratedGradients(Explanation):
    N_STEPS = 100 # default 50

    def __init__(self, device: str = 'cpu'):
        super().__init__("IntegratedGradients", device)

    def _compute_explanation(self, model: Model, images: torch.Tensor) -> torch.Tensor:
        targets = model(images.to(self.device)).max(1).indices
        attributions = self.integrated_gradients_explanation_method.attribute(images.to(self.device), 
                            target=targets, n_steps=self.N_STEPS, internal_batch_size=images.shape[0]) 
        
        explanations = attributions.cpu().detach().numpy().sum(axis=1)
        for i in range(explanations.shape[0]):
            high_percentile = np.percentile(explanations[i], 99.7)
            low_percentile = np.percentile(explanations[i], 0.3)
            explanations[i, explanations[i] > high_percentile] = high_percentile
            explanations[i, explanations[i] < low_percentile] = low_percentile
            explanations[i] = explanations[i] / np.max(np.abs(explanations[i]))
        return explanations

    def _update_explanation_method(self, model: Model):        
        self.integrated_gradients_explanation_method = IntegratedGradients(model.get_model()) 
        