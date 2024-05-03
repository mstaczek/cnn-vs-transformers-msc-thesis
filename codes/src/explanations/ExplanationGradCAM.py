from src.explanations import Explanation
from src.models import Model
import torch
from pytorch_grad_cam import GradCAM
import cv2
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image

class ExplanationGradCAM(Explanation):
    def __init__(self, device: str = 'cpu'):
        super().__init__("GradCAM", device)

    def _compute_explanation(self, model: Model, images: torch.Tensor) -> torch.Tensor:
        explanations = self.gradcam_explanation_method(input_tensor=images.to(self.device))
        return explanations

    def _update_explanation_method(self, model: Model):        
        self.gradcam_explanation_method = GradCAM(model=model.get_model(), **model.explanation_parameters_gradcam)
