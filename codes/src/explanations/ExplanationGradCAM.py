from src.explanations.Explanation import Explanation
from src.models.Model import Model
import torch
from pytorch_grad_cam import GradCAM

class ExplanationGradCAM(Explanation):
    def __init__(self):
        super().__init__("GradCAM")

    def _compute_explanation(self, model: Model, images: torch.Tensor) -> torch.Tensor:
        explanations = self.gradcam_explanation_method(input_tensor=images)
        return explanations

    def _update_explanation_method(self, model: Model):        
        self.gradcam_explanation_method = GradCAM(model=model.get_model(), **model.explanation_parameters_gradcam)
        