from src.explanations.Explanation import Explanation
from src.models.Model import Model
import torch
from pytorch_grad_cam import GradCAM
import cv2
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image

class ExplanationGradCAM(Explanation):
    def __init__(self):
        super().__init__("GradCAM")

    def _compute_explanation(self, model: Model, images: torch.Tensor) -> torch.Tensor:
        explanations = self.gradcam_explanation_method(input_tensor=images)
        return explanations

    def _update_explanation_method(self, model: Model):        
        self.gradcam_explanation_method = GradCAM(model=model.get_model(), **model.explanation_parameters_gradcam)
        
    def save(self, path_image: str, path_explanation: str, explanation: torch.Tensor):
        image = cv2.imread(path_image, 1)[:, :, ::-1]
        image = cv2.resize(image, (224, 224))
        image = np.float32(image) / 255
        image_with_explanation = show_cam_on_image(image, explanation)
        cv2.imwrite(path_explanation, image_with_explanation)