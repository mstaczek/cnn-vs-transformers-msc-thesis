from src.explanations import Explanation
from src.models import Model
import torch
from captum.attr import KernelShap
from skimage import segmentation
import cv2
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings

class ExplanationKernelSHAP(Explanation):
    N_SEGMENTS = 64
    N_SAMPLES = 100

    def __init__(self, device: str = 'cpu'):
        super().__init__("KernelSHAP", device)

    def _compute_explanation(self, model: Model, images: torch.Tensor) -> torch.Tensor:
        targets = model(images.to(self.device)).max(1).indices
        baselines = torch.zeros_like(images)
        feature_masks = torch.Tensor(np.array([
            segmentation.slic(input_tensor.clone().detach().cpu().numpy().squeeze().transpose((1, 2, 0)), n_segments=self.N_SEGMENTS, start_label=0)
            for input_tensor in images
        ])).int()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            attributions = self.kernel_shap_explanation_method.attribute(images.to(self.device), target=targets, 
                                                                         baselines=baselines.to(self.device), n_samples=self.N_SAMPLES, 
                                                                         feature_mask=feature_masks.to(self.device))
        results = attributions.cpu().detach().numpy().sum(axis=1)
        explanations = (results - np.min(results)) / (np.max(results) - np.min(results))
        return explanations

    def _update_explanation_method(self, model: Model):        
        self.kernel_shap_explanation_method = KernelShap(model.get_model())
        
    def save(self, path_image: str, path_explanation: str, explanation: torch.Tensor):
        image = cv2.imread(path_image, 1)[:, :, ::-1]
        image = cv2.resize(image, (224, 224))
        image = np.float32(image) / 255
        image_with_explanation = show_cam_on_image(image, explanation)
        cv2.imwrite(path_explanation, image_with_explanation)