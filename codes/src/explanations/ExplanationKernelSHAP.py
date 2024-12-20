from src.explanations import Explanation
from src.models import Model
import torch
from captum.attr import KernelShap
from skimage import segmentation
import cv2
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image

class ExplanationKernelSHAP(Explanation):
    N_SEGMENTS = 50
    N_SAMPLES = 350

    def __init__(self, device: str = 'cpu'):
        super().__init__("KernelSHAP", device)

    def _compute_explanation(self, model: Model, images: torch.Tensor) -> torch.Tensor:
        targets = model(images.to(self.device)).max(1).indices
        baselines = torch.zeros_like(images)
        feature_masks = torch.Tensor(np.array([
            segmentation.slic(input_tensor.clone().detach().cpu().numpy().squeeze().transpose((1, 2, 0)), n_segments=self.N_SEGMENTS, start_label=0)
            for input_tensor in images
        ])).int()
        attr_list = []
        for input_image, baseline, target, mask in\
                zip(images.to(self.device), baselines.to(self.device), targets, feature_masks.to(self.device)):
            attr = self.kernel_shap_explanation_method.attribute(
                input_image.unsqueeze(0),
                target=target.unsqueeze(0),
                baselines=baseline.unsqueeze(0),
                feature_mask=mask.unsqueeze(0),
                n_samples=self.N_SAMPLES
            )
            attr_list.append(attr)
        attributions = torch.concat(attr_list)
        results = attributions.cpu().detach().numpy().sum(axis=1)
        explanations = results / np.max(np.array(np.max(results), np.max(-results)))
        return explanations

    def _update_explanation_method(self, model: Model):        
        self.kernel_shap_explanation_method = KernelShap(model.get_model())
