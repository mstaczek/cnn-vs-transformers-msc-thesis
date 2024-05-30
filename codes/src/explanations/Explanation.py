from src.models import Model
import torch
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib.colors import LinearSegmentedColormap
import copy
import numpy as np

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
    
    def after_computing_explanations(self, model):
        model.move_to_device('cpu')
        torch.cuda.empty_cache()

    def save(self, path_image: str, path_explanation: str, explanation: torch.Tensor):
        only_explanation_path = path_explanation.replace('.png', '_only_explanation.png')
        self._save_explanation_as_png(only_explanation_path, explanation)
        
        image = cv2.imread(path_image, 1)[:, :, ::-1]
        image = cv2.resize(image, (224, 224))
        image = np.float32(image) / 255
        image_with_explanation = show_cam_on_image(image, explanation)
        cv2.imwrite(path_explanation, image_with_explanation)
        
    def _save_explanation_as_png(self, path_explanation: str, explanation: torch.Tensor):
        explanation = copy.deepcopy(explanation)
        cmap = LinearSegmentedColormap.from_list("red-white-green", ["red", "white", "green"])
        vmin, vmax = -1, 1
        attribution_normalized = 255 * (explanation - vmin) / (vmax - vmin)
        attribution_normalized = np.clip(attribution_normalized, 0, 255).astype(np.uint8)    
        attribution_colored = cmap(attribution_normalized / 255.0)[:, :, :3] 
        attribution_colored = (attribution_colored * 255).astype(np.uint8)     
        attribution_colored_bgr = cv2.cvtColor(attribution_colored, cv2.COLOR_RGB2BGR)    
        cv2.imwrite(path_explanation, attribution_colored_bgr)
    
