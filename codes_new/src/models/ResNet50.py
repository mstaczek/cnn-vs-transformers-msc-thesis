from src.models.Model import Model
import torch
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(Model):
    def __init__(self):
        super().__init__("ResNet50")

    def _build_model(self) -> torch.nn.Module:
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    def make_sure_is_initialized(self):
        if self.is_initialized is False:
            self.model = self._build_model()
            self.explanation_parameters_gradcam = {
                'target_layers': [self.model.layer4[-1]], # resulting size is 7 x 7
            }