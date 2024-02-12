from src.models.Model import Model
import torch
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18(Model):
    def __init__(self):
        super().__init__("ResNet18")

    def _build_model(self) -> torch.nn.Module:
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    def make_sure_is_initialized(self):
        if self.is_initialized is False:
            self.model = self._build_model()
            self.explanation_parameters_gradcam = {
                'target_layers': [self.model.layer4[-1]],
            }