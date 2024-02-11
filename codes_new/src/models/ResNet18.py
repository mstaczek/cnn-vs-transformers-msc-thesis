from src.models.Model import Model
import torch
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18(Model):
    def __init__(self):
        super().__init__("ResNet18")
        self.explanation_parameters_gradcam = {
            'target_layers': [self.model.layer4[-1]],
        }

    def _build_model(self) -> torch.nn.Module:
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
