from src.models.Model import Model
import torch
from torchvision.models import swin_t, Swin_T_Weights

class Swin_T(Model):
    def __init__(self):
        super().__init__("Swin_T")

    def _build_model(self) -> torch.nn.Module:
        return swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    
    def make_sure_is_initialized(self):
        if self.is_initialized is False:
            self.model = self._build_model()
            self.explanation_parameters_gradcam = {
                'target_layers': [self.model.features[-1][-1]], # last Swin Transformer Block
                'reshape_transform' : self._gradcam_reshape_transform
            }
            
    def _gradcam_reshape_transform(self, tensor): # Swin-T has tensor size BATCH x 7 x 7 x 768 at this layer
        result = tensor.transpose(2, 3).transpose(1, 2)
        return result