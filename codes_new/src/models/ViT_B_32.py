from src.models.Model import Model
import torch
from torchvision.models import vit_b_32, ViT_B_32_Weights

class ViT_B_32(Model):
    def __init__(self):
        super().__init__("ViT_B_32")

    def _build_model(self) -> torch.nn.Module:
        return vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    
    def make_sure_is_initialized(self):
        if self.is_initialized is False:
            self.model = self._build_model()
            self.explanation_parameters_gradcam = {
                'target_layers': [self.model.encoder.layers[-1]], # last ViT Encoder Block
                'reshape_transform' : self._gradcam_reshape_transform
            }

    def _gradcam_reshape_transform(self, tensor, height=7, width=7): # ViT-B/32 has tensor size BATCH x 50 x 768 at this layer
        result = tensor[:, 1:, :] # remove class token
        result = result.reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result