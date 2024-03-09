from src.models import Model
import torch
import timm

class Swin_T(Model):
    def __init__(self):
        super().__init__("Swin_T")

    def _build_model(self) -> torch.nn.Module:
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        print(f"Loaded model: https://huggingface.co/{model.default_cfg['hf_hub_id']}")
        return model
    
    def make_sure_is_initialized(self):
        if self.is_initialized is False:
            self.model = self._build_model()
            self.explanation_parameters_gradcam = {
                'target_layers': [self.model.layers[-1].blocks[-1]], # last Swin Transformer Block
                'reshape_transform' : self._gradcam_reshape_transform
            }
            self.is_initialized = True
            
    def _gradcam_reshape_transform(self, tensor): # Swin-T has tensor size BATCH x 7 x 7 x 768 at this layer
        result = tensor.transpose(2, 3).transpose(1, 2)
        return result