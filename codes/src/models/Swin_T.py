from src.models import Model
import torch
import timm

class Swin_T(Model):
    def __init__(self, **kwargs):
        super().__init__("Swin_T", **kwargs)

    def _load_model(self) -> torch.nn.Module:
        if self.pretrained_weights_name != 'imagenet':
            model = self._load_model_from_disk()
        if self.pretrained_weights_name == 'imagenet':
            model = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k', pretrained=True)
            print(f"Loaded default imagenet-pretrained model: https://huggingface.co/{model.default_cfg['hf_hub_id']}")
        return model
    
    def _initialize_model(self):
        self.explanation_parameters_gradcam = {
            'target_layers': [self.model.layers[-1].blocks[-1]], # last Swin Transformer Block
            'reshape_transform' : self._gradcam_reshape_transform
        }
            
    def _gradcam_reshape_transform(self, tensor): # Swin-T has tensor size BATCH x 7 x 7 x 768 at this layer
        result = tensor.transpose(2, 3).transpose(1, 2)
        return result