from src.models import Model
import torch
import timm

class DeiT_T(Model):
    def __init__(self, **kwargs):
        super().__init__("DeiT_T", **kwargs)

    def _build_model(self) -> torch.nn.Module:
        if self.pretrained_weights_name != 'imagenet':
            model = self._load_model_from_disk()
        if self.pretrained_weights_name == 'imagenet':
            model = timm.create_model('deit_tiny_patch16_224.fb_in1k', pretrained=True)
            print(f"Loaded default imagenet-pretrained model: https://huggingface.co/{model.default_cfg['hf_hub_id']}")
        return model
    
    def make_sure_is_initialized(self):
        if self.is_initialized is False:
            self.model = self._build_model()
            if self.pretrained_weights_name == 'imagenet':
                self.explanation_parameters_gradcam = {
                    'target_layers': [self.model.blocks[-1].norm1], # resulting size is 14 x 14, last before last attention block
                    'reshape_transform' : self._gradcam_reshape_transform
                }
            else: 
                self.explanation_parameters_gradcam = {
                    'target_layers': [self.model[0].model.blocks[-1].norm1], # resulting size is 14 x 14, last before last attention block
                    'reshape_transform' : self._gradcam_reshape_transform
                }
            self.is_initialized = True

    def _gradcam_reshape_transform(self, tensor, height=14, width=14): # DeiT-T/16 has tensor size BATCH x 197 x 192 at this layer
        result = tensor[:, 1:, :] # remove class token
        result = result.reshape(result.size(0), height, width, result.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result