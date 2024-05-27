from src.models import Model
import torch
import timm

class DeiT_S(Model):
    def __init__(self, **kwargs):
        super().__init__("DeiT_S", **kwargs)

    def _load_model(self) -> torch.nn.Module:
        if self.pretrained_weights_name != 'imagenet':
            model = self._load_model_from_disk()
        if self.pretrained_weights_name == 'imagenet':
            model = timm.create_model('deit_small_patch16_224.fb_in1k', pretrained=True)
            print(f"Loaded default imagenet-pretrained model: https://huggingface.co/{model.default_cfg['hf_hub_id']}")
        return model
    
    def _initialize_model(self):
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

    def _gradcam_reshape_transform(self, tensor, height=14, width=14): # DeiT-S/16 has tensor size BATCH x 197 x 384 at this layer
        result = tensor[:, 1:, :] # remove class token
        result = result.reshape(result.size(0), height, width, result.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result