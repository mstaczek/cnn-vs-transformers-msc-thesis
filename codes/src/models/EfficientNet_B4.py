from src.models import Model
import torch
import timm

class EfficientNet_B4(Model):
    def __init__(self, **kwargs):
        super().__init__("EfficientNet_B4", **kwargs)

    def _load_model(self) -> torch.nn.Module:
        if self.pretrained_weights_name != 'imagenet':
            model = self._load_model_from_disk()
        if self.pretrained_weights_name == 'imagenet':
            model = timm.create_model('efficientnet_b4.ra2_in1k', pretrained=True)
            print(f"Loaded default imagenet-pretrained model: https://huggingface.co/{model.default_cfg['hf_hub_id']}")
        return model
    
    def _initialize_model(self):
        if self.pretrained_weights_name == 'imagenet':
            self.explanation_parameters_gradcam = {
                'target_layers': [self.model.conv_head], # resulting size is 7 x 7
                'reshape_transform' : self._gradcam_fix_transform
            }
        else: 
            self.explanation_parameters_gradcam = {
                'target_layers': [self.model[0].model.conv_head], # resulting size is 7 x 7
                'reshape_transform' : self._gradcam_fix_transform
            }

    def _gradcam_fix_transform(self, tensor): # remove top-right corner with average
        result = tensor
        result[:,:,0,6] = (torch.sum(result, dim=[2,3]) - result[:,:,0,6]) / (result.size(2) * result.size(3) - 1)
        return result