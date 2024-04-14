from src.models import Model
import torch
import timm

class EfficientNet_B4(Model):
    def __init__(self):
        super().__init__("EfficientNet_B4")

    def _build_model(self) -> torch.nn.Module:
        model = timm.create_model('efficientnet_b4.ra2_in1k', pretrained=True)
        print(f"Loaded model: https://huggingface.co/{model.default_cfg['hf_hub_id']}")
        return model
    
    def make_sure_is_initialized(self):
        if self.is_initialized is False:
            self.model = self._build_model()
            self.explanation_parameters_gradcam = {
                'target_layers': [self.model.conv_head], # resulting size is 7 x 7
                'reshape_transform' : self._gradcam_fix_transform
            }
            self.is_initialized = True

    def _gradcam_fix_transform(self, tensor): # remove top-right corner with average
        result = tensor
        result[:,:,0,6] = (torch.sum(result, dim=[2,3]) - result[:,:,0,6]) / (result.size(2) * result.size(3) - 1)
        return result