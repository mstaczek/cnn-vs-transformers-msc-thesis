from src.models import Model
import torch
import timm

class MobileNetV3(Model):
    def __init__(self, **kwargs):
        super().__init__("MobileNetV3", **kwargs)

    def _load_model(self) -> torch.nn.Module:
        if self.pretrained_weights_name != 'imagenet':
            model = self._load_model_from_disk()
        if self.pretrained_weights_name == 'imagenet':
            model = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True)
            print(f"Loaded default imagenet-pretrained model: https://huggingface.co/{model.default_cfg['hf_hub_id']}")
        return model
    
    def _initialize_model(self):
        if self.pretrained_weights_name == 'imagenet':
            self.explanation_parameters_gradcam = {
                'target_layers': [self.model.blocks], # resulting size is 7 x 7
            }
        else: 
            self.explanation_parameters_gradcam = {
                'target_layers': [self.model[0].model.blocks], # resulting size is 7 x 7
            }
