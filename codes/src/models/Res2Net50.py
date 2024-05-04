from src.models import Model
import torch
import timm

class Res2Net50(Model):
    def __init__(self, **kwargs):
        super().__init__("Res2Net50", **kwargs)

    def _build_model(self) -> torch.nn.Module:
        if self.pretrained_weights_name != 'imagenet':
            model = self._load_model_from_disk()
        if self.pretrained_weights_name == 'imagenet':
            model = timm.create_model('res2net50_14w_8s.in1k', pretrained=True)
            print(f"Loaded default imagenet-pretrained model: https://huggingface.co/{model.default_cfg['hf_hub_id']}")
        return model
    
    def make_sure_is_initialized(self):
        if self.is_initialized is False:
            self.model = self._build_model()
            if self.pretrained_weights_name == 'imagenet':
                self.explanation_parameters_gradcam = {
                    'target_layers': [self.model.layer4[-1]], # resulting size is 7 x 7
                }
            else: 
                self.explanation_parameters_gradcam = {
                    'target_layers': [self.model[0].model.layer4[-1]], # resulting size is 7 x 7
                }
            self.is_initialized = True