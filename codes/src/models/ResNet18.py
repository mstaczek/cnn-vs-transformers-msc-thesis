from src.models import Model
import torch
import timm

class ResNet18(Model):
    def __init__(self):
        super().__init__("ResNet18")

    def _build_model(self) -> torch.nn.Module:
        model = timm.create_model('resnet18d.ra2_in1k', pretrained=True)
        print(f"Loaded model: https://huggingface.co/{model.default_cfg['hf_hub_id']}")
        return model
    
    def make_sure_is_initialized(self):
        if self.is_initialized is False:
            self.model = self._build_model()
            self.explanation_parameters_gradcam = {
                'target_layers': [self.model.layer4[-1]], # resulting size is 7 x 7
            }
            self.is_initialized = True