from src.models import Model
import torch
import timm

class PyramidViT_V2_B2(Model):
    def __init__(self):
        super().__init__("PyramidViT_V2_B2")

    def _build_model(self) -> torch.nn.Module:
        model = timm.create_model('pvt_v2_b2.in1k', pretrained=True)
        print(f"Loaded model: https://huggingface.co/{model.default_cfg['hf_hub_id']}")
        return model
    
    def make_sure_is_initialized(self):
        if self.is_initialized is False:
            self.model = self._build_model()
            self.explanation_parameters_gradcam = {
                'target_layers': [self.model.stages[-1].blocks[-1]], # resulting size is 7 x 7
                'reshape_transform' : self._gradcam_reshape_transform
            }
            self.is_initialized = True
            
    def _gradcam_reshape_transform(self, tensor, height=7, width=7): # PVT_V2_B has tensor size BATCH x 49 x 512 at this layer
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result