from src.models import Model
import torch
import timm

class ViT_B_32(Model):
    def __init__(self):
        super().__init__("ViT_B_32")

    def _build_model(self) -> torch.nn.Module:
        model = timm.create_model('vit_base_patch32_224.augreg_in21k_ft_in1k', pretrained=True)
        print(f"Loaded model: https://huggingface.co/{model.default_cfg['hf_hub_id']}")
        return model

    def make_sure_is_initialized(self):
        if self.is_initialized is False:
            self.model = self._build_model()
            self.explanation_parameters_gradcam = {
                'target_layers': [self.model.blocks[-2]], # one but last ViT Encoder Block
                'reshape_transform' : self._gradcam_reshape_transform
            }
            self.is_initialized = True

    def _gradcam_reshape_transform(self, tensor, height=7, width=7): # ViT-B/32 has tensor size BATCH x 50 x 768 at this layer
        result = tensor[:, 1:, :] # remove class token
        result = result.reshape(result.size(0), height, width, result.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result