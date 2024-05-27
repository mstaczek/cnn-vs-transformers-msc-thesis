from src.models import Model
import torch
import timm

class ViT_B_32(Model):
    def __init__(self, **kwargs):
        super().__init__("ViT_B_32", **kwargs)

    def _load_model(self) -> torch.nn.Module:
        if self.pretrained_weights_name != 'imagenet':
            model = self._load_model_from_disk()
        if self.pretrained_weights_name == 'imagenet':
            model = timm.create_model('vit_base_patch32_224.augreg_in21k_ft_in1k', pretrained=True)
            print(f"Loaded default imagenet-pretrained model: https://huggingface.co/{model.default_cfg['hf_hub_id']}")
        return model

    def _initialize_model(self):
        if self.pretrained_weights_name == 'imagenet':
            self.explanation_parameters_gradcam = {
                'target_layers': [self.model.blocks[-2]], # one but last ViT Encoder Block
                'reshape_transform' : self._gradcam_reshape_transform
            }
        else: 
            self.explanation_parameters_gradcam = {
                'target_layers': [self.model[0].model.blocks[-2]], # one but last ViT Encoder Block
                'reshape_transform' : self._gradcam_reshape_transform
            }

    def _gradcam_reshape_transform(self, tensor, height=7, width=7): # ViT-B/32 has tensor size BATCH x 50 x 768 at this layer
        result = tensor[:, 1:, :] # remove class token
        result = result.reshape(result.size(0), height, width, result.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result