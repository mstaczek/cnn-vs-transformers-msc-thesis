from src.explanations import Explanation
from src.models import Model
import torch
import numpy as np

class ExplanationAttentionRollout(Explanation):
    def __init__(self, device: str = 'cpu'):
        super().__init__("AttentionRollout", device)

    def _compute_explanation(self, model: Model, images: torch.Tensor) -> torch.Tensor:
        explanations = self.attention_rollout(input_tensors=images.to(self.device))
        return explanations

    def _update_explanation_method(self, model: Model):        
        if model.name != "ViT_B_32":
            raise Exception("Attention rollout can only be used with the ViT_B_32 model.")
        self.attention_rollout = VITAttentionRollout(model.get_model(), discard_ratio=0.9, head_fusion='max')


# SOURCE: https://github.com/jacobgil/vit-explain/blob/main/vit_rollout.py
# accessed on: 2024.11.02
import torch
from scipy.ndimage import zoom

def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1, keepdim=True)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
        discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensors):
        self.attentions = []
        explanations = []
        for input_tensor in input_tensors:
            with torch.no_grad():
                output = self.model(input_tensor.unsqueeze(0))
            explanation = rollout(self.attentions, self.discard_ratio, self.head_fusion)
            self.attentions = []
            zoom_ratio = [a/b for a, b in zip(input_tensor.shape[1:], explanation.shape)]
            explanation = zoom(explanation, zoom_ratio, order=1) # linear interpolation
            explanations.append(explanation)
        explanations = np.stack(explanations, axis=0)
        return explanations