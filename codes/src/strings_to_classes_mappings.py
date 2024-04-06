from src.datasets import Imagenette2Manager
from src.models import ResNet18, ResNet50, ViT_B_32, Swin_T, DeiT_S, DeiT_T, EfficientNet_B3, EfficientNet_B4
from src.explanations import ExplanationGradCAM, ExplanationKernelSHAP

datasets_mapping = {
    'imagenette2' : Imagenette2Manager,
}

models_mapping = {
    'DeiT_S'          : DeiT_S,
    'DeiT_T'          : DeiT_T,
    'EfficientNet_B3' : EfficientNet_B3,
    'EfficientNet_B4' : EfficientNet_B4,
    'Swin_T'          : Swin_T,
    'ResNet18'        : ResNet18,
    'ResNet50'        : ResNet50,
    'ViT_B_32'        : ViT_B_32,
}

explanations_mapping = {
    'GradCAM' : ExplanationGradCAM,
    'KernelSHAP' : ExplanationKernelSHAP
}