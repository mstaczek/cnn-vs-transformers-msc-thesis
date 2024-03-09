from src.datasets.Imagenette2Manager import Imagenette2Manager
from src.models import ResNet18, ResNet50, ViT_B_32, Swin_T
from src.explanations import ExplanationGradCAM, ExplanationKernelSHAP

datasets_mapping = {
    'imagenette2' : Imagenette2Manager,
}

models_mapping = {
    'Swin_T'   : Swin_T,
    'ResNet18' : ResNet18,
    'ResNet50' : ResNet50,
    'ViT_B_32' : ViT_B_32,
}

explanations_mapping = {
    'GradCAM' : ExplanationGradCAM,
    'KernelSHAP' : ExplanationKernelSHAP
}