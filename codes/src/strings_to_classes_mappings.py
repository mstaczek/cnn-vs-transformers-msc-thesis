from src.datasets import Imagenette2Manager
from src.models import *
from src.explanations import ExplanationGradCAM, ExplanationKernelSHAP, ExplanationIntegratedGradients, ExplanationAttentionRollout

datasets_mapping = {
    'imagenette2' : Imagenette2Manager,
}

models_mapping = {
    'ConvNeXtV2_Nano' : ConvNeXtV2_Nano,
    'DeiT_S'          : DeiT_S,
    'DeiT_T'          : DeiT_T,
    'DenseNet121'     : DenseNet121,
    'EfficientNet_B3' : EfficientNet_B3,
    'EfficientNet_B4' : EfficientNet_B4,
    'MobileNetV3'     : MobileNetV3,
    'PyramidViT_V2_B2': PyramidViT_V2_B2,
    'Res2Net50'       : Res2Net50,
    'ResNet18'        : ResNet18,
    'ResNet50'        : ResNet50,
    'ResNeXt50'       : ResNeXt50,
    'Swin_T'          : Swin_T,
    'ViT_B_32'        : ViT_B_32,
}

explanations_mapping = {
    'GradCAM' : ExplanationGradCAM,
    'IntegratedGradients' : ExplanationIntegratedGradients,
    'KernelSHAP' : ExplanationKernelSHAP,
    'AttentionRollout' : ExplanationAttentionRollout
}