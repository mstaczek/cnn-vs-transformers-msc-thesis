from src.datasets.Imagenette2Manager import Imagenette2Manager
from src.models.ResNet18 import ResNet18
from src.explanations.ExplanationGradCAM import ExplanationGradCAM

datasets_mapping = {
    'imagenette2' : Imagenette2Manager,
}

models_mapping = {
    'ResNet18' : ResNet18,
}

explanations_mapping = {
    'GradCAM' : ExplanationGradCAM
}