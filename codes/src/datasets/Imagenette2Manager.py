from timm.data import resolve_model_data_config, create_transform
from src.models import Model
from torchvision import transforms
from src.datasets.DatasetManager import DatasetManager

class Imagenette2Manager(DatasetManager):
    def __init__(self, root_images: str, root_explanations: str):
        super(Imagenette2Manager, self).__init__('imagenette2', root_images, root_explanations)

    def _get_transforms(self, model: Model):
        model_object = model.get_model()
        try:
            data_config = resolve_model_data_config(model_object)
            model_specific_transforms = create_transform(**data_config, is_training=False)
            normalization = [t for t in model_specific_transforms.transforms if isinstance(t, transforms.Normalize)][0]
        except:
            normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalization
        ])