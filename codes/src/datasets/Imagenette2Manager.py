from timm.data import resolve_model_data_config, create_transform
from src.models import Model
from torchvision import transforms
from src.datasets.DatasetManager import DatasetManager

class Imagenette2Manager(DatasetManager):
    def __init__(self, root_images: str, root_explanations: str):
        super(Imagenette2Manager, self).__init__('imagenette2', root_images, root_explanations)

    def _get_transforms(self, model: Model):
        try: 
            data_config = resolve_model_data_config(model)
            transforms = create_transform(**data_config, is_training=False)
            return transforms
        except:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.461], std=[0.219])
            ])