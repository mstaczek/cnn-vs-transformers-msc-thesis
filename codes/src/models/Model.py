import torch
import os

class Model:
    def __init__(self, name: str, pretrained_weights_name: str = None, root_trained_models: str = None):
        self.name = name
        self.model = None
        self.is_initialized = False
        self.pretrained_weights_name = pretrained_weights_name if pretrained_weights_name is not None else 'imagenet'
        self.root_trained_models = root_trained_models + f'/{self.name}/' if root_trained_models is not None else f'../trained_models/{self.pretrained_weights_name}/{self.model}/'

    def _build_model(self) -> torch.nn.Module:
        raise NotImplementedError
    
    def make_sure_is_initialized(self):
        raise NotImplementedError
    
    def get_model(self) -> torch.nn.Module:
        if self.model is None:
            self.model = self._build_model()
        return self.model
    
    def move_to_device(self, device: str):
        self.model.to(device)
    
    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        return self.get_model()(*args, **kwds)

    def _load_model_from_disk(self) -> torch.nn.Module:
        try:
            filename = self.root_trained_models + os.listdir(self.root_trained_models)[-1]
            model = torch.load(filename, map_location='cpu')
            print(f"Loaded model: {filename}")
            print(model)
            return model
        except:
            print(f'Did not find weigths for {self.name} model pretrained on {self.pretrained_weights_name} in folder {self.root_trained_models}.')
            print(f"Defaulting to pretrained imagenet weights")
            self.pretrained_weights_name = 'imagenet'
            return None