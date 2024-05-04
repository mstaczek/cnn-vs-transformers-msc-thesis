import torch

class Model:
    def __init__(self, name: str, pretrained_weights_name: str = None):
        self.name = name
        self.model = None
        self.is_initialized = False
        self.pretrained_weights_name = pretrained_weights_name if pretrained_weights_name is not None else 'imagenet'

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