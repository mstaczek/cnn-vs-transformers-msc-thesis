import torch

class Model:
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_initialized = False

    def _build_model(self) -> torch.nn.Module:
        raise NotImplementedError
    
    def make_sure_is_initialized(self):
        raise NotImplementedError
    
    def get_model(self) -> torch.nn.Module:
        if self.model is None:
            self.model = self._build_model()
        return self.model
    
    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        return self.get_model()(*args, **kwds)