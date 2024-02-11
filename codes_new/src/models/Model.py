import torch

class Model:
    def __init__(self, name: str):
        self.name = name
        self.model = self._build_model()

    def _build_model(self) -> torch.nn.Module:
        raise NotImplementedError