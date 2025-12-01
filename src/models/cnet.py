import torch.nn as nn
import torch.nn.functional as F


class ClassifierModel(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int] = (28, 28),
        num_classes: int = 10,
        num_hidden_layers: int = 3,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        input_dim = img_size[0] * img_size[1]
        self.flatten = nn.Flatten()
        self.num_hidden_layers = max(0, num_hidden_layers)

        if self.num_hidden_layers == 0:
            self.output_layer = nn.Linear(input_dim, num_classes)
            self.input_layer = None
            self.hidden_layers = nn.ModuleList()
        else:
            self.input_layer = nn.Linear(input_dim, hidden_dim)
            self.hidden_layers = nn.ModuleList(
                nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_hidden_layers - 1)
            )
            self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        if self.num_hidden_layers == 0:
            return self.output_layer(x)

        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)
