from torch import nn


class CircleModel_v0(nn.Module):
    """
       A simple feedforward neural network for binary classification
       on 2D datasets (e.g., sklearn's make_circles).

       Architecture:
           - Input layer: 2 features
           - Hidden layer 1: Linear(2 → 10) + ReLU
           - Hidden layer 2: Linear(10 → 12) + ReLU
           - Output layer: Linear(12 → 1), raw logits

       Forward Pass:
           x -> Linear(2,10) -> ReLU
             -> Linear(10,12) -> ReLU
             -> Linear(12,1)

       Notes:
           - Outputs logits (not probabilities).
           - Use `torch.sigmoid` during evaluation to convert logits to probabilities.
           - Designed for binary classification tasks with BCEWithLogitsLoss.
       """
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.relu =nn.ReLU()
        self.layer_2 = nn.Linear(in_features=10, out_features=12)

        self.layer_3 = nn.Linear(in_features=12, out_features=1)

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

