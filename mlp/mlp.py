"""
Multi-Layer Perceptron (MLP) module.

This module contains implementations of Multi-Layer Perceptron neural networks
and training examples demonstrating their capabilities on classic problems like XOR.
"""

from typing import override

import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    layer1: nn.Linear
    relu: nn.ReLU
    layer2: nn.Linear

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """
        An MLP is a composition of affine transformations and non-linearities.
        """
        super().__init__()

        # First Affine Transformation: R^2 -> R^4
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        # Non-linear Activation (ReLU)
        self.relu = nn.ReLU()
        # Second Affine Transformation: R^4 -> R^1
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # f(x) = L2(ReLU(L1(x)))
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def go() -> None:
    # 2. Prepare Data (The XOR Gate)
    # XOR inputs: (0,0), (0,1), (1,0), (1,1)
    X: torch.Tensor = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32
    )
    # Expected Outputs: 0, 1, 1, 0
    Y: torch.Tensor = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    # Initialize Model, Loss, and Optimizer
    model: MLP = MLP(input_dim=2, hidden_dim=4, output_dim=1)
    criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=0.05)

    print("\nStarting training...")
    epoch: int
    for epoch in range(1001):
        # Zero out the gradients (no accumulation)
        optimizer.zero_grad()

        # Forward Pass: Predict Y based on X
        predictions: torch.Tensor = model(X)

        # Calculate Loss: Scalar distance between prediction and truth
        loss: torch.Tensor = criterion(predictions, Y)

        # Backward Pass: Calculate dLoss/dWeight for every weight
        loss.backward()

        # Step: Update weights (W = W - lr * grad)
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

    # Verification
    print("\nFinal Results:")
    with torch.no_grad():  # Don't track gradients during inference
        final_output: torch.Tensor = torch.sigmoid(
            model(X)
        )  # Apply sigmoid to get 0-1 range
        i: int
        for i in range(len(X)):
            print(
                f"Input: {X[i].tolist()} -> Target: {Y[i].item()} | Predicted: {final_output[i].item():.4f}"
            )
        print()
