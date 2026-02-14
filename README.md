# ML Practice Implementations

A Python project for practicing machine learning implementations from scratch using PyTorch.

## Requirements

- Python >= 3.14
- PyTorch

## Installation

```bash
pip install torch
```

## Usage

Run the main script to execute the MLP XOR example:

```bash
python main.py
```

This will train a Multi-Layer Perceptron to learn the XOR gate function.

## Project Structure

```
ml/
├── main.py           # Entry point
├── mlp/              # Multi-Layer Perceptron module
│   ├── __init__.py
│   └── mlp.py        # MLP class and XOR example
└── pyproject.toml    # Project configuration
```

## What's Included

### Multi-Layer Perceptron (MLP)

A fully-connected feedforward neural network that learns the XOR function:

| Input 1 | Input 2 | Output |
| ------- | ------- | ------ |
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

The XOR problem demonstrates how neural networks can learn non-linearly separable patterns.
