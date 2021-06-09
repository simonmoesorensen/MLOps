import pytest
import torch

from models.model import WorldsBestModel


@pytest.mark.parametrize('input, out, layers', [(12, 4, [6]), (784, 1, [256, 64])])
def test_model(input, out, layers):
    model = WorldsBestModel(input, out, layers)
    X = torch.randn(input, 1)

    output = model(X.T)
    assert len(output[0]) == out


def test_model_input_error():
    with pytest.raises(ValueError):
        model = WorldsBestModel(64, 5, [32, 12])
        X = torch.randn(63, 1)

        model(X.T)


def test_model_no_hidden_layers():
    with pytest.raises(ValueError):
        model = WorldsBestModel(12, 4, [])
