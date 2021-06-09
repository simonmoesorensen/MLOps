import pytest
import torch

from src.data.make_dataset import get_data


def test_data():
    train, test = get_data('../../data')
    assert len(train.dataset) == 60000
    assert len(test.dataset) == 10000
    img, lbl = next(iter(train))
    assert img[0].shape == torch.Size([1, 28, 28])
    assert sum(lbl.unique() - torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])) == 0


def test_data_missing_file():
    with pytest.raises(FileNotFoundError):
        get_data('data')
