import os

from src.models.train_model import train

root_dir = os.path.dirname(__file__)[:-6]


def test_training():
    if os.path.exists(root_dir + '/models/unit_test_model.pth'):
        os.remove(root_dir + '/models/unit_test_model.pth')

    train(epochs=1, model_name='unit_test_model')

    assert os.path.exists(root_dir + '/models/unit_test_model.pth')
    os.remove(root_dir + '/models/unit_test_model.pth')
    assert not os.path.exists(root_dir + '/models/unit_test_model.pth')
