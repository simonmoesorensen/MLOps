from models.train_model import train
import os

cur_dir = os.path.dirname(__file__)


def test_training():
    if os.path.exists(cur_dir + '../../models/unit_test_model.pth'):
        os.remove(cur_dir + '../../models/unit_test_model.pth')

    train(epochs=1, model_name='unit_test_model')

    assert os.path.exists(cur_dir + '../../models/unit_test_model.pth')
    os.remove(cur_dir + '../../models/unit_test_model.pth')
    assert not os.path.exists(cur_dir + '../../models/unit_test_model.pth')
