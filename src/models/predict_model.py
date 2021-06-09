import os

import torch

from data.make_dataset import get_data
from models.model import WorldsBestModel

cur_path = os.path.dirname(__file__)
data_path = os.path.relpath('data', cur_path)


def get_predictions(model_path, data):
    model = load_model(model_path)
    tensor_data = torch.tensor(data)

    ps = torch.exp(model(tensor_data))
    _, top_class = ps.topk(1, dim=1)
    return top_class


def evaluate(model_path):
    model = load_model(model_path)

    _, test_set = get_data(data_path)

    # turn off gradients
    with torch.no_grad():
        # validation pass here
        accuracy = 0
        for images, labels in test_set:
            ps = torch.exp(model(images))
            _, top_class = ps.topk(1, dim=1)

            equals = top_class == labels.view(*top_class.shape)
            accuracy += equals.float().mean().item()

        print(f'Test set accuracy: {accuracy / len(test_set)}%')


def load_model(model_path):
    print(f'Loading model {model_path}')
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    model = WorldsBestModel(checkpoint['n_input'], checkpoint['n_output'],
                            checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    # Evaluation mode
    model.eval()
    return model
