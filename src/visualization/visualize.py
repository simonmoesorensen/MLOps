import os
from torch import nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data.make_dataset import get_data
from models.predict_model import load_model
from sklearn.manifold import TSNE

cur_path = os.path.dirname(__file__)
vis_path = os.path.relpath('../../reports/figures/', cur_path) + '\\'


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract hidden layers without the output layer
        self.layers = nn.Sequential(*model.hidden_layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.layers(x)
        return out


def tSNE(model_path):
    model = load_model(model_path)
    feature_model = FeatureExtractor(model)

    _, test_set = get_data(cur_path + '/../../data')

    features = np.zeros((1, 64))
    targets = np.zeros((1, 1))

    for images, labels in test_set:
        feature = feature_model(images)
        targets = np.vstack([targets, labels.cpu().detach().numpy().reshape(labels.shape[0], 1)])
        features = np.vstack([features, feature.cpu().detach().numpy()])

    features = features[1:]
    targets = targets[1:].astype(int)
    print('Transforming features through TSNE')
    x_tsne = TSNE(n_components=2).fit_transform(features)

    print('Plotting t-sne plot')
    x_plot = np.hstack([x_tsne, targets])
    sns.scatterplot(x=x_plot[:, 0], y=x_plot[:, 1], hue=x_plot[:, 2], legend='full',
                    palette=sns.color_palette('muted'))
    plt.title('t-SNE plot for the final hidden layer (64 nodes)')
    print('Saving t-sne plot')
    plt.savefig(cur_path + '\\..\\..\\reports\\figures\\t-sne.png')


if __name__ == '__main__':
    model_path = os.path.relpath('../../models/model.pth', cur_path)
    tSNE(model_path)
