import datetime
import os
import time
import numpy as np

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from src.data.make_dataset import get_data
from src.models.model import WorldsBestModel

import matplotlib.pyplot as plt

cur_path = os.path.dirname(__file__)
data_path = os.path.relpath('data', cur_path)

writer = SummaryWriter(log_dir=cur_path + '/runs/')


def train(lr=0.003, epochs=50, model_name='model'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define architecture
    n_input = 784
    n_output = 10
    batch_size = 256
    hidden_layers = [256, 64]

    # Get model and data
    model = WorldsBestModel(n_input, n_output, hidden_layers)
    if torch.cuda.is_available():
        print('Moving to GPU')
        model = model.to(device)

    print(f'Got model: \n{model}')
    train_set, _ = get_data('/../../data', batch_size=batch_size)

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print('Training model...')

    # Run training loop
    print_every = 50
    steps = 0
    train_losses = []
    time1 = time.time()
    for e in range(epochs):
        model.train()

        running_loss = 0
        tot_loss = 0
        for images, labels in train_set:
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)

            steps += 1

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tot_loss += loss.item()

            writer.add_graph(model, images)

            if steps % print_every == 0:
                print("Epoch: {}/{}.. Step: {}/{}".format(e + 1, epochs, steps, len(train_set)),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every))

                running_loss = 0

        if steps % print_every != 0:
            print("Epoch: {}/{}.. Step: {}/{}".format(e + 1, epochs, steps, len(train_set)),
                  "Training Loss: {:.3f}.. ".format(running_loss / print_every))

        train_losses.append(tot_loss / len(train_set))
        steps = 0

        # Tensorboard
        writer.add_scalar('Loss/train', train_losses[-1], e)
        writer.add_histogram('Loss/train/hist', train_losses[-1], e)

    # hyperparams
    time_to_train = '{:.3f}'.format(time.time() - time1)
    writer.add_hparams({
        'epochs': epochs,
        'learning_rate': lr,
        'model_name': model_name,
        'device': str(device),
        'batch_size': batch_size,
        'n_input': n_input,
        'hidden_layers': str(hidden_layers),
        'n_output': n_output
    },
        {
            'train_loss': np.array(train_losses)
        })

    print(f'Time to train: {time_to_train}')
    # Save model
    print('Saving model')
    checkpoint = {'n_input': n_input,
                  'n_output': n_output,
                  'hidden_layers': hidden_layers,
                  'state_dict': model.state_dict()}

    full_model_pth = cur_path[0:-11] + f'\\models\\{model_name}.pth'
    print(f'Saving model @ {full_model_pth}')
    torch.save(checkpoint, full_model_pth)
    print('Model saved')

    # Plot training losses
    print('Plotting training curve')
    plt.plot(train_losses)
    plt.title('Training loss pr epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.legend(['Training set'])

    strftime = datetime.datetime.now().strftime('%y-%m-%d_%H%M%S')
    path_ext = '/reports/figures/training_curve_{}.png'.format(strftime)
    fig_path = cur_path[0:-11] + path_ext
    print(f'Saving figure @ {fig_path}')
    plt.savefig(fig_path)


if __name__ == '__main__':
    train(epochs=2, model_name='sample_model.pth')
