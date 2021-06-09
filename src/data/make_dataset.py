# -*- coding: utf-8 -*-
import os

import click
import logging
from pathlib import Path

import torch
from dotenv import find_dotenv, load_dotenv

from torchvision import datasets, transforms


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    return get_data(input_filepath, True)


def get_data(input_filepath, download=False, batch_size=64):
    cur_dir = os.path.dirname(__file__)
    os.chdir(cur_dir)

    if not download and not (
            os.path.exists(f'{cur_dir}/{input_filepath}/MNIST/processed/test.pt') and
            os.path.exists(f'{cur_dir}/{input_filepath}/MNIST/processed/training.pt')):
        raise FileNotFoundError(f'Missing MNIST folder in {cur_dir}/{input_filepath}')

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    if download:
        logger.info('Downloading data...')

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    # Download and load the training data
    logger.info('Loading the training data')
    train = datasets.MNIST(cur_dir + '/' + input_filepath, download=download, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
    logger.info('Loading the test data')
    test = datasets.MNIST(cur_dir + '/' + input_filepath, download=download, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(test, batch_size, shuffle=True)

    return trainloader, testloader

