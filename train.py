from mmcv.utils import Registry
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from safetensors.torch import save_model
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dcgan import Generator, Discriminator, FashionMNISTDataset, train_dcgan, MODEL_REGISTRY


def main(config):
    train_data_path = 'fashion-mnist_train.csv'
    train_dataset = FashionMNISTDataset(train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    discriminator = MODEL_REGISTRY.build(config['discriminator']).to(device)
    generator = MODEL_REGISTRY.build(config['generator']).to(device)
    print(discriminator)
    # optimizer_d = optim.Adam(discriminator.parameters(), lr=config['train']['lr_d'])
    # optimizer_g = optim.Adam(generator.parameters(), lr=config['train']['lr_g'])
    # criterion = nn.BCEWithLogitsLoss()
    #
    # train_dcgan(generator, discriminator, criterion, optimizer_g, optimizer_d, train_loader,
    #             config['train']['epochs'], device, config['train']['noise_dim'])


if __name__ == '__main__':
    config_dc = {
        'discriminator': {
            'type': 'Discriminator'
        },
        'generator': {
            'type': 'Generator'
        },
        'train': {
            'batch_size': 256,
            'epochs': 50,
            'noise_dim': 100,
            'lr_d': 1e-4,
            'lr_g': 1e-4
        }
    }

    main(config_dc)
