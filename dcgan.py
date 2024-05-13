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
from mmcv.utils import Registry

MODEL_REGISTRY = Registry('models')
class FashionMNISTDataset(Dataset):
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        self.images = data.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.float32) / 255.0
        self.images = np.expand_dims(self.images, axis=1)
        self.labels = data.iloc[:, 0].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        label = torch.tensor(self.labels[idx])
        return image, label


@MODEL_REGISTRY.register_module()
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2, padding=2),  # Output: 64 x 14 x 14
            nn.LeakyReLU(0.3),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # Output: 128 x 7 x 7
            nn.LeakyReLU(0.3),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)
        )

    def forward(self, x):
        return self.main(x)


@MODEL_REGISTRY.register_module()
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = nn.Linear(100, 7 * 7 * 256)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2),  # Output: 128 x 7 x 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),  # Output: 64 x 14 x 14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(64, 1, 5, stride=2, padding=2, output_padding=1),  # Output: 1 x 28 x 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.dense(x)
        x = x.view(-1, 256, 7, 7)
        return self.main(x)


def get_noise(n_samples, noise_dim, device):
    return torch.randn(n_samples, noise_dim, device=device)


def plot_metrics(discriminator_losses, generator_losses, discriminator_accuracies):
    plt.figure(figsize=(10, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(discriminator_losses, label="Discriminator Loss")
    plt.plot(generator_losses, label="Generator Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Generator and Discriminator Losses")

    # Plot discriminator accuracy
    plt.subplot(1, 2, 2)
    plt.plot(discriminator_accuracies, label="Discriminator Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Discriminator Accuracy")

    plt.tight_layout()
    plt.show()


train_data_path = 'fashion-mnist_train.csv'
train_dataset = FashionMNISTDataset(train_data_path)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

discriminator = Discriminator().to(device)
generator = Generator().to(device)
optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4)
optimizer_g = optim.Adam(generator.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()


# Training function
def train_dcgan(generator, discriminator, criterion, optimizer_g, optimizer_d, dataloader, epochs, device, noise_dim):
    generator.train()
    discriminator.train()

    checkpoint_dir = './ckpt'
    os.makedirs(checkpoint_dir, exist_ok=True)

    losses_g = []
    losses_d = []
    accuracies_d = []

    for epoch in range(epochs):
        epoch_losses_d = []
        epoch_losses_g = []
        epoch_accuracies_d = []

        loop = tqdm(dataloader, leave=True)
        for images, _ in loop:
            images = images.to(device)
            batch_size = images.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            optimizer_d.zero_grad()
            outputs_real = discriminator(images)
            loss_real = criterion(outputs_real, real_labels)

            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_images = generator(noise)
            outputs_fake = discriminator(fake_images.detach())
            loss_fake = criterion(outputs_fake, fake_labels)

            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            optimizer_d.step()

            # Calculate "accuracy"
            correct_real = (outputs_real > 0).float()
            correct_fake = (outputs_fake < 0).float()
            acc_d = (correct_real.mean() + correct_fake.mean()) / 2.0

            # Train Generator
            optimizer_g.zero_grad()
            outputs_fake_for_gen = discriminator(fake_images)
            loss_g = criterion(outputs_fake_for_gen, real_labels)
            loss_g.backward()
            optimizer_g.step()

            epoch_losses_d.append(loss_d.item())
            epoch_losses_g.append(loss_g.item())
            epoch_accuracies_d.append(acc_d.item())

            loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(loss_d=loss_d.item(), loss_g=loss_g.item(), acc_d=acc_d.item())

        losses_d.extend(epoch_losses_d)
        losses_g.extend(epoch_losses_g)
        accuracies_d.extend(epoch_accuracies_d)

        if epoch in [9, 29, 49]:
            torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f'generator_epoch_{epoch + 1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch + 1}.pth'))
            with torch.no_grad():
                fake_images = generator(torch.randn(16, noise_dim, device=device)).detach().cpu()
                plt.figure(figsize=(8, 8))
                plt.imshow(np.transpose(torchvision.utils.make_grid(fake_images, nrow=4, padding=2, normalize=True),
                                        (1, 2, 0)))
                plt.show()

    plot_metrics(losses_d, losses_g, accuracies_d)


if __name__ == '__main__':
    train_dcgan(generator, discriminator, criterion, optimizer_g, optimizer_d, train_loader, 50, device, 100)


