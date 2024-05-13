import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# 超参数设置
lr = 1e-5
batch_size = 64
epochs = 50
noise_dim = 100
discriminator_steps = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载路径
train_data_path = 'fashion-mnist_train.csv'
train_dataset = FashionMNISTDataset(train_data_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 判别器类
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.model = nn.Sequential(
            nn.LeakyReLU(0.3),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        print(x.shape)
        return self.model(x)

# 生成器类
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 7 * 7 * 128),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, output_padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.model(x)
        return output

discriminator = Discriminator().to(device)
generator = Generator().to(device)

criterion = nn.BCELoss()
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
optimizer_G = optim.Adam(generator.parameters(), lr=lr)

def generate_noise(size):
    return torch.randn(size, noise_dim).to(device)

# 初始化保存损失和准确率的列表
discriminator_losses = []
generator_losses = []
discriminator_accuracies = []

# 训练循环
for epoch in range(epochs):
    correct_predictions = 0
    total_predictions = 0
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
        real_labels = torch.ones(images.size(0), 1).to(device)
        fake_labels = torch.zeros(images.size(0), 1).to(device)

        # 训练判别器多次
        for _ in range(discriminator_steps):
            outputs = discriminator(images)
            loss_real = criterion(outputs, real_labels)
            correct_predictions += torch.sum(outputs > 0.5).item()
            total_predictions += images.size(0)

            noise = generate_noise(images.size(0))
            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            loss_fake = criterion(outputs, fake_labels)
            correct_predictions += torch.sum(outputs < 0.5).item()
            total_predictions += fake_images.size(0)

            loss_D = loss_real + loss_fake
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

        # 训练生成器
        noise = generate_noise(images.size(0))
        fake_images = generator(noise)
        outputs = discriminator(fake_images)
        loss_G = criterion(outputs, real_labels)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        discriminator_losses.append(loss_D.item())
        generator_losses.append(loss_G.item())

    # 计算准确率
    accuracy = correct_predictions / total_predictions
    discriminator_accuracies.append(accuracy)

    print(f'Epoch [{epoch + 1}/{epochs}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}, Accuracy: {accuracy:.2f}')

    if epoch + 1 in [10, 30, 50]:
        with torch.no_grad():
            noise = generate_noise(batch_size)
            fake_images = generator(noise)
            save_image(fake_images, f'dcgan_generated_{epoch + 1}.png', normalize=True)

# 绘制损失曲线和准确率曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(discriminator_losses, label="Discriminator Loss")
plt.plot(generator_losses, label="Generator Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("Generator and Discriminator Losses")

plt.subplot(1, 2, 2)
plt.plot(discriminator_accuracies, label="Discriminator Accuracy", color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Discriminator Accuracy")

plt.tight_layout()
plt.show()
