import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
# from safetensors.torch import load_model

from dcgan import Generator, Discriminator


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    return model

def imshow(img):
    img = img / 2 + 0.5     # 反标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    plt.show()

class GeneratorWithExtraLayer(Generator):
    def __init__(self):
        super(GeneratorWithExtraLayer, self).__init__()
        # 添加新的层
        self.extra_layer = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),  # 示例：添加额外的卷积层
            nn.ReLU(inplace=True)
        )
        # 初始化新层的参数
        for param in self.extra_layer.parameters():
            nn.init.zeros_(param)  # 将权重初始化为0，通常不推荐

    def forward(self, x):
        x = self.main(x)
        x = self.extra_layer(x)  # 通过新层传递输出
        return x


# 创建设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 初始化模型
model = GeneratorWithExtraLayer().to(device)
discriminator = Discriminator().to(device)
# print(device.type)
# 加载预训练权重
model = load_model(model, './ckpt/generator_epoch_50.pth', device=device)
# discriminator = load_model(discriminator, './ckpt/discriminator_epoch_50.pth', device)
print(model)
# 生成随机噪声作为生成器的输入
noise = torch.randn(1, 256, 28, 28, device=device)  # 假设噪声维度是100
print(noise.shape)
# 使用生成器生成图像
with torch.no_grad():  # 关闭梯度计算
    fake_image = model(noise)
    # 处理fake_image或显示图像等

if fake_image.dim() == 4 and fake_image.size(1) == 1:
    # 移动至CPU，转换为plt可用的格式
    image_to_show = fake_image.squeeze(0).cpu()  # 去除批处理维度，保持[1, 28, 28]
    imshow(image_to_show)
else:
    print("Invalid image dimensions.")