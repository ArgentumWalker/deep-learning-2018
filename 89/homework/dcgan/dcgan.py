import torch.nn as nn


class DCGenerator(nn.Module):
    """Assume that latent space size is 100 and image size is power of 2"""
    def __init__(self, latent_size, image_size, channels_alpha=6):
        super(DCGenerator, self).__init__()
        layers_count = _calculate_pow(image_size[1]) - 2
        channels_count = 2 ** (layers_count + channels_alpha)
        layers = [nn.ConvTranspose2d(latent_size, channels_count, 4, bias=False),
                  nn.BatchNorm2d(channels_count), nn.ReLU(inplace=True)]
        for _ in range(layers_count):
            layers.append(nn.ConvTranspose2d(channels_count, channels_count // 2, 5, 2, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(channels_count // 2))
            layers.append(nn.ReLU(inplace=True))
            channels_count //= 2
        layers[-3] = nn.ConvTranspose2d(2 * channels_count, image_size[0], 5, 2, 2, 1, bias=False)
        layers.pop(-2)
        layers[-1] = nn.Tanh()
        self._content = nn.Sequential(*layers)

    def forward(self, data):
        return self._content.forward(data)


class DCDiscriminator(nn.Module):
    def __init__(self, image_size, channels_alpha=6):
        super(DCDiscriminator, self).__init__()
        layers_count = _calculate_pow(image_size[1])
        channels_count = 2 ** channels_alpha
        layers = [nn.Conv2d(image_size[0], channels_count, 1), nn.BatchNorm2d(channels_count), nn.LeakyReLU(inplace=True)]
        for _ in range(layers_count):
            layers.append(nn.Conv2d(channels_count, channels_count * 2, 3, 2, 1))
            layers.append(nn.BatchNorm2d(channels_count * 2))
            layers.append(nn.LeakyReLU(inplace=True))
            channels_count *= 2
        layers[-3] = nn.Conv2d(channels_count // 2, 1, 3, 2, 1)
        layers.pop(-2)
        layers[-1] = nn.Sigmoid()
        self._content = nn.Sequential(*layers)

    def forward(self, data):
        return self._content.forward(data)


def _calculate_pow(num):
    power = 0
    current = 1
    while current * 2 <= num:
        power += 1
        current *= 2
    return power
