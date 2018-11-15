import argparse
import logging
import os

import torch
import torchvision.datasets as datasets
from torch.optim import Adam
from torchvision import transforms

from vae.trainer import VAETrainer
from vae.vae import VAE, get_loss_function


def get_config():
    parser = argparse.ArgumentParser(description='Training DCGAN on CIFAR10')

    parser.add_argument('--log-root', type=str, default='./logs')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--log-name', type=str, default='train_vae.log')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train ')
    parser.add_argument('--image-size', type=int, default=32,
                        help='size of images to generate')
    parser.add_argument('--n_show_samples', type=int, default=32)
    parser.add_argument('--log_interval', type=int, default=10)
    config = parser.parse_args()
    config.cuda = not config.no_cuda and torch.cuda.is_available()

    return config


def main():
    config = get_config()
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_root,
                                             config.log_name)),
            logging.StreamHandler()],
        level=logging.INFO)

    transform = transforms.Compose([transforms.Scale(config.image_size), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(root=config.data_root, download=True,
                                     transform=transform, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=4, pin_memory=True)

    test_dataset = datasets.CIFAR10(root=config.data_root, download=True,
                                    transform=transform, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True,
                                              num_workers=4, pin_memory=True)

    vae = VAE(3, config.image_size, latent_size=64)

    trainer = VAETrainer(vae, train_loader, test_loader, Adam(vae.parameters(), lr=0.0002, betas=(0.5, 0.999)),
                         get_loss_function(3, config.image_size),
                         'cuda' if torch.cuda.is_available() and not config.no_cuda else 'cpu')

    trainer.train(config.epochs, config.n_show_samples, config.log_interval)


if __name__ == '__main__':
    main()
