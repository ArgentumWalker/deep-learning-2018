import logging
import os

import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


class VAETrainer:

    def __init__(self, model, train_loader, test_loader, optimizer,
                 loss_function, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.writer = SummaryWriter()

        self.model.to(self.device)

    def train(self, n_epoch=25, n_show_samples=16, log_interval=10):
        for epoch in range(n_epoch):
            self._train(epoch, log_interval)
            self._test(epoch, log_interval)
            self._plot_generated(epoch, n_show_samples)

    def _train(self, epoch, log_interval):
        self.model.train()
        epoch_loss = 0

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()

            recon_x, mu, logvar = self.model(data)
            train_loss = self.loss_function(recon_x, data, mu, logvar)
            train_loss.backward()

            self.optimizer.step()

            epoch_loss += train_loss
            norm_train_loss = train_loss / len(data)

            if batch_idx % log_interval == 0:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    norm_train_loss)
                logging.info(msg)

                batch_size = self.train_loader.batch_size
                train_size = len(self.train_loader.dataset)
                batches_per_epoch_train = train_size // batch_size
                self.writer.add_scalar(tag='data/train_loss',
                                       scalar_value=norm_train_loss,
                                       global_step=batches_per_epoch_train * epoch + batch_idx)

        epoch_loss /= len(self.train_loader.dataset)
        logging.info(f'====> Epoch: {epoch} Average loss: {epoch_loss:.4f}')
        self.writer.add_scalar(tag='data/train_epoch_loss',
                               scalar_value=epoch_loss,
                               global_step=epoch)

    def _test(self, epoch, log_interval):
        self.model.eval()
        test_epoch_loss = 0

        for batch_idx, (data, _) in enumerate(self.test_loader):
            data = data.to(self.device)

            recon_x, mu, logvar = self.model(data)
            test_loss = self.loss_function(recon_x, data, mu, logvar)
            test_epoch_loss += test_loss

            if batch_idx % log_interval == 0:
                msg = 'Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(self.test_loader.dataset),
                    100. * batch_idx / len(self.test_loader),
                    test_loss / len(data))
                logging.info(msg)

                batches_per_epoch_test = len(self.test_loader.dataset) // len(data)
                self.writer.add_scalar(tag='data/test_loss',
                                       scalar_value=test_loss / len(data),
                                       global_step=batches_per_epoch_test * (epoch - 1) + batch_idx)

        test_epoch_loss /= len(self.test_loader.dataset)
        logging.info('====> Test set loss: {:.4f}'.format(test_epoch_loss))
        self.writer.add_scalar(tag='data/test_epoch_loss',
                               scalar_value=test_epoch_loss,
                               global_step=epoch)

    def _plot_generated(self, epoch, show_samples):
        for data, _ in self.test_loader:
            data = data.to(self.device)
            recon, _, _ = self.model(data)
            recon = recon.view(-1, self.model.channels, self.model.image_size, self.model.image_size)

            x = vutils.make_grid(recon[:show_samples, :, :, :], normalize=True, scale_each=True)
            self.writer.add_image('img/recon', x, epoch)

            y = vutils.make_grid(data[:show_samples, :, :, :], normalize=True, scale_each=True)
            self.writer.add_image('img/real', y, epoch)

            break

    def save(self, checkpoint_path):
        dir_name = os.path.dirname(checkpoint_path)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
