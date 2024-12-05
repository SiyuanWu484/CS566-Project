import torch, time, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F

# Generator class definition
class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, class_num=10):
        '''
        Initialize the Generator Network
        :param input_dim: Latent dimension size
        :param output_dim: Number of output channels, representing the generated image channels
        :param class_num: Number of different image categories (for conditional GAN)
        '''
        super(Generator, self).__init__()
        # The input is the concatenation of the latent dimension and condition vector
        self.input_dim = input_dim + class_num
        self.output_dim = output_dim

        # Fully connected layer to transform the latent input
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(),
        )

        # Deconvolutional layers to upsample to image size
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        # Flatten input and pass through fully connected layer
        x = self.fc(input)
        x = x.view(-1, 128, 7, 7)
        x = self.deconv(x)
        return x

# Discriminator class definition
class Discriminator(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        '''
        Initialize the Discriminator Network
        :param input_dim: Number of input channels (image channels)
        :param output_dim: Number of output channels (1 for binary classification)
        '''
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Convolutional layers to extract features
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        # Fully connected layers to classify whether image is real or fake
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            # nn.Sigmoid() can be applied to output for binary classification
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc(x)
        return x

# Image Generator class to manage training process
class ImageGenerator(object):
    def __init__(self):
        # Initialize hyperparameters, dataset, and networks
        self.epoch = 50
        self.sample_num = 100
        self.batch_size = 64
        self.z_dim = 100
        self.lr = 0.0002
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_dataloader()
        self.G = Generator(input_dim=self.z_dim, output_dim=self.output_dim, class_num=0).to(self.device)
        self.D = Discriminator(input_dim=self.output_dim, output_dim=1).to(self.device)
        self.initialize_weights(self.G)
        self.initialize_weights(self.D)
        self.optimizerG = optim.RMSprop(self.G.parameters(), lr=self.lr)
        self.optimizerD = optim.RMSprop(self.D.parameters(), lr=self.lr)
        self.c = 0.01
        self.n_critic = 5
        self.fixed_z = torch.rand((self.sample_num, self.z_dim)).to(self.device)

    def initialize_weights(self, net):
        # Initialize weights for the neural networks
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def init_dataloader(self):
        # Initialize dataset and dataloader
        tf = transforms.Compose([
            transforms.ToTensor(),
            # Optionally normalize: transforms.Normalize((0.1307,), (0.3081,)) 
        ])
        train_dataset = MNIST('./data/', train=True, download=True, transform=tf)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_dataset = MNIST('./data/', train=False, download=True, transform=tf)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.output_dim = self.train_dataloader.__iter__().__next__()[0].shape[1]

    def train(self):
        # Train the GAN network
        print('Training Started!')
        for epoch in range(self.epoch):
            self.G.train()
            self.D.train()
            loss_mean_G = 0
            loss_mean_D = 0
            for i, (images, labels) in enumerate(self.train_dataloader):
                z = torch.rand((self.batch_size, self.z_dim)).to(self.device)
                images, labels = images.to(self.device), labels.to(self.device)

                ### Update Discriminator ###
                self.optimizerD.zero_grad()
                D_real = self.D(images)
                D_real_loss = -torch.mean(D_real)
                images_fake = self.G(z)
                D_fake = self.D(images_fake)
                D_fake_loss = torch.mean(D_fake)

                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()
                self.optimizerD.step()
                loss_mean_D += D_loss.item()

                # Clipping D's parameters
                for p in self.D.parameters():
                    p.data.clamp_(-self.c, self.c)

                ### Update Generator ###
                if (i+1) % self.n_critic == 0:
                    self.optimizerG.zero_grad()
                    images_fake = self.G(z)
                    D_fake = self.D(images_fake)
                    G_loss = -torch.mean(D_fake)
                    G_loss.backward()
                    self.optimizerG.step()
                    loss_mean_G += G_loss.item()

            train_loss_G = loss_mean_G / len(self.train_dataloader) * self.n_critic
            train_loss_D = loss_mean_D / len(self.train_dataloader)
            print(f'Epoch:{epoch}, Training Loss G:{train_loss_G:.4f}, Loss D:{train_loss_D:.4f}')
            self.visualize_results(epoch)

    @torch.no_grad()
    def visualize_results(self, epoch):
        # Visualize and save the generated images
        self.G.eval()
        output_path = 'results/GAN'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        tot_num_samples = self.sample_num
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        z = self.fixed_z
        generated_images = self.G(z)
        save_image(generated_images, os.path.join(output_path, f'{epoch}.jpg'), nrow=image_frame_dim)


if __name__ == '__main__':
    generator = ImageGenerator()
    generator.train()

