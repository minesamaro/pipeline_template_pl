import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetSurvival(nn.Module):
    def __init__(self, in_channels=1, base_filters=64, latent_dim=128, input_size: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_filters * 4, base_filters * 8)

        # Latent layer (mu and logvar)
        # Dynamically infer the flattened size instead of assuming a fixed 512 input.
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            d1 = self.enc1(dummy)
            d2 = self.enc2(self.pool1(d1))
            d3 = self.enc3(self.pool2(d2))
            d4 = self.enc4(self.pool3(d3))
            enc_out_dim = d4.numel()
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, enc_out_dim)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4)
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_filters * 2, base_filters)

        # Output reconstruction
        self.reconstruction_head = nn.Conv2d(base_filters, 1, kernel_size=1)

        # Survival prediction branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.survival_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # For CoxPH (or 1 output + sigmoid for binary)
        )
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))

        # Latent space
        x_flat = self.flatten(x4)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        z = self.reparameterize(mu, logvar)

        # Decoder
        dec_input = self.fc_decode(z).view(x4.shape)
        x = self.up3(dec_input)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        reconstruction = self.reconstruction_head(x)

        # Survival branch
        pooled = self.global_pool(x4)  # Output from deepest encoder
        survival_output = self.survival_head(pooled)

        return survival_output, reconstruction, mu, logvar
