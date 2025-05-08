import torch
import torch.nn as nn

bottleneck_size = 256
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Input: 64x64x3, Output: 32x32x16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output: 16x16x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output: 8x8x64
        )

        # Bottleneck (Flatten and Linear Layers)
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(8 * 8 * 64, bottleneck_size)  # Starting bottleneck size of 128
        self.fc_decode = nn.Linear(bottleneck_size, 8 * 8 * 64)
        self.unflatten = nn.Unflatten(1, (64, 8, 8))

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: 16x16x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: 32x32x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 64x64x3
            nn.Sigmoid()  # Output pixel values should be in the range [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        batch_size = encoded.size(0)
        encoded = self.flatten(encoded)
        bottleneck = self.fc_encode(encoded)
        decoded = self.fc_decode(bottleneck)
        decoded = self.unflatten(decoded)
        decoded = self.decoder(decoded)
        return decoded, bottleneck # Return both reconstructed output and the bottleneck representation

# Instantiate the model
model = ConvAutoencoder()
print(model)

import torch
import torch.nn as nn

class ConvAutoencoderV2(nn.Module):
    def __init__(self, bottleneck_size=128):
        super(ConvAutoencoderV2, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 8x8 -> 4x4
        )

        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(4 * 4 * 256, bottleneck_size)
        self.fc_decode = nn.Linear(bottleneck_size, 4 * 4 * 256)
        self.unflatten = nn.Unflatten(1, (256, 4, 4))

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 4x4 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # 16x16 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # 32x32 -> 64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        batch_size = encoded.size(0)
        encoded = self.flatten(encoded)
        bottleneck = self.fc_encode(encoded)
        decoded = self.fc_decode(bottleneck)
        decoded = self.unflatten(decoded)
        decoded = self.decoder(decoded)
        return decoded, bottleneck

# Instantiate the new model
model_v2 = ConvAutoencoderV2()
print(model_v2)


import torch
import torch.nn as nn

class ConvAutoencoderV3(nn.Module):
    def __init__(self, bottleneck_size=32):
        super(ConvAutoencoderV3, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 8x8 -> 4x4
        )

        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(4 * 4 * 256, bottleneck_size)
        self.fc_decode = nn.Linear(bottleneck_size, 4 * 4 * 256)
        self.unflatten = nn.Unflatten(1, (256, 4, 4))

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 4x4 -> 8x8
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # 16x16 -> 32x32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # 32x32 -> 64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        batch_size = encoded.size(0)
        encoded = self.flatten(encoded)
        bottleneck = self.fc_encode(encoded)
        decoded = self.fc_decode(bottleneck)
        decoded = self.unflatten(decoded)
        decoded = self.decoder(decoded)
        return decoded, bottleneck

# Instantiate the new model (you'll need to update your training script to use this class)
model_v3 = ConvAutoencoderV3()
print(model_v3)