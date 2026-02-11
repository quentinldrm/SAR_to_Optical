"""
Pix2Pix Architecture for SAR-to-Optical Translation

Implementation based on Isola et al. (2017)
"Image-to-Image Translation with Conditional Adversarial Networks"

Components:
- U-Net Generator: 5 channels (SAR + S2 Cloudy) → 3 channels (S2 Clear)
- PatchGAN Discriminator: 8 channels → 30×30 classification grid
"""

import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    """
    U-Net Generator with encoder-decoder architecture.

    Optimizations:
    - No BatchNorm on the first layer (preserves raw input statistics)
    - Dropout(0.5) applied in the first 3 decoder blocks for regularization
    - Tanh activation for output scaled to [-1, 1]

    Args:
        in_channels (int): Number of input channels (default: 5)
        out_channels (int): Number of output channels (default: 3)
    """

    def __init__(self, in_channels=5, out_channels=3):
        super(UNetGenerator, self).__init__()

        # Encoder: Progressive downsampling
        self.down1 = self.conv_block(in_channels, 64, normalize=False)
        self.down2 = self.conv_block(64, 128)
        self.down3 = self.conv_block(128, 256)
        self.down4 = self.conv_block(256, 512)
        self.down5 = self.conv_block(512, 512)

        # Decoder: Upsampling with skip connections
        # Note: Input features double in some layers due to torch.cat skip connections
        self.up1 = self.up_block(512, 512, dropout=True)
        self.up2 = self.up_block(1024, 256, dropout=True)
        self.up3 = self.up_block(512, 128, dropout=True)
        self.up4 = self.up_block(256, 64)

        # Final Layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def conv_block(self, in_f, out_f, normalize=True):
        """
        Convolutional block for the encoder.

        Args:
            in_f (int): Number of input features
            out_f (int): Number of output features
            normalize (bool): Apply BatchNorm (default: True)

        Returns:
            nn.Sequential: Layer sequence
        """
        layers = [
            nn.Conv2d(in_f, out_f, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_f))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def up_block(self, in_f, out_f, dropout=False):
        """
        Deconvolutional block for the decoder.

        Args:
            in_f (int): Number of input features
            out_f (int): Number of output features
            dropout (bool): Apply Dropout(0.5) (default: False)

        Returns:
            nn.Sequential: Layer sequence
        """
        layers = [
            nn.ConvTranspose2d(
                in_f, out_f, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_f),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass with U-Net skip connections.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W)
        """
        # Encoder pass
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # Decoder pass with skip connections (Concatenation along channel dim)
        u1 = self.up1(d5)
        u1 = torch.cat([u1, d4], dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)

        output = self.final(u4)

        return output


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator with a 70×70 pixel receptive field.

    Architecture:
    - Classifies local patches rather than the entire image.
    - Promotes local texture consistency and high-frequency detail.
    - Output: NxN grid of classification scores.

    Args:
        in_channels (int): Number of input channels (default: 8)
    """

    def __init__(self, in_channels=8):
        super(PatchGANDiscriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 256)
        self.conv4 = self.conv_block(256, 512, stride=1)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def conv_block(self, in_f, out_f, stride=2):
        """
        Standard convolutional block for discriminator.
        """
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_f),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input_img, output_img):
        """
        Forward pass of the discriminator.
        Concatenates the condition (input_img) and the target (output_img).
        """
        x = torch.cat([input_img, output_img], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


def test_generator():
    """Test utility for the U-Net Generator."""
    print("=" * 60)
    print("TEST: U-Net Generator")
    print("=" * 60)

    model = UNetGenerator(in_channels=5, out_channels=3)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Parameters : {total_params:,}")

    batch_size = 4
    x = torch.randn(batch_size, 5, 256, 256)

    with torch.no_grad():
        output = model(x)

    print(f"Input      : {x.shape}")
    print(f"Output     : {output.shape}")
    print(f"Range      : [{output.min():.3f}, {output.max():.3f}]")

    assert output.shape == (batch_size, 3, 256, 256), "Incorrect Output Shape!"
    print("\n✓ Generator OK\n")


def test_discriminator():
    """Test utility for the PatchGAN Discriminator."""
    print("=" * 60)
    print("TEST: PatchGAN Discriminator")
    print("=" * 60)

    model = PatchGANDiscriminator(in_channels=8)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Parameters : {total_params:,}")

    batch_size = 4
    input_img = torch.randn(batch_size, 5, 256, 256)
    output_img = torch.randn(batch_size, 3, 256, 256)

    with torch.no_grad():
        prediction = model(input_img, output_img)

    print(f"Input      : {input_img.shape}")
    print(f"Output     : {output_img.shape}")
    print(f"Prediction : {prediction.shape}")
    print(f"  → Grid of {prediction.shape[2]}×{prediction.shape[3]} patches")
    print(f"  → Each patch receptive field ≈ 70×70 pixels")
    print(f"Range      : [{prediction.min():.3f}, {prediction.max():.3f}]")

    assert prediction.shape == (batch_size, 1, 30, 30), "Incorrect Prediction Shape!"
    print("\n✓ Discriminator OK\n")


def test_pix2pix():
    """Complete validation of the Pix2Pix architecture."""
    print("\n" + "=" * 60)
    print("FULL SYSTEM TEST: Pix2Pix Architecture")
    print("=" * 60 + "\n")

    test_generator()
    test_discriminator()

    print("=" * 60)
    print("✓ Pix2Pix Architecture Validated")
    print("=" * 60)


if __name__ == "__main__":
    test_pix2pix()
