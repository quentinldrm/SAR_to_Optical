"""
Architecture Pix2Pix pour traduction SAR vers Optique

Implémentation basée sur Isola et al. (2017)
"Image-to-Image Translation with Conditional Adversarial Networks"

Composants :
- Générateur U-Net : 5 canaux (SAR + S2 Cloudy) → 3 canaux (S2 Clear)
- Discriminateur PatchGAN : 8 canaux → grille de classification 30×30
"""

import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    """
    Générateur U-Net avec architecture encodeur-décodeur.

    Optimisations :
    - Pas de BatchNorm sur la première couche (préserve les statistiques brutes)
    - Dropout(0.5) dans les 3 premiers blocs du décodeur
    - Activation Tanh pour sortie dans [-1, 1]

    Args:
        in_channels (int): Nombre de canaux d'entrée (défaut: 5)
        out_channels (int): Nombre de canaux de sortie (défaut: 3)
    """

    def __init__(self, in_channels=5, out_channels=3):
        super(UNetGenerator, self).__init__()

        # Encodeur : downsampling progressif
        self.down1 = self.conv_block(in_channels, 64, normalize=False)
        self.down2 = self.conv_block(64, 128)
        self.down3 = self.conv_block(128, 256)
        self.down4 = self.conv_block(256, 512)
        self.down5 = self.conv_block(512, 512)

        # Décodeur : upsampling avec skip connections
        self.up1 = self.up_block(512, 512, dropout=True)
        self.up2 = self.up_block(1024, 256, dropout=True)
        self.up3 = self.up_block(512, 128, dropout=True)
        self.up4 = self.up_block(256, 64)

        # Couche finale
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def conv_block(self, in_f, out_f, normalize=True):
        """
        Bloc de convolution pour l'encodeur.

        Args:
            in_f (int): Nombre de features en entrée
            out_f (int): Nombre de features en sortie
            normalize (bool): Appliquer BatchNorm (défaut: True)

        Returns:
            nn.Sequential: Séquence de couches
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
        Bloc de déconvolution pour le décodeur.

        Args:
            in_f (int): Nombre de features en entrée
            out_f (int): Nombre de features en sortie
            dropout (bool): Appliquer Dropout(0.5) (défaut: False)

        Returns:
            nn.Sequential: Séquence de couches
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
        Forward pass avec skip connections U-Net.

        Args:
            x (torch.Tensor): Tenseur d'entrée de forme (B, in_channels, H, W)

        Returns:
            torch.Tensor: Tenseur de sortie de forme (B, out_channels, H, W)
        """
        # Encodeur
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # Décodeur avec skip connections
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
    Discriminateur PatchGAN avec champ réceptif de 70×70 pixels.

    Architecture :
    - Classifie des patchs locaux plutôt que l'image entière
    - Encourage la cohérence des textures locales
    - Sortie : grille NxN de classifications

    Args:
        in_channels (int): Nombre de canaux d'entrée (défaut: 8)
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
        Bloc de convolution standard.

        Args:
            in_f (int): Nombre de features en entrée
            out_f (int): Nombre de features en sortie
            stride (int): Stride de la convolution (défaut: 2)

        Returns:
            nn.Sequential: Séquence de couches
        """
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_f),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input_img, output_img):
        """
        Forward pass du discriminateur.

        Args:
            input_img (torch.Tensor): Image d'entrée (B, 5, H, W)
            output_img (torch.Tensor): Image de sortie (B, 3, H, W)

        Returns:
            torch.Tensor: Grille de prédictions (B, 1, H', W')
        """
        x = torch.cat([input_img, output_img], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


def test_generator():
    """Teste le générateur U-Net."""
    print("="*60)
    print("TEST : Générateur U-Net")
    print("="*60)

    model = UNetGenerator(in_channels=5, out_channels=3)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Paramètres : {total_params:,}")

    batch_size = 4
    x = torch.randn(batch_size, 5, 256, 256)
    
    with torch.no_grad():
        output = model(x)

    print(f"Input      : {x.shape}")
    print(f"Output     : {output.shape}")
    print(f"Range      : [{output.min():.3f}, {output.max():.3f}]")

    assert output.shape == (batch_size, 3, 256, 256), "Shape incorrecte!"
    print("\n✓ Générateur OK\n")


def test_discriminator():
    """Teste le discriminateur PatchGAN."""
    print("="*60)
    print("TEST : Discriminateur PatchGAN")
    print("="*60)

    model = PatchGANDiscriminator(in_channels=8)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Paramètres : {total_params:,}")

    batch_size = 4
    input_img = torch.randn(batch_size, 5, 256, 256)
    output_img = torch.randn(batch_size, 3, 256, 256)

    with torch.no_grad():
        prediction = model(input_img, output_img)

    print(f"Input      : {input_img.shape}")
    print(f"Output     : {output_img.shape}")
    print(f"Prediction : {prediction.shape}")
    print(f"  → Grille de {prediction.shape[2]}×{prediction.shape[3]} patchs")
    print(f"  → Chaque patch = 70×70 pixels")
    print(f"Range      : [{prediction.min():.3f}, {prediction.max():.3f}]")

    assert prediction.shape == (batch_size, 1, 30, 30), "Shape incorrecte!"
    print("\n✓ Discriminateur OK\n")


def test_pix2pix():
    """Test complet de l'architecture Pix2Pix."""
    print("\n" + "="*60)
    print("TEST COMPLET : Architecture Pix2Pix")
    print("="*60 + "\n")

    test_generator()
    test_discriminator()

    print("="*60)
    print("✓ Architecture Pix2Pix validée")
    print("="*60)


if __name__ == "__main__":
    test_pix2pix()
