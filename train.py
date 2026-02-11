"""
Pix2Pix Training Script for SAR to Optical Translation

Complete implementation featuring:
- GAN Loss + L1 Loss (lambda=100)
- Adam optimizers (lr=0.0002, beta1=0.5, beta2=0.999)
- Mixed Precision Training (AMP)
- Validation metrics (PSNR, SSIM)
- Train/validation split
- Checkpoint saving

Based on: Isola et al. (2017)
"Image-to-Image Translation with Conditional Adversarial Networks"
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import rasterio
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import math
from PIL import Image

from model import UNetGenerator, PatchGANDiscriminator


def calculate_psnr(img1, img2, max_value=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    PSNR measures the quality of reconstructed images compared to originals.
    Higher values indicate better reconstruction quality.
    Typical range for good quality: 20-40 dB

    Args:
        img1 (torch.Tensor): Predicted image (B, C, H, W)
        img2 (torch.Tensor): Target image (B, C, H, W)
        max_value (float): Maximum pixel value (1.0 for normalized images)

    Returns:
        float: Average PSNR across the batch in dB
    """
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    return torch.mean(psnr).item()


def calculate_ssim(img1, img2, window_size=11, channel=3):
    """
    Calculate Structural Similarity Index (SSIM).
    
    SSIM measures perceptual similarity between images, considering
    luminance, contrast, and structure. More closely aligned with
    human visual perception than MSE-based metrics.
    Range: [-1, 1], where 1 indicates perfect similarity.

    Args:
        img1 (torch.Tensor): Predicted image (B, C, H, W)
        img2 (torch.Tensor): Target image (B, C, H, W)
        window_size (int): Gaussian window size
        channel (int): Number of channels

    Returns:
        float: Average SSIM across the batch
    """
    # Convert from [-1, 1] to [0, 1] range
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2

    # Constants for numerical stability (from original SSIM paper)
    C1 = (0.01) ** 2  # Stabilize luminance comparison
    C2 = (0.03) ** 2  # Stabilize contrast comparison

    # Calculate means
    mu1 = torch.mean(img1, dim=[2, 3], keepdim=True)
    mu2 = torch.mean(img2, dim=[2, 3], keepdim=True)

    # Calculate variances and covariance
    sigma1_sq = torch.mean((img1 - mu1) ** 2, dim=[2, 3], keepdim=True)
    sigma2_sq = torch.mean((img2 - mu2) ** 2, dim=[2, 3], keepdim=True)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2), dim=[2, 3], keepdim=True)

    # SSIM formula
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return torch.mean(ssim_map).item()


class SEN12Dataset(Dataset):
    """
    PyTorch Dataset for SAR + Cloudy Optical → Clear Optical triplets.
    
    Loading modes:
    - 'csv': Load triplets from a CSV file (recommended for cleaned datasets)
    - 'auto': Automatically discover all available triplets
    
    Args:
        data_folder (str): Root directory of the dataset
        csv_file (str, optional): Path to CSV file (required if mode='csv')
        mode (str): Discovery mode ('csv' or 'auto')
        transform (callable, optional): Transformations to apply
        augment (bool): Whether to apply data augmentation
    """

    def __init__(self, data_folder, csv_file=None, mode='auto', transform=None, augment=True):
        self.data_folder = Path(data_folder)
        self.transform = transform
        self.augment = augment
        self.triplets = []
        
        if mode == 'csv':
            if csv_file is None:
                raise ValueError("csv_file required in 'csv' mode")
            self._load_from_csv(csv_file)
        elif mode == 'auto':
            self._auto_discover_triplets()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        print(f"Dataset loaded: {len(self.triplets)} triplets (mode={mode})")
    
    def _load_from_csv(self, csv_file):
        """
        Load triplets from CSV file.
        
        Supports three CSV formats:
        - 10 columns: Full format with root folders (multi-season)
        - 7 columns: Legacy format (summer only)
        - 4 columns: Minimal format
        
        Args:
            csv_file (str): Path to CSV file
        """
        with open(csv_file, "r") as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if i == 0:  # Skip header
                continue
            
            parts = line.strip().split(",")
            if len(parts) == 10:
                # New format with root folders (multi-season support)
                patch_id, s1_root, s1_folder, s1_file, s2_root, s2_folder, s2_file, s2_cloudy_root, s2_cloudy_folder, s2_cloudy_file = parts
                self.triplets.append({
                    "id": patch_id,
                    "s1_root_folder": s1_root,
                    "s1_folder": s1_folder,
                    "s1_file": s1_file,
                    "s2_root_folder": s2_root,
                    "s2_folder": s2_folder,
                    "s2_file": s2_file,
                    "s2_cloudy_root_folder": s2_cloudy_root,
                    "s2_cloudy_folder": s2_cloudy_folder,
                    "s2_cloudy_file": s2_cloudy_file,
                })
            elif len(parts) == 7:
                # Legacy format (backward compatibility for summer only)
                patch_id, s1_folder, s1_file, s2_folder, s2_file, s2_cloudy_folder, s2_cloudy_file = parts
                self.triplets.append({
                    "id": patch_id,
                    "s1_root_folder": "ROIs1868_summer_s1",
                    "s1_folder": s1_folder,
                    "s1_file": s1_file,
                    "s2_root_folder": "ROIs1868_summer_s2",
                    "s2_folder": s2_folder,
                    "s2_file": s2_file,
                    "s2_cloudy_root_folder": "ROIs1868_summer_s2_cloudy",
                    "s2_cloudy_folder": s2_cloudy_folder,
                    "s2_cloudy_file": s2_cloudy_file,
                })
            elif len(parts) == 4:
                # Minimal format
                patch_id, s1_folder, s2_folder, s2_cloudy_folder = parts
                self.triplets.append({
                    "id": patch_id,
                    "s1_root_folder": "ROIs1868_summer_s1",
                    "s1_folder": s1_folder,
                    "s1_file": None,
                    "s2_root_folder": "ROIs1868_summer_s2",
                    "s2_folder": s2_folder,
                    "s2_file": None,
                    "s2_cloudy_root_folder": "ROIs1868_summer_s2_cloudy",
                    "s2_cloudy_folder": s2_cloudy_folder,
                    "s2_cloudy_file": None,
                })
    
    def _auto_discover_triplets(self):
        """
        Automatically discover all available triplets (all seasons).
        
        Searches for matching SAR, clear optical, and cloudy optical patches
        across all seasonal folders (summer, winter, etc.).
        """
        # Find ALL folders matching patterns (summer, winter, etc.)
        s1_folders = list(self.data_folder.glob("ROIs*_s1"))
        s2_folders = list(self.data_folder.glob("ROIs*_s2"))
        s2_cloudy_folders = list(self.data_folder.glob("ROIs*_s2_cloudy"))
        
        # Build patch dictionaries for each modality
        s1_patches = {}
        for s1_folder in s1_folders:
            if s1_folder.exists():
                for folder in s1_folder.iterdir():
                    if folder.is_dir():
                        season = s1_folder.name.split("_")[1]  # 'summer', 'winter', etc.
                        patch_name = folder.name.replace("s1_", "")
                        patch_id = f"{season}_{patch_name}"
                        s1_patches[patch_id] = (s1_folder.name, folder.name)
        
        s2_patches = {}
        for s2_folder in s2_folders:
            if s2_folder.exists():
                for folder in s2_folder.iterdir():
                    if folder.is_dir() and "cloudy" not in folder.name:
                        season = s2_folder.name.split("_")[1]
                        patch_name = folder.name.replace("s2_", "")
                        patch_id = f"{season}_{patch_name}"
                        s2_patches[patch_id] = (s2_folder.name, folder.name)
        
        s2_cloudy_patches = {}
        for s2_cloudy_folder in s2_cloudy_folders:
            if s2_cloudy_folder.exists():
                for folder in s2_cloudy_folder.iterdir():
                    if folder.is_dir():
                        season = s2_cloudy_folder.name.split("_")[1]
                        patch_name = folder.name.replace("s2_cloudy_", "")
                        patch_id = f"{season}_{patch_name}"
                        s2_cloudy_patches[patch_id] = (s2_cloudy_folder.name, folder.name)
        
        # Find complete triplets (intersection of all three modalities)
        complete_patches = set(s1_patches.keys()) & set(s2_patches.keys()) & set(s2_cloudy_patches.keys())
        
        for patch_id in sorted(complete_patches):
            s1_root, s1_folder = s1_patches[patch_id]
            s2_root, s2_folder = s2_patches[patch_id]
            s2_cloudy_root, s2_cloudy_folder = s2_cloudy_patches[patch_id]
            self.triplets.append({
                "id": patch_id,
                "s1_root_folder": s1_root,
                "s1_folder": s1_folder,
                "s1_file": None,
                "s2_root_folder": s2_root,
                "s2_folder": s2_folder,
                "s2_file": None,
                "s2_cloudy_root_folder": s2_cloudy_root,
                "s2_cloudy_folder": s2_cloudy_folder,
                "s2_cloudy_file": None,
            })

    @staticmethod
    def normalize_sar(sar_array):
        """
        Normalize SAR backscatter values to [-1, 1] range.
        
        SAR data typically ranges from -30 to 0 dB. This normalization
        preserves the physical relationship while scaling to the GAN's
        expected input range.
        
        Args:
            sar_array (np.ndarray): SAR values in dB scale
        
        Returns:
            np.ndarray: Normalized array in [-1, 1]
        """
        sar_clipped = np.clip(sar_array, -30, 0)  # Clip extreme outliers
        return (sar_clipped + 30) / 30 * 2 - 1  # Map: -30dB → -1, 0dB → 1

    @staticmethod
    def normalize_optical(optical_array):
        """
        Normalize optical reflectance values to [-1, 1] range.
        
        Sentinel-2 data ranges from 0-10000. Division by 5000 centers
        typical vegetation reflectance (3000-4000) around 0.
        
        Args:
            optical_array (np.ndarray): Reflectance values (0-10000)
        
        Returns:
            np.ndarray: Normalized array in [-1, 1]
        """
        return (optical_array / 5000.0) - 1.0

    @staticmethod
    def denormalize_optical(normalized_array):
        """
        Reverse optical normalization for visualization.
        
        Args:
            normalized_array (np.ndarray): Normalized values in [-1, 1]
        
        Returns:
            np.ndarray: Reflectance values in 0-10000 range
        """
        return (normalized_array + 1.0) * 5000.0

    def __len__(self):
        """Return the total number of triplets."""
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        Get a single triplet sample.
        
        Workflow:
        1. Load SAR, clear optical, and cloudy optical images
        2. Apply normalization to each modality
        3. Apply data augmentation if enabled
        4. Concatenate SAR + cloudy as input condition
        
        Args:
            idx (int): Sample index
        
        Returns:
            dict: {
                'condition': Input tensor (5, 256, 256) - SAR VV/VH + cloudy RGB
                'target': Target tensor (3, 256, 256) - clear RGB
            }
        """
        triplet = self.triplets[idx]
        
        # Build file paths with multi-season support
        s1_root = triplet.get("s1_root_folder", "ROIs1868_summer_s1")
        s2_root = triplet.get("s2_root_folder", "ROIs1868_summer_s2")
        s2_cloudy_root = triplet.get("s2_cloudy_root_folder", "ROIs1868_summer_s2_cloudy")
        
        # Construct full paths
        s1_folder_path = self.data_folder / s1_root / triplet["s1_folder"]
        s2_folder_path = self.data_folder / s2_root / triplet["s2_folder"]
        s2_cloudy_folder_path = self.data_folder / s2_cloudy_root / triplet["s2_cloudy_folder"]
        
        # Find .tif files (flexible naming)
        if triplet["s1_file"]:
            s1_path = s1_folder_path / triplet["s1_file"]
        else:
            s1_files = list(s1_folder_path.glob("*.tif"))
            s1_path = s1_files[0] if s1_files else None
        
        if triplet["s2_file"]:
            s2_path = s2_folder_path / triplet["s2_file"]
        else:
            s2_files = list(s2_folder_path.glob("*.tif"))
            s2_path = s2_files[0] if s2_files else None
        
        if triplet["s2_cloudy_file"]:
            s2_cloudy_path = s2_cloudy_folder_path / triplet["s2_cloudy_file"]
        else:
            s2_cloudy_files = list(s2_cloudy_folder_path.glob("*.tif"))
            s2_cloudy_path = s2_cloudy_files[0] if s2_cloudy_files else None

        # Load SAR (VV and VH channels)
        with rasterio.open(s1_path) as src:
            sar = src.read([1, 2]).astype(np.float32)  # Shape: (2, 256, 256)
        
        # Load optical RGB (bands 4, 3, 2 = Red, Green, Blue)
        with rasterio.open(s2_path) as src:
            s2_clear = src.read([4, 3, 2]).astype(np.float32)  # Shape: (3, 256, 256)
        
        with rasterio.open(s2_cloudy_path) as src:
            s2_cloudy = src.read([4, 3, 2]).astype(np.float32)  # Shape: (3, 256, 256)

        # Apply normalization
        sar = self.normalize_sar(sar)
        s2_clear = self.normalize_optical(s2_clear)
        s2_cloudy = self.normalize_optical(s2_cloudy)

        # Convert to PyTorch tensors
        sar = torch.from_numpy(sar)
        s2_clear = torch.from_numpy(s2_clear)
        s2_cloudy = torch.from_numpy(s2_cloudy)

        # Apply data augmentation if enabled
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                sar = torch.flip(sar, dims=[2])
                s2_clear = torch.flip(s2_clear, dims=[2])
                s2_cloudy = torch.flip(s2_cloudy, dims=[2])
            
            # Random vertical flip
            if torch.rand(1) > 0.5:
                sar = torch.flip(sar, dims=[1])
                s2_clear = torch.flip(s2_clear, dims=[1])
                s2_cloudy = torch.flip(s2_cloudy, dims=[1])
            
            # Random 90° rotation (0°, 90°, 180°, 270°)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                sar = torch.rot90(sar, k, dims=[1, 2])
                s2_clear = torch.rot90(s2_clear, k, dims=[1, 2])
                s2_cloudy = torch.rot90(s2_cloudy, k, dims=[1, 2])

        # Apply custom transform if provided
        if self.transform:
            sar = self.transform(sar)
            s2_clear = self.transform(s2_clear)
            s2_cloudy = self.transform(s2_cloudy)

        # Concatenate SAR and cloudy optical as input condition
        condition = torch.cat([sar, s2_cloudy], dim=0)  # (5, 256, 256)

        return {
            "condition": condition,  # Input: SAR + cloudy RGB
            "target": s2_clear,      # Target: clear RGB
        }

    def split_train_val(self, val_split=0.2):
        """
        Split dataset into training and validation sets.
        
        Args:
            val_split (float): Fraction of data to use for validation
        
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        val_size = int(len(self) * val_split)
        train_size = len(self) - val_size
        
        return random_split(
            self,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible split
        )


class Pix2PixTrainer:
    """
    Trainer class for Pix2Pix GAN.
    
    Implements the complete training loop with:
    - Adversarial loss (BCEWithLogitsLoss)
    - L1 reconstruction loss (MAE)
    - Mixed precision training (AMP)
    - Validation metrics (PSNR, SSIM)
    - Checkpoint saving
    - Visualization of results
    
    Args:
        generator (nn.Module): Generator network
        discriminator (nn.Module): Discriminator network
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to use (cuda/cpu)
        config (dict): Training configuration
    """

    def __init__(
        self, generator, discriminator, train_loader, val_loader, device, config
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # Loss functions
        # BCEWithLogitsLoss includes sigmoid activation for numerical stability
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()  # Mean Absolute Error
        self.lambda_l1 = config["lambda_l1"]  # Weight for L1 loss (default: 100)

        # Optimizers (as per Pix2Pix paper)
        # beta1=0.5 instead of 0.9 for better GAN stability
        self.optimizer_G = torch.optim.Adam(
            generator.parameters(), lr=config["lr"], betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            discriminator.parameters(), lr=config["lr"], betas=(0.5, 0.999)
        )

        # Mixed precision training
        self.use_amp = config.get("use_amp", True)
        self.scaler = GradScaler() if self.use_amp else None

        # Tracking
        self.current_epoch = 0
        self.history = {"loss_G": [], "loss_D": []}

        # Directories
        self.save_dir = Path(config["save_dir"])
        self.results_dir = Path(config["results_dir"])
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)

    def train_epoch(self):
        """
        Train for one epoch.
        
        Training procedure (per batch):
        1. Update Discriminator:
           - Classify real (condition + target) pairs
           - Classify fake (condition + generated) pairs
           - Backpropagate discriminator loss
        
        2. Update Generator:
           - Generate fake images
           - Fool discriminator (adversarial loss)
           - Minimize L1 distance to target (reconstruction loss)
           - Backpropagate combined loss
        
        Returns:
            dict: Average losses for the epoch
        """
        self.generator.train()
        self.discriminator.train()

        # Accumulators for epoch statistics
        epoch_loss_G = 0.0
        epoch_loss_G_gan = 0.0
        epoch_loss_G_l1 = 0.0
        epoch_loss_D = 0.0
        epoch_loss_D_real = 0.0
        epoch_loss_D_fake = 0.0

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            condition = batch["condition"].to(self.device)  # SAR + cloudy (5, 256, 256)
            target = batch["target"].to(self.device)        # Clear RGB (3, 256, 256)
            batch_size = condition.size(0)

            # Real and fake labels
            real_labels = torch.ones(batch_size, 1, 30, 30).to(self.device)
            fake_labels = torch.zeros(batch_size, 1, 30, 30).to(self.device)

            # ==================== Train Discriminator ====================
            self.optimizer_D.zero_grad()

            with autocast(enabled=self.use_amp):
                # Generate fake images
                fake_target = self.generator(condition)

                # Real loss: discriminator should classify real pairs as real
                pred_real = self.discriminator(condition, target)
                loss_D_real = self.criterion_gan(pred_real, real_labels)

                # Fake loss: discriminator should classify fake pairs as fake
                pred_fake = self.discriminator(condition, fake_target.detach())
                loss_D_fake = self.criterion_gan(pred_fake, fake_labels)

                # Total discriminator loss (average of real and fake)
                loss_D = (loss_D_real + loss_D_fake) * 0.5

            # Backpropagation
            if self.use_amp:
                self.scaler.scale(loss_D).backward()
                self.scaler.step(self.optimizer_D)
            else:
                loss_D.backward()
                self.optimizer_D.step()

            # ==================== Train Generator ====================
            self.optimizer_G.zero_grad()

            with autocast(enabled=self.use_amp):
                # Generate fake images
                fake_target = self.generator(condition)

                # GAN loss: generator should fool discriminator
                pred_fake = self.discriminator(condition, fake_target)
                loss_G_gan = self.criterion_gan(pred_fake, real_labels)

                # L1 loss: generator should reconstruct target
                loss_G_l1 = self.criterion_l1(fake_target, target)

                # Combined generator loss
                # lambda_l1 weights reconstruction vs adversarial loss
                loss_G = loss_G_gan + self.lambda_l1 * loss_G_l1

            # Backpropagation
            if self.use_amp:
                self.scaler.scale(loss_G).backward()
                self.scaler.step(self.optimizer_G)
                self.scaler.update()
            else:
                loss_G.backward()
                self.optimizer_G.step()

            # Accumulate losses
            epoch_loss_G += loss_G.item()
            epoch_loss_G_gan += loss_G_gan.item()
            epoch_loss_G_l1 += loss_G_l1.item()
            epoch_loss_D += loss_D.item()
            epoch_loss_D_real += loss_D_real.item()
            epoch_loss_D_fake += loss_D_fake.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "G": f"{loss_G.item():.4f}",
                    "D": f"{loss_D.item():.4f}",
                }
            )

        # Calculate epoch averages
        n_batches = len(self.train_loader)
        losses = {
            "loss_G": epoch_loss_G / n_batches,
            "loss_G_gan": epoch_loss_G_gan / n_batches,
            "loss_G_l1": epoch_loss_G_l1 / n_batches,
            "loss_D": epoch_loss_D / n_batches,
            "loss_D_real": epoch_loss_D_real / n_batches,
            "loss_D_fake": epoch_loss_D_fake / n_batches,
        }

        self.current_epoch += 1
        return losses

    @torch.no_grad()
    def validate(self):
        """
        Run validation and calculate metrics.
        
        Evaluates the generator on the validation set using:
        - PSNR: Peak Signal-to-Noise Ratio (measures pixel-level accuracy)
        - SSIM: Structural Similarity Index (measures perceptual similarity)
        
        Returns:
            dict: Validation metrics
        """
        self.generator.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        n_batches = 0

        for batch in self.val_loader:
            condition = batch["condition"].to(self.device)
            target = batch["target"].to(self.device)

            # Generate predictions
            fake_target = self.generator(condition)

            # Calculate metrics
            total_psnr += calculate_psnr(fake_target, target)
            total_ssim += calculate_ssim(fake_target, target)
            n_batches += 1

        # Average metrics
        metrics = {
            "psnr": total_psnr / n_batches,
            "ssim": total_ssim / n_batches,
        }

        return metrics

    @torch.no_grad()
    def save_validation_results(self, n_samples=4):
        """
        Save a grid of validation results for visual inspection.
        
        Creates a comparison grid showing:
        - Column 1: SAR VV (grayscale, enhanced contrast)
        - Column 2: Cloudy optical RGB
        - Column 3: Generated clear RGB
        - Column 4: Ground truth clear RGB
        
        Args:
            n_samples (int): Number of samples to visualize
        """
        self.generator.eval()

        # Get a batch from validation set
        batch = next(iter(self.val_loader))
        condition = batch["condition"].to(self.device)
        target = batch["target"].to(self.device)
        n_samples = min(n_samples, condition.size(0))

        # Generate predictions
        fake_target = self.generator(condition[:n_samples])

        # Prepare visualization (denormalize to [0, 1] range)
        def to_display(tensor):
            """Convert from [-1, 1] to [0, 1] range and clip."""
            return torch.clamp((tensor + 1) / 2, 0, 1)

        # Create figure
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
        if n_samples == 1:
            axes = axes[np.newaxis, :]

        for i in range(n_samples):
            # SAR VV channel (enhanced contrast for visibility)
            sar_vv = condition[i, 0].cpu().numpy()
            sar_display = np.clip((sar_vv - sar_vv.min()) / (sar_vv.max() - sar_vv.min() + 1e-6), 0, 1)
            axes[i, 0].imshow(sar_display, cmap='gray')
            axes[i, 0].set_title("SAR VV (Enhanced)")
            axes[i, 0].axis("off")

            # Cloudy optical RGB
            cloudy_rgb = to_display(condition[i, 2:5]).permute(1, 2, 0).cpu().numpy()
            axes[i, 1].imshow(cloudy_rgb)
            axes[i, 1].set_title("Cloudy Optical")
            axes[i, 1].axis("off")

            # Generated clear RGB
            generated = to_display(fake_target[i]).permute(1, 2, 0).cpu().numpy()
            axes[i, 2].imshow(generated)
            axes[i, 2].set_title("Generated Clear")
            axes[i, 2].axis("off")

            # Ground truth clear RGB
            ground_truth = to_display(target[i]).permute(1, 2, 0).cpu().numpy()
            axes[i, 3].imshow(ground_truth)
            axes[i, 3].set_title("Ground Truth")
            axes[i, 3].axis("off")

        plt.suptitle(
            f"Validation Results - Epoch {self.current_epoch}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save figure
        save_path = self.results_dir / f"epoch_{self.current_epoch:03d}_validation.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"   Validation results saved: {save_path}")

    @torch.no_grad()
    def save_individual_images(self, n_samples=4):
        """
        Save individual validation images for detailed inspection.
        
        Creates separate PNG files for each sample containing:
        - SAR channels (VV, VH)
        - Cloudy optical RGB
        - Generated clear RGB
        - Ground truth clear RGB
        
        Args:
            n_samples (int): Number of samples to save
        """
        self.generator.eval()

        # Get a batch from validation set
        batch = next(iter(self.val_loader))
        condition = batch["condition"].to(self.device)
        target = batch["target"].to(self.device)
        n_samples = min(n_samples, condition.size(0))

        # Generate predictions
        fake_target = self.generator(condition[:n_samples])

        # Denormalization function
        def to_display(tensor):
            """Convert from [-1, 1] to [0, 255] range for saving."""
            return torch.clamp((tensor + 1) / 2 * 255, 0, 255).byte()

        # Create sample directory
        sample_dir = self.results_dir / f"epoch_{self.current_epoch:03d}_samples"
        sample_dir.mkdir(exist_ok=True, parents=True)

        for i in range(n_samples):
            sample_subdir = sample_dir / f"sample_{i:02d}"
            sample_subdir.mkdir(exist_ok=True, parents=True)

            # SAR VV channel
            sar_vv = condition[i, 0].cpu().numpy()
            sar_vv_display = np.clip((sar_vv - sar_vv.min()) / (sar_vv.max() - sar_vv.min() + 1e-6) * 255, 0, 255).astype(np.uint8)
            Image.fromarray(sar_vv_display).save(sample_subdir / "sar_vv.png")

            # SAR VH channel
            sar_vh = condition[i, 1].cpu().numpy()
            sar_vh_display = np.clip((sar_vh - sar_vh.min()) / (sar_vh.max() - sar_vh.min() + 1e-6) * 255, 0, 255).astype(np.uint8)
            Image.fromarray(sar_vh_display).save(sample_subdir / "sar_vh.png")

            # Cloudy optical RGB
            cloudy_rgb = to_display(condition[i, 2:5]).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(cloudy_rgb).save(sample_subdir / "cloudy_optical.png")

            # Generated clear RGB
            generated = to_display(fake_target[i]).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(generated).save(sample_subdir / "generated_clear.png")

            # Ground truth clear RGB
            ground_truth = to_display(target[i]).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(ground_truth).save(sample_subdir / "ground_truth.png")

        print(f"   Individual images saved: {sample_dir}")

    def save_checkpoint(self, filename=None):
        """
        Save model checkpoint.
        
        Saves:
        - Generator and discriminator state dicts
        - Optimizer state dicts
        - Current epoch
        - Training history
        - Configuration
        
        Args:
            filename (str, optional): Checkpoint filename (auto-generated if None)
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch:03d}.pth"

        checkpoint = {
            "epoch": self.current_epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
            "history": self.history,
            "config": self.config,
        }

        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"   Checkpoint saved: {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        
        Restores:
        - Generator and discriminator weights
        - Optimizer states
        - Training progress
        - History
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.history = checkpoint["history"]

        print(f"   Checkpoint loaded: epoch {self.current_epoch}")

    @staticmethod
    def init_training_log(log_path):
        """
        Initialize CSV file for logging training metrics.
        
        Creates a CSV file with headers for tracking:
        - Epoch number
        - Generator losses (total, GAN, L1)
        - Discriminator losses (total, real, fake)
        - Validation metrics (PSNR, SSIM)
        
        Args:
            log_path (Path): Path to CSV file
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            f.write("Epoch,Loss_G,Loss_G_GAN,Loss_G_L1,Loss_D,Loss_D_Real,Loss_D_Fake,PSNR_Val,SSIM_Val\n")
        print(f"   Training log initialized: {log_path}")

    def log_metrics(self, log_path, losses, metrics):
        """
        Log training metrics to CSV file.
        
        Appends a row to the training log with current epoch's metrics.
        Useful for tracking convergence and creating training curves.
        
        Args:
            log_path (Path): Path to CSV file
            losses (dict): Training losses for the epoch
            metrics (dict): Validation metrics (PSNR, SSIM)
        """
        with open(log_path, 'a') as f:
            f.write(f"{self.current_epoch},"
                   f"{losses['loss_G']:.6f},"
                   f"{losses['loss_G_gan']:.6f},"
                   f"{losses['loss_G_l1']:.6f},"
                   f"{losses['loss_D']:.6f},"
                   f"{losses['loss_D_real']:.6f},"
                   f"{losses['loss_D_fake']:.6f},"
                   f"{metrics.get('psnr', 0.0):.4f},"
                   f"{metrics.get('ssim', 0.0):.6f}\n")


def main():
    """
    Main training script with validation and metrics.
    
    Workflow:
    1. Load dataset (from CSV or auto-discover)
    2. Split into train/validation sets
    3. Initialize generator and discriminator
    4. Train for specified number of epochs
    5. Save checkpoints and validation results
    6. Log metrics to CSV for analysis
    
    Configuration can be modified in the config dictionary below.
    """

    # ==================== Configuration ====================
    config = {
        # Dataset settings
        "data_folder": "data/sen_1_2",
        "dataset_mode": "csv",  # 'csv' or 'auto'
        "csv_file": "data/sen_1_2/cleaned_triplets.csv",
        
        # Training hyperparameters
        "batch_size": 48,          # Optimized for RTX 3080 (10GB VRAM)
        "num_epochs": 200,         # Typical: 100-200 epochs for convergence
        "lr": 0.0002,              # Learning rate (as per Pix2Pix paper)
        "lambda_l1": 100,          # L1 loss weight (higher = more reconstruction focus)
        
        # Optimization
        "use_amp": True,           # Mixed precision training (faster, less memory)
        
        # Directories
        "save_dir": "checkpoints",
        "results_dir": "results",
        
        # Validation and saving
        "val_split": 0.15,         # 15% of data for validation
        "val_freq": 2,             # Validate every N epochs
        "save_freq": 10,           # Save checkpoint every N epochs
        "save_individual": True,   # Save individual validation images
    }

    # ==================== Device Setup ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "=" * 60)
    print(f"PIX2PIX TRAINING: SAR to Optical (Cloud Removal)")
    print("=" * 60)
    print(f"  Device          : {device}")
    if torch.cuda.is_available():
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM            : {vram_gb:.2f} GB")
    print("=" * 60 + "\n")

    # ==================== Dataset Loading ====================
    if config["dataset_mode"] == "auto":
        # Auto-discover all available triplets
        full_dataset = SEN12Dataset(
            data_folder=config["data_folder"],
            mode="auto"
        )
    else:
        # Load from CSV (recommended for cleaned datasets)
        full_dataset = SEN12Dataset(
            data_folder=config["data_folder"],
            csv_file=config["csv_file"],
            mode="csv"
        )

    # Split into train/validation sets
    train_dataset, val_dataset = full_dataset.split_train_val(
        val_split=config["val_split"]
    )

    print(f"\n{'='*60}")
    print(f"DATASET VERIFICATION")
    print(f"{'='*60}")
    print(f"  Total triplets  : {len(full_dataset)}")
    print(f"  Train samples   : {len(train_dataset)}")
    print(f"  Val samples     : {len(val_dataset)}")
    print(f"  CSV file used   : {config.get('csv_file', 'N/A')}")
    print(f"  Mode            : {config['dataset_mode']}")
    print(f"{'='*60}\n")

    # ==================== Data Loaders ====================
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=8,           # Parallel data loading
        pin_memory=True,         # Faster GPU transfer
        persistent_workers=True, # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"  Batch size      : {config['batch_size']}")
    print(f"  Train batches   : {len(train_loader)}")
    print(f"  Val batches     : {len(val_loader)}\n")

    # ==================== Model Initialization ====================
    generator = UNetGenerator(in_channels=5, out_channels=3)
    discriminator = PatchGANDiscriminator(in_channels=8)

    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"  Generator       : {gen_params:,} parameters")
    print(f"  Discriminator   : {disc_params:,} parameters\n")

    # ==================== Trainer Initialization ====================
    trainer = Pix2PixTrainer(
        generator, discriminator, train_loader, val_loader, device, config
    )

    # Initialize training log CSV
    log_path = Path(config["results_dir"]) / "training_log.csv"
    Pix2PixTrainer.init_training_log(log_path)

    # ==================== Training Loop ====================
    print("=" * 60)
    print("TRAINING START")
    print("=" * 60 + "\n")

    start_time = time.time()

    try:
        for epoch in range(config["num_epochs"]):
            # Train for one epoch
            losses = trainer.train_epoch()

            # Display training losses
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}:")
            print(
                f"   Loss G: {losses['loss_G']:.4f} "
                + f"(GAN: {losses['loss_G_gan']:.4f}, L1: {losses['loss_G_l1']:.4f})"
            )
            print(
                f"   Loss D: {losses['loss_D']:.4f} "
                + f"(Real: {losses['loss_D_real']:.4f}, Fake: {losses['loss_D_fake']:.4f})"
            )

            # Validation and visualization
            if (epoch + 1) % config["val_freq"] == 0:
                metrics = trainer.validate()
                print(
                    f"   Validation - PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}"
                )
                trainer.save_validation_results(n_samples=4)
                
                # Log metrics to CSV
                trainer.log_metrics(log_path, losses, metrics)
                
                # Save individual images for detailed inspection
                if config.get("save_individual", True):
                    trainer.save_individual_images(n_samples=4)
            else:
                # Log only losses (without validation)
                trainer.log_metrics(log_path, losses, {})

            # Save checkpoint
            if (epoch + 1) % config["save_freq"] == 0:
                trainer.save_checkpoint()

        # Save final model
        trainer.save_checkpoint("final_model.pth")

    except KeyboardInterrupt:
        # Handle manual interruption (Ctrl+C)
        print("\n\nTraining interrupted by user")
        trainer.save_checkpoint("interrupted_model.pth")

    # ==================== Training Complete ====================
    elapsed_time = time.time() - start_time
    print(f"\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"  Total time      : {elapsed_time / 3600:.2f} hours")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
