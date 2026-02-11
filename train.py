"""
Script d'entraînement Pix2Pix pour traduction SAR vers Optique

Implémentation complète avec :
- GAN Loss + L1 Loss (lambda=100)
- Optimiseurs Adam (lr=0.0002, beta1=0.5, beta2=0.999)
- Mixed Precision Training (AMP)
- Métriques de validation (PSNR, SSIM)
- Split train/validation
- Sauvegarde des checkpoints

Basé sur : Isola et al. (2017)
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
    Calcule le Peak Signal-to-Noise Ratio (PSNR).

    Args:
        img1 (torch.Tensor): Image prédite (B, C, H, W)
        img2 (torch.Tensor): Image cible (B, C, H, W)
        max_value (float): Valeur maximale des pixels

    Returns:
        float: PSNR moyen sur le batch
    """
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    return torch.mean(psnr).item()


def calculate_ssim(img1, img2, window_size=11, channel=3):
    """
    Calcule le Structural Similarity Index (SSIM).

    Args:
        img1 (torch.Tensor): Image prédite (B, C, H, W)
        img2 (torch.Tensor): Image cible (B, C, H, W)
        window_size (int): Taille de la fenêtre gaussienne
        channel (int): Nombre de canaux

    Returns:
        float: SSIM moyen sur le batch
    """
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mu1 = torch.mean(img1, dim=[2, 3], keepdim=True)
    mu2 = torch.mean(img2, dim=[2, 3], keepdim=True)

    sigma1_sq = torch.mean((img1 - mu1) ** 2, dim=[2, 3], keepdim=True)
    sigma2_sq = torch.mean((img2 - mu2) ** 2, dim=[2, 3], keepdim=True)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2), dim=[2, 3], keepdim=True)

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return torch.mean(ssim_map).item()


class SEN12Dataset(Dataset):
    """
    Dataset pour triplets SAR + S2 Cloudy → S2 Clear.
    
    Modes de chargement :
    - 'csv' : Charge les triplets depuis un fichier CSV
    - 'auto' : Découvre automatiquement tous les triplets disponibles
    
    Args:
        data_folder (str): Dossier racine du dataset
        csv_file (str, optional): Chemin vers le CSV (requis si mode='csv')
        mode (str): Mode de découverte ('csv' ou 'auto')
        transform (callable, optional): Transformations à appliquer
    """

    def __init__(self, data_folder, csv_file=None, mode='auto', transform=None, augment=True):
        self.data_folder = Path(data_folder)
        self.transform = transform
        self.augment = augment
        self.triplets = []
        
        if mode == 'csv':
            if csv_file is None:
                raise ValueError("csv_file requis en mode 'csv'")
            self._load_from_csv(csv_file)
        elif mode == 'auto':
            self._auto_discover_triplets()
        else:
            raise ValueError(f"Mode inconnu : {mode}")

        print(f"Dataset chargé : {len(self.triplets)} triplets (mode={mode})")
    
    def _load_from_csv(self, csv_file):
        """Load triplets from CSV file. Supports old (7 cols) and new (10 cols) formats."""
        with open(csv_file, "r") as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if i == 0:
                continue
            
            parts = line.strip().split(",")
            if len(parts) == 10:
                # Nouveau format avec root_folders (multisaison)
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
                # Ancien format (rétrocompatibilité summer only)
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
                # Format minimal
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
        """Découvre automatiquement tous les triplets disponibles (toutes saisons)."""
        # Chercher TOUS les dossiers qui matchent les patterns (summer, winter, etc.)
        s1_folders = list(self.data_folder.glob("ROIs*_s1"))
        s2_folders = list(self.data_folder.glob("ROIs*_s2"))
        s2_cloudy_folders = list(self.data_folder.glob("ROIs*_s2_cloudy"))
        
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
        
        complete_patches = set(s1_patches.keys()) & set(s2_patches.keys()) & set(s2_cloudy_patches.keys())
        
        for patch_id in sorted(complete_patches):
            s1_root, s1_folder = s1_patches[patch_id]
            s2_root, s2_folder = s2_patches[patch_id]
            s2_cloudy_root, s2_cloudy_folder = s2_cloudy_patches[patch_id]
            self.triplets.append({
                "id": patch_id,
                "s1_root_folder": s1_root,
                "s1_folder": s1_folder,
                "s2_root_folder": s2_root,
                "s2_folder": s2_folder,
                "s2_cloudy_root_folder": s2_cloudy_root,
                "s2_cloudy_folder": s2_cloudy_folder,
            })
        
        print(f"  S1: {len(s1_patches)} | S2: {len(s2_patches)} | S2_cloudy: {len(s2_cloudy_patches)}")
        print(f"  Triplets complets: {len(complete_patches)}")

    def __len__(self):
        """
        Retourne la taille effective du dataset.
        Avec augmentation : chaque triplet génère 4 variations (×4).
        """
        if self.augment:
            return len(self.triplets) * 4  # Original + flip_h + flip_v + flip_hv
        return len(self.triplets)

    def split_train_val(self, val_split=0.15, seed=42):
        """
        Sépare le dataset en ensembles train/validation.

        Args:
            val_split (float): Proportion de validation
            seed (int): Seed pour la reproductibilité

        Returns:
            tuple: (train_dataset, val_dataset)
        """
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        
        indices = list(range(len(self.triplets)))
        random.shuffle(indices)
        
        val_size = int(len(indices) * val_split)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        train_triplets = [self.triplets[i] for i in train_indices]
        val_triplets = [self.triplets[i] for i in val_indices]
        
        train_dataset = SEN12Dataset.__new__(SEN12Dataset)
        train_dataset.data_folder = self.data_folder
        train_dataset.transform = self.transform
        train_dataset.augment = True
        train_dataset.triplets = train_triplets
        
        val_dataset = SEN12Dataset.__new__(SEN12Dataset)
        val_dataset.data_folder = self.data_folder
        val_dataset.transform = self.transform
        val_dataset.augment = False
        val_dataset.triplets = val_triplets

        print(f"Split : {len(train_triplets)} triplets → {len(train_dataset)} train samples (with augmentation)")
        print(f"        {len(val_triplets)} triplets → {len(val_dataset)} val samples (no augmentation)")
        return train_dataset, val_dataset

    def __getitem__(self, idx):
        """
        Retourne un triplet (input, target, patch_id).
        
        Avec augmentation, chaque triplet génère 4 variations:
        - idx % 4 == 0 : Original
        - idx % 4 == 1 : Flip horizontal
        - idx % 4 == 2 : Flip vertical  
        - idx % 4 == 3 : Flip horizontal + vertical
        
        Returns:
            tuple: (input_tensor, target_tensor, patch_id)
                - input : (5, H, W) - VV, VH, R_cloudy, G_cloudy, B_cloudy
                - target : (3, H, W) - R_clear, G_clear, B_clear
                - patch_id : str - Identifiant du patch pour traçabilité
        """
        # Calculer l'indice réel du triplet et la variation d'augmentation
        if self.augment:
            triplet_idx = idx // 4
            aug_variant = idx % 4
        else:
            triplet_idx = idx
            aug_variant = 0
            
        triplet = self.triplets[triplet_idx]

        # Utiliser root_folders pour supporter multisaison
        s1_root = triplet.get("s1_root_folder", "ROIs1868_summer_s1")
        s2_root = triplet.get("s2_root_folder", "ROIs1868_summer_s2")
        s2_cloudy_root = triplet.get("s2_cloudy_root_folder", "ROIs1868_summer_s2_cloudy")
        
        s1_path = self.data_folder / s1_root / triplet["s1_folder"]
        s2_path = self.data_folder / s2_root / triplet["s2_folder"]
        s2_cloudy_path = self.data_folder / s2_cloudy_root / triplet["s2_cloudy_folder"]

        if triplet.get("s1_file"):
            s1_file = s1_path / triplet["s1_file"]
            s2_file = s2_path / triplet["s2_file"]
            s2_cloudy_file = s2_cloudy_path / triplet["s2_cloudy_file"]
        else:
            s1_file = list(s1_path.glob("*.tif"))[0]
            s2_file = list(s2_path.glob("*.tif"))[0]
            s2_cloudy_file = list(s2_cloudy_path.glob("*.tif"))[0]

        with rasterio.open(s1_file) as src:
            vv_raw = src.read(1)
            vh_raw = src.read(2)

        vv = self.normalize_sar_vv(vv_raw)
        vh = self.normalize_sar_vh(vh_raw)

        with rasterio.open(s2_cloudy_file) as src:
            bands_cloudy = src.read()

        r_cloudy = self.normalize_optical(bands_cloudy[3])
        g_cloudy = self.normalize_optical(bands_cloudy[2])
        b_cloudy = self.normalize_optical(bands_cloudy[1])

        with rasterio.open(s2_file) as src:
            bands_clear = src.read()

        r_clear = self.normalize_optical(bands_clear[3])
        g_clear = self.normalize_optical(bands_clear[2])
        b_clear = self.normalize_optical(bands_clear[1])

        # VÉRIFICATION ORDRE DES CANAUX (Correct ✓)
        # Canal 0 : S1 VV (SAR polarisation verticale-verticale)
        # Canal 1 : S1 VH (SAR polarisation verticale-horizontale)
        # Canaux 2-4 : S2 Cloudy RGB (Rouge, Vert, Bleu)
        input_tensor = torch.from_numpy(
            np.stack([vv, vh, r_cloudy, g_cloudy, b_cloudy], axis=0)
        )
        target_tensor = torch.from_numpy(np.stack([r_clear, g_clear, b_clear], axis=0))

        # Appliquer l'augmentation de manière déterministe selon la variante
        if self.augment and aug_variant > 0:
            # Variante 1 ou 3 : flip horizontal
            if aug_variant in [1, 3]:
                input_tensor = torch.flip(input_tensor, dims=[2])
                target_tensor = torch.flip(target_tensor, dims=[2])
            
            # Variante 2 ou 3 : flip vertical
            if aug_variant in [2, 3]:
                input_tensor = torch.flip(input_tensor, dims=[1])
                target_tensor = torch.flip(target_tensor, dims=[1])

        patch_id = triplet["id"]
        
        # Ajouter suffixe pour identifier la variante d'augmentation
        if self.augment and aug_variant > 0:
            aug_suffix = ["", "_fh", "_fv", "_fhv"][aug_variant]
            patch_id = f"{patch_id}{aug_suffix}"
            
        return input_tensor, target_tensor, patch_id

    @staticmethod
    def normalize_sar_vv(data):
        """
        Normalise les données SAR VV (co-pol) vers [-1, 1].
        
        Détecte automatiquement si les données sont en dB ou en intensité linéaire.
        Range typique VV: [-25, 0] dB
        Adapté pour été (végétation) et hiver (sols nus).
        """
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Détecter si déjà en dB : majorité des valeurs négatives ou dans range [-40, 5]
        if np.sum(data < 0) > 0.8 * data.size or (data.min() < 0 and data.max() < 10):
            # Déjà en dB
            db = data
        else:
            # En intensité linéaire, convertir en dB
            data = np.maximum(data, 1e-6)
            db = 10 * np.log10(data)
        
        db = np.clip(db, -25, 0)
        normalized = ((db + 25.0) / 25.0) * 2.0 - 1.0
        return normalized.astype(np.float32)
    
    @staticmethod
    def normalize_sar_vh(data):
        """
        Normalise les données SAR VH (cross-pol) vers [-1, 1].
        
        Détecte automatiquement si les données sont en dB ou en intensité linéaire.
        Range typique VH: [-35, -5] dB (plus faible que VV)
        Adapté pour été (dépolarisation végétation) et hiver (rugosité sols).
        """
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Détecter si déjà en dB : majorité des valeurs négatives ou dans range [-50, 5]
        if np.sum(data < 0) > 0.8 * data.size or (data.min() < 0 and data.max() < 10):
            # Déjà en dB
            db = data
        else:
            # En intensité linéaire, convertir en dB
            data = np.maximum(data, 1e-6)
            db = 10 * np.log10(data)
        
        db = np.clip(db, -35, -5)
        normalized = ((db + 35.0) / 30.0) * 2.0 - 1.0
        return normalized.astype(np.float32)

    @staticmethod
    def normalize_optical(data):
        """
        Normalise les données optiques vers [-1, 1].
        
        Normalisation fixe par 3000 (réflectance typique Sentinel-2).
        """
        data = np.nan_to_num(data.astype(np.float32), nan=0.0)
        data = np.clip(data, 0, 3000)
        normalized = (data / 3000.0) * 2.0 - 1.0
        return normalized.astype(np.float32)


class Pix2PixTrainer:
    """
    Trainer pour l'entraînement Pix2Pix.

    Fonctionnalités :
    - Mixed Precision Training (AMP)
    - GAN Loss + L1 Loss (lambda=100)
    - Métriques de validation (PSNR, SSIM)
    - Sauvegarde checkpoints et résultats
    
    Args:
        generator (nn.Module): Modèle générateur
        discriminator (nn.Module): Modèle discriminateur
        train_loader (DataLoader): Loader d'entraînement
        val_loader (DataLoader): Loader de validation
        device (torch.device): Device de calcul
        config (dict): Configuration d'entraînement
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

        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        self.lambda_l1 = config.get("lambda_l1", 100)

        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=config.get("lr", 0.0002), betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.get("lr", 0.0002),
            betas=(0.5, 0.999),
        )

        self.use_amp = config.get("use_amp", True)
        self.scaler_G = GradScaler(enabled=self.use_amp)
        self.scaler_D = GradScaler(enabled=self.use_amp)

        self.current_epoch = 0
        self.history = {
            "loss_G": [],
            "loss_D": [],
            "loss_G_gan": [],
            "loss_G_l1": [],
            "loss_D_real": [],
            "loss_D_fake": [],
            "val_psnr": [],
            "val_ssim": [],
        }

        self.save_dir = Path(config.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(exist_ok=True)
        self.results_dir = Path(config.get("results_dir", "results"))
        self.results_dir.mkdir(exist_ok=True)

        print(f"Trainer initialisé")
        print(f"  Device          : {device}")
        print(f"  Mixed Precision : {self.use_amp}")
        print(f"  Lambda L1       : {self.lambda_l1}")
        print(f"  Learning Rate   : {config.get('lr', 0.0002)}")

    def train_epoch(self):
        """
        Entraîne une époque complète.
        
        Returns:
            dict: Statistiques de l'époque (losses moyennes)
        """
        self.generator.train()
        self.discriminator.train()

        epoch_losses = {
            "loss_G": 0,
            "loss_D": 0,
            "loss_G_gan": 0,
            "loss_G_l1": 0,
            "loss_D_real": 0,
            "loss_D_fake": 0,
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for i, batch in enumerate(pbar):
            input_data, target_data, _ = batch
            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

            self.optimizer_D.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                fake_output = self.generator(input_data)

                pred_real = self.discriminator(input_data, target_data)
                loss_D_real = self.criterion_gan(pred_real, torch.ones_like(pred_real))

                pred_fake = self.discriminator(input_data, fake_output.detach())
                loss_D_fake = self.criterion_gan(pred_fake, torch.zeros_like(pred_fake))

                loss_D = (loss_D_real + loss_D_fake) * 0.5

            self.scaler_D.scale(loss_D).backward()
            self.scaler_D.step(self.optimizer_D)
            self.scaler_D.update()

            self.optimizer_G.zero_grad()

            with autocast(enabled=self.use_amp):
                fake_output = self.generator(input_data)

                pred_fake = self.discriminator(input_data, fake_output)
                loss_G_gan = self.criterion_gan(pred_fake, torch.ones_like(pred_fake))

                loss_G_l1 = self.criterion_l1(fake_output, target_data)

                loss_G = loss_G_gan + self.lambda_l1 * loss_G_l1

            self.scaler_G.scale(loss_G).backward()
            self.scaler_G.step(self.optimizer_G)
            self.scaler_G.update()

            epoch_losses["loss_G"] += loss_G.item()
            epoch_losses["loss_D"] += loss_D.item()
            epoch_losses["loss_G_gan"] += loss_G_gan.item()
            epoch_losses["loss_G_l1"] += loss_G_l1.item()
            epoch_losses["loss_D_real"] += loss_D_real.item()
            epoch_losses["loss_D_fake"] += loss_D_fake.item()

            pbar.set_postfix(
                {
                    "G": f"{loss_G.item():.4f}",
                    "D": f"{loss_D.item():.4f}",
                    "L1": f"{loss_G_l1.item():.4f}",
                }
            )

        n_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
            self.history[key].append(epoch_losses[key])

        self.current_epoch += 1
        return epoch_losses

    def validate(self):
        """
        Valide le modèle et calcule les métriques.
        
        Returns:
            dict: Métriques de validation (PSNR, SSIM)
        """
        self.generator.eval()

        total_psnr = 0
        total_ssim = 0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(
                self.val_loader, desc="Validation"
            ):
                input_data, target_data, _ = batch
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    predictions = self.generator(input_data)

                psnr = calculate_psnr(predictions, target_data, max_value=2.0)
                ssim = calculate_ssim(predictions, target_data)

                total_psnr += psnr
                total_ssim += ssim
                n_batches += 1

        avg_psnr = total_psnr / n_batches
        avg_ssim = total_ssim / n_batches

        self.history["val_psnr"].append(avg_psnr)
        self.history["val_ssim"].append(avg_ssim)

        return {"psnr": avg_psnr, "ssim": avg_ssim}
    
    def save_individual_images(self, n_samples=4):
        """
        Sauvegarde des images individuelles au format PNG.
        
        Conversion correcte : [-1, 1] → [0, 255] pour visualisation.
        
        Args:
            n_samples (int): Nombre d'échantillons à sauvegarder
        """
        self.generator.eval()
        
        batch = next(iter(self.val_loader))
        inputs, targets, patch_ids = batch
        inputs = inputs[:n_samples].to(self.device)
        targets = targets[:n_samples].to(self.device)
        patch_ids = patch_ids[:n_samples]
        
        with torch.no_grad():
            predictions = self.generator(inputs)
        
        epoch_dir = self.results_dir / f"epoch_{self.current_epoch:03d}_individual"
        epoch_dir.mkdir(exist_ok=True)
        
        for i in range(n_samples):
            patch_id = patch_ids[i]
            
            pred_np = predictions[i].cpu().numpy()
            pred_np = np.transpose(pred_np, (1, 2, 0))
            pred_np = ((pred_np + 1) / 2 * 255).astype(np.uint8)
            
            target_np = targets[i].cpu().numpy()
            target_np = np.transpose(target_np, (1, 2, 0))
            target_np = ((target_np + 1) / 2 * 255).astype(np.uint8)
            
            cloudy_np = inputs[i, 2:5].cpu().numpy()
            cloudy_np = np.transpose(cloudy_np, (1, 2, 0))
            cloudy_np = ((cloudy_np + 1) / 2 * 255).astype(np.uint8)
            
            Image.fromarray(pred_np).save(epoch_dir / f"{patch_id}_generated.png")
            Image.fromarray(target_np).save(epoch_dir / f"{patch_id}_target.png")
            Image.fromarray(cloudy_np).save(epoch_dir / f"{patch_id}_input.png")
        
        print(f"   Images individuelles sauvegardées : {epoch_dir}")
        self.generator.train()

    def save_validation_results(self, n_samples=4):
        """
        Sauvegarde planche comparative des résultats de validation.
        
        Génère une planche d'images montrant les résultats de validation.
        Colonnes : [SAR VV | S2 Cloudy | Prédiction | Ground Truth]
        """
        self.generator.eval()

        batch = next(iter(self.val_loader))
        inputs, targets, patch_ids = batch
        inputs = inputs[:n_samples].to(self.device)
        targets = targets[:n_samples].to(self.device)
        patch_ids = patch_ids[:n_samples]

        with torch.no_grad():
            predictions = self.generator(inputs)

        def to_numpy_img(tensor):
            img = tensor.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img + 1) / 2
            return np.clip(img, 0, 1)
        
        fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5 * n_samples))
        if n_samples == 1:
            axes = axes[np.newaxis, :]

        for i in range(n_samples):
            patch_id = patch_ids[i]
            
            # Colonne 1 : SAR VV normalisé ([-1,1] → [0,1])
            sar_vv = (inputs[i, 0].cpu().numpy() + 1) / 2
            axes[i, 0].imshow(sar_vv, cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_title(f"SAR VV - {patch_id}", fontsize=10)
            axes[i, 0].axis("off")

            # Colonne 2 : S2 Cloudy (Input RGB)
            cloudy = to_numpy_img(inputs[i, 2:5])
            axes[i, 1].imshow(cloudy)
            axes[i, 1].set_title("S2 Cloudy (Input)", fontsize=10)
            axes[i, 1].axis("off")

            # Colonne 3 : Prédiction du modèle
            pred = to_numpy_img(predictions[i])
            axes[i, 2].imshow(pred)
            axes[i, 2].set_title("Prédiction IA", fontsize=10)
            axes[i, 2].axis("off")

            # Colonne 4 : Ground Truth (S2 Clear)
            target = to_numpy_img(targets[i])
            axes[i, 3].imshow(target)
            axes[i, 3].set_title("Ground Truth (S2 Clear)", fontsize=10)
            axes[i, 3].axis("off")

        plt.tight_layout()

        save_path = self.results_dir / f"epoch_{self.current_epoch:03d}_validation.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"   Validation results saved: {save_path}")

        self.generator.train()

    def visualize_results(self, n_samples=3):
        """Visualize predictions from training set."""
        self.generator.eval()

        batch = next(iter(self.train_loader))
        inputs, targets, patch_ids = batch
        inputs = inputs[:n_samples].to(self.device)
        targets = targets[:n_samples].to(self.device)
        patch_ids = patch_ids[:n_samples]

        with torch.no_grad():
            predictions = self.generator(inputs)

        def to_numpy_img(tensor):
            img = tensor.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img + 1) / 2
            return np.clip(img, 0, 1)

        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
        if n_samples == 1:
            axes = axes[np.newaxis, :]

        for i in range(n_samples):
            patch_id = patch_ids[i]
            
            sar_vv = (inputs[i, 0].cpu().numpy() + 1) / 2
            axes[i, 0].imshow(sar_vv, cmap="gray")
            axes[i, 0].set_title(f"SAR VV - {patch_id}")
            axes[i, 0].axis("off")

            cloudy = to_numpy_img(inputs[i, 2:5])
            axes[i, 1].imshow(cloudy)
            axes[i, 1].set_title("S2 Cloudy (Input)")
            axes[i, 1].axis("off")

            pred = to_numpy_img(predictions[i])
            axes[i, 2].imshow(pred)
            axes[i, 2].set_title("Prediction (Generated)")
            axes[i, 2].axis("off")

            target = to_numpy_img(targets[i])
            axes[i, 3].imshow(target)
            axes[i, 3].set_title("Ground Truth (S2 Clear)")
            axes[i, 3].axis("off")

        plt.tight_layout()

        save_path = self.save_dir / f"epoch_{self.current_epoch:03d}_results.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"   Visualization saved: {save_path}")

        self.generator.train()

    def save_checkpoint(self, filename=None):
        """Save model checkpoint."""
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
        """Load model checkpoint."""
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
        Initialise le fichier CSV de logging des métriques.
        
        Args:
            log_path (Path): Chemin vers le fichier CSV
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            f.write("Epoch,Loss_G,Loss_G_GAN,Loss_G_L1,Loss_D,Loss_D_Real,Loss_D_Fake,PSNR_Val,SSIM_Val\n")
        print(f"   Training log initialized: {log_path}")

    def log_metrics(self, log_path, losses, metrics):
        """
        Enregistre les métriques dans le CSV de suivi.
        
        Args:
            log_path (Path): Chemin vers le fichier CSV
            losses (dict): Dictionnaire des losses d'entraînement
            metrics (dict): Dictionnaire des métriques de validation (PSNR, SSIM)
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
    """Main training script with validation and metrics."""

    config = {
        "data_folder": "data/sen_1_2",
        "dataset_mode": "csv",
        "csv_file": "data/sen_1_2/cleaned_triplets.csv",
        "batch_size": 48,
        "num_epochs": 200,
        "lr": 0.0002,
        "lambda_l1": 100,
        "use_amp": True,
        "save_dir": "checkpoints",
        "results_dir": "results",
        "val_split": 0.15,
        "val_freq": 2,
        "save_freq": 10,
        "save_individual": True,
    }

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

    if config["dataset_mode"] == "auto":
        full_dataset = SEN12Dataset(
            data_folder=config["data_folder"],
            mode="auto"
        )
    else:
        full_dataset = SEN12Dataset(
            data_folder=config["data_folder"],
            csv_file=config["csv_file"],
            mode="csv"
        )

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
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

    generator = UNetGenerator(in_channels=5, out_channels=3)
    discriminator = PatchGANDiscriminator(in_channels=8)

    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"  Generator       : {gen_params:,} parameters")
    print(f"  Discriminator   : {disc_params:,} parameters\n")

    trainer = Pix2PixTrainer(
        generator, discriminator, train_loader, val_loader, device, config
    )

    # Initialiser le fichier CSV de logging
    log_path = Path(config["results_dir"]) / "training_log.csv"
    Pix2PixTrainer.init_training_log(log_path)

    print("=" * 60)
    print("TRAINING START")
    print("=" * 60 + "\n")

    start_time = time.time()

    try:
        for epoch in range(config["num_epochs"]):
            losses = trainer.train_epoch()

            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}:")
            print(
                f"   Loss G: {losses['loss_G']:.4f} "
                + f"(GAN: {losses['loss_G_gan']:.4f}, L1: {losses['loss_G_l1']:.4f})"
            )
            print(
                f"   Loss D: {losses['loss_D']:.4f} "
                + f"(Real: {losses['loss_D_real']:.4f}, Fake: {losses['loss_D_fake']:.4f})"
            )

            if (epoch + 1) % config["val_freq"] == 0:
                metrics = trainer.validate()
                print(
                    f"   Validation - PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}"
                )
                trainer.save_validation_results(n_samples=4)
                
                # Logger les métriques dans le CSV
                trainer.log_metrics(log_path, losses, metrics)
                
                if config.get("save_individual", True):
                    trainer.save_individual_images(n_samples=4)
            else:
                # Logger uniquement les losses (sans validation)
                trainer.log_metrics(log_path, losses, {})

            if (epoch + 1) % config["save_freq"] == 0:
                trainer.save_checkpoint()

        trainer.save_checkpoint("final_model.pth")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        trainer.save_checkpoint("interrupted_model.pth")

    elapsed_time = time.time() - start_time
    print(f"\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"  Total time      : {elapsed_time / 3600:.2f} hours")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
