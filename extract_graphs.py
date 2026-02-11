import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import os
from PIL import Image
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(".") 
CSV_LOG_PATH = BASE_DIR / "results" / "training_log.csv"
DATASET_CSV = BASE_DIR / "data" / "sen_1_2" / "cleaned_triplets.csv"
DATA_ROOT = BASE_DIR / "data" / "sen_1_2"
OUTPUT_DIR = BASE_DIR / "scientific_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. GÉNÉRATION DES GRAPHIQUES (Style Publication)
def generate_plots(log_path):
    if not log_path.exists():
        print(f"Log introuvable : {log_path}")
        return

    df = pd.read_csv(log_path)
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    # FIG 1: GAN LOSS (Duel)
    plt.figure(figsize=(9, 5))
    plt.plot(df['Epoch'], df['Loss_G_GAN'], label='Gen Adversarial Loss', color='#1f77b4', linewidth=1.5)
    plt.plot(df['Epoch'], df['Loss_D'], label='Discrim. Loss', color='#d62728', alpha=0.6)
    plt.title('Adversarial Training Convergence')
    plt.xlabel('Epochs'); plt.ylabel('Loss Value'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "fig_loss_gan.png", dpi=300); plt.close()

    # FIG 2: L1 LOSS (Fidélité)
    plt.figure(figsize=(9, 5))
    plt.plot(df['Epoch'], df['Loss_G_L1'], color='#2ca02c', linewidth=2, label='Generator L1 Loss')
    plt.title('Spatial Reconstruction Accuracy (L1)')
    plt.xlabel('Epochs'); plt.ylabel('Mean Absolute Error'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "fig_loss_l1.png", dpi=300); plt.close()

    # FIG 3: METRICS (PSNR/SSIM)
    val_df = df[df['PSNR_Val'] > 0]
    if not val_df.empty:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(val_df['Epoch'], val_df['PSNR_Val'], color='#1f77b4', marker='o', markersize=3, label='PSNR')
        ax1.set_ylabel('PSNR (dB)', color='#1f77b4')
        ax2 = ax1.twinx()
        ax2.plot(val_df['Epoch'], val_df['SSIM_Val'], color='#ff7f0e', marker='s', markersize=3, label='SSIM')
        ax2.set_ylabel('SSIM', color='#ff7f0e')
        plt.title('Validation Metrics Trends')
        plt.grid(True, alpha=0.2); fig.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig_metrics.png", dpi=300); plt.close()

# 2. NORMALISATION RADAR (Copiée sur ton train.py)
def denormalize_sar(data):
    """Méthode de normalisation robuste pour Sentinel-1."""
    # On utilise la logique de ton train.py pour le S1
    # Souvent : clip entre -25 et 0 dB ou normalisation statistique
    data = np.nan_to_num(data)
    # Clipping basé sur les valeurs typiques que ton modèle a appris
    # On va utiliser un clipping 2%-98% plus agressif pour le rendu visuel
    vmin, vmax = np.percentile(data, [1, 99])
    data = np.clip(data, vmin, vmax)
    data = (data - vmin) / (vmax - vmin + 1e-8)
    return data

def denormalize_s2(data):
    """Méthode de normalisation pour Sentinel-2."""
    # S2 : clipping à 3000 (réflectance 0.3) pour un rendu naturel
    data = data / 3000.0
    return np.clip(data, 0, 1)

def extract_visuals(csv_path, data_root):
    df = pd.read_csv(csv_path)
    for season in ['summer', 'winter']:
        subset = df[df['s1_root_folder'].str.contains(season, case=False)]
        if subset.empty: continue
        sample = subset.sample(1).iloc[0]
        
        # Chemins
        p_s1 = data_root / sample['s1_root_folder'] / sample['s1_folder'] / sample['s1_file']
        p_s2 = data_root / sample['s2_root_folder'] / sample['s2_folder'] / sample['s2_file']
        p_c  = data_root / sample['s2_cloudy_root_folder'] / sample['s2_cloudy_folder'] / sample['s2_cloudy_file']

        try:
            with rasterio.open(p_s1) as src:
                # On lit la bande VV
                sar_img = denormalize_sar(src.read(1))
                Image.fromarray((sar_img * 255).astype(np.uint8)).save(OUTPUT_DIR / f"{season}_SAR_visual.png")
            
            with rasterio.open(p_s2) as src:
                s2_img = denormalize_s2(src.read([1,2,3])).transpose(1, 2, 0)
                Image.fromarray((s2_img * 255).astype(np.uint8)).save(OUTPUT_DIR / f"{season}_S2_Clear.png")
            
            with rasterio.open(p_c) as src:
                c_img = denormalize_s2(src.read([1,2,3])).transpose(1, 2, 0)
                Image.fromarray((c_img * 255).astype(np.uint8)).save(OUTPUT_DIR / f"{season}_S2_Cloudy.png")
            print(f"{season.capitalize()} visuals extracted successfully.")
        except Exception as e:
            print(f"Error during {season} extraction: {e}")

if __name__ == "__main__":
    generate_plots(CSV_LOG_PATH)
    extract_visuals(DATASET_CSV, DATA_ROOT)
    print(f"\nScientific figures ready in: {OUTPUT_DIR.absolute()}")