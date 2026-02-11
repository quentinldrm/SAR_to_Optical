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


# 1. PLOT GENERATION (Publication Quality)
def generate_plots(log_path):
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return

    df = pd.read_csv(log_path)
    # Set global plotting parameters for a professional look
    plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})

    # FIG 1: GAN LOSS (Generator vs Discriminator)
    plt.figure(figsize=(9, 5))
    plt.plot(
        df["Epoch"],
        df["Loss_G_GAN"],
        label="Gen Adversarial Loss",
        color="#1f77b4",
        linewidth=1.5,
    )
    plt.plot(
        df["Epoch"], df["Loss_D"], label="Discrim. Loss", color="#d62728", alpha=0.6
    )
    plt.title("Adversarial Training Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "fig_loss_gan.png", dpi=300)
    plt.close()

    # FIG 2: L1 LOSS (Spatial Fidelity)
    plt.figure(figsize=(9, 5))
    plt.plot(
        df["Epoch"],
        df["Loss_G_L1"],
        color="#2ca02c",
        linewidth=2,
        label="Generator L1 Loss",
    )
    plt.title("Spatial Reconstruction Accuracy (L1)")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "fig_loss_l1.png", dpi=300)
    plt.close()

    # FIG 3: VALIDATION METRICS (PSNR/SSIM)
    # Filter out entries where validation metrics haven't been recorded (0)
    val_df = df[df["PSNR_Val"] > 0]
    if not val_df.empty:
        fig, ax1 = plt.subplots(figsize=(9, 5))

        # Primary axis: PSNR
        ax1.plot(
            val_df["Epoch"],
            val_df["PSNR_Val"],
            color="#1f77b4",
            marker="o",
            markersize=3,
            label="PSNR",
        )
        ax1.set_ylabel("PSNR (dB)", color="#1f77b4")

        # Secondary axis: SSIM
        ax2 = ax1.twinx()
        ax2.plot(
            val_df["Epoch"],
            val_df["SSIM_Val"],
            color="#ff7f0e",
            marker="s",
            markersize=3,
            label="SSIM",
        )
        ax2.set_ylabel("SSIM", color="#ff7f0e")

        plt.title("Validation Metrics Trends")
        plt.grid(True, alpha=0.2)
        fig.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig_metrics.png", dpi=300)
        plt.close()


# 2. RADAR NORMALIZATION (Mirrors logic in train.py)
def denormalize_sar(data):
    """
    Robust normalization for Sentinel-1 (SAR) data.
    Uses percentile clipping to handle outliers before scaling to [0, 1].
    """
    data = np.nan_to_num(data)
    # Aggressive 1%-99% clipping for better visual contrast in figures
    vmin, vmax = np.percentile(data, [1, 99])
    data = np.clip(data, vmin, vmax)
    data = (data - vmin) / (vmax - vmin + 1e-8)
    return data


def denormalize_s2(data):
    """
    Normalization for Sentinel-2 (Optical) data.
    Clips at 3000 (reflectance ~0.3) for a natural visual rendering.
    """
    data = data / 3000.0
    return np.clip(data, 0, 1)


def extract_visuals(csv_path, data_root):
    """
    Extracts random sample pairs (SAR, Clear, Cloudy) for different seasons
    to demonstrate dataset variety in reports.
    """
    df = pd.read_csv(csv_path)
    for season in ["summer", "winter"]:
        # Filter samples based on the root folder name
        subset = df[df["s1_root_folder"].str.contains(season, case=False)]
        if subset.empty:
            continue

        sample = subset.sample(1).iloc[0]

        # Construct file paths
        p_s1 = (
            data_root
            / sample["s1_root_folder"]
            / sample["s1_folder"]
            / sample["s1_file"]
        )
        p_s2 = (
            data_root
            / sample["s2_root_folder"]
            / sample["s2_folder"]
            / sample["s2_file"]
        )
        p_c = (
            data_root
            / sample["s2_cloudy_root_folder"]
            / sample["s2_cloudy_folder"]
            / sample["s2_cloudy_file"]
        )

        try:
            # Process SAR (Single Band VV/VH)
            with rasterio.open(p_s1) as src:
                sar_img = denormalize_sar(src.read(1))
                Image.fromarray((sar_img * 255).astype(np.uint8)).save(
                    OUTPUT_DIR / f"{season}_SAR_visual.png"
                )

            # Process S2 Clear (RGB Bands)
            with rasterio.open(p_s2) as src:
                # Transpose from (C, H, W) to (H, W, C) for PIL
                s2_img = denormalize_s2(src.read([1, 2, 3])).transpose(1, 2, 0)
                Image.fromarray((s2_img * 255).astype(np.uint8)).save(
                    OUTPUT_DIR / f"{season}_S2_Clear.png"
                )

            # Process S2 Cloudy (RGB Bands)
            with rasterio.open(p_c) as src:
                c_img = denormalize_s2(src.read([1, 2, 3])).transpose(1, 2, 0)
                Image.fromarray((c_img * 255).astype(np.uint8)).save(
                    OUTPUT_DIR / f"{season}_S2_Cloudy.png"
                )

            print(f"{season.capitalize()} visuals extracted successfully.")

        except Exception as e:
            print(f"Error during {season} extraction: {e}")


if __name__ == "__main__":
    print("Generating training curves and sample visuals...")
    generate_plots(CSV_LOG_PATH)
    extract_visuals(DATASET_CSV, DATA_ROOT)
    print(f"\nScientific figures ready in: {OUTPUT_DIR.absolute()}")
