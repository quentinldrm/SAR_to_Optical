"""
SEN12 Multi-season Dataset Cleaning Script

Filters out corrupted images or those without significant content.
ADJUSTED Validation Criteria:
- SAR: standard deviation > 0.0001 and max value > 0.001 (removes oceans/corrupted files)
- Optical: mean RGB standard deviation > 3.0 (lowered to support Winter conditions)

Operating Modes:
- Mode 1: Clean an existing CSV
- Mode 2: Scan all directories and create a cleaned CSV (RECOMMENDED)
"""

import rasterio
import numpy as np
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def discover_all_triplets(data_root):
    """
    Scans the directory structure to identify available triplets.

    Each triplet consists of:
    - S1: SAR Image (Sentinel-1)
    - S2: Clear optical image (Sentinel-2)
    - S2_cloudy: Cloudy optical image (Sentinel-2)

    Args:
        data_root (str): Root path of the dataset

    Returns:
        list: List of dictionaries containing file paths for each triplet
    """
    data_root = Path(data_root)

    # Search for ALL folders matching seasonal patterns (summer, winter, etc.)
    s1_folders = list(data_root.glob("ROIs*_s1"))
    s2_folders = list(data_root.glob("ROIs*_s2"))
    s2_cloudy_folders = list(data_root.glob("ROIs*_s2_cloudy"))

    print("Discovering available triplets...")
    print(f"  S1 Folders         : {len(s1_folders)} found")
    print(f"  S2 Folders         : {len(s2_folders)} found")
    print(f"  S2 Cloudy Folders  : {len(s2_cloudy_folders)} found")
    for folder in s1_folders + s2_folders + s2_cloudy_folders:
        print(f"    - {folder.name}")
    print()

    # Map S1 files: key = {season}_{patch_id}
    s1_files = {}
    for s1_folder in s1_folders:
        if s1_folder.exists():
            for parent_folder in s1_folder.iterdir():
                if parent_folder.is_dir():
                    for tif_file in parent_folder.glob("*.tif"):
                        filename = tif_file.stem
                        if "_s1_" in filename:
                            patch_id = filename.split("_s1_")[-1]
                            season = s1_folder.name.split("_")[1]
                            full_patch_id = f"{season}_{patch_id}"
                            s1_files[full_patch_id] = (
                                s1_folder.name,
                                parent_folder.name,
                                tif_file.name,
                            )
    print(f"  S1 unique files    : {len(s1_files)}")

    # Map S2 (Clear) files
    s2_files = {}
    for s2_folder in s2_folders:
        if s2_folder.exists():
            for parent_folder in s2_folder.iterdir():
                if parent_folder.is_dir():
                    for tif_file in parent_folder.glob("*.tif"):
                        filename = tif_file.stem
                        if "_s2_" in filename and "cloudy" not in filename:
                            patch_id = filename.split("_s2_")[-1]
                            season = s2_folder.name.split("_")[1]
                            full_patch_id = f"{season}_{patch_id}"
                            s2_files[full_patch_id] = (
                                s2_folder.name,
                                parent_folder.name,
                                tif_file.name,
                            )
    print(f"  S2 unique files    : {len(s2_files)}")

    # Map S2 (Cloudy) files
    s2_cloudy_files = {}
    for s2_cloudy_folder in s2_cloudy_folders:
        if s2_cloudy_folder.exists():
            for parent_folder in s2_cloudy_folder.iterdir():
                if parent_folder.is_dir():
                    for tif_file in parent_folder.glob("*.tif"):
                        filename = tif_file.stem
                        if "_s2_cloudy_" in filename:
                            patch_id = filename.split("_s2_cloudy_")[-1]
                            season = s2_cloudy_folder.name.split("_")[1]
                            full_patch_id = f"{season}_{patch_id}"
                            s2_cloudy_files[full_patch_id] = (
                                s2_cloudy_folder.name,
                                parent_folder.name,
                                tif_file.name,
                            )
    print(f"  S2 Cloudy files    : {len(s2_cloudy_files)}")

    # Find the intersection of all sets to ensure we have complete triplets
    s1_ids = set(s1_files.keys())
    s2_ids = set(s2_files.keys())
    s2_cloudy_ids = set(s2_cloudy_files.keys())

    complete_ids = s1_ids & s2_ids & s2_cloudy_ids

    print(f"\n  Complete Triplets  : {len(complete_ids)}")

    # Breakdown stats by season
    seasons = {}
    for patch_id in complete_ids:
        season = patch_id.split("_")[0]
        seasons[season] = seasons.get(season, 0) + 1

    for season, count in sorted(seasons.items()):
        print(f"    - {season.capitalize()}: {count} triplets")

    if 0 < len(complete_ids) < 10:
        samples = sorted(list(complete_ids))[:5]
        print(f"  Samples            : {samples}")
    print()

    triplets = []
    for patch_id in sorted(complete_ids):
        s1_root, s1_parent, s1_file = s1_files[patch_id]
        s2_root, s2_parent, s2_file = s2_files[patch_id]
        s2_cloudy_root, s2_cloudy_parent, s2_cloudy_file = s2_cloudy_files[patch_id]

        triplets.append(
            {
                "id": patch_id,
                "s1_root_folder": s1_root,
                "s1_folder": s1_parent,
                "s1_file": s1_file,
                "s2_root_folder": s2_root,
                "s2_folder": s2_parent,
                "s2_file": s2_file,
                "s2_cloudy_root_folder": s2_cloudy_root,
                "s2_cloudy_folder": s2_cloudy_parent,
                "s2_cloudy_file": s2_cloudy_file,
            }
        )

    return triplets


def validate_triplet(row, data_root):
    """
    Validates a triplet based on quality criteria.

    Adjusted thresholds:
    - SAR: checks for non-zero variance and signal presence (removes empty/ocean patches).
    - Optical: ensures enough texture/contrast. Threshold is lower for Winter to avoid
      rejecting snowy but valid landscapes.
    """
    # Fallback for old CSV formats without root_folder columns
    s1_root = row.get("s1_root_folder", "ROIs1868_summer_s1")
    s2_root = row.get("s2_root_folder", "ROIs1868_summer_s2")

    s1_path = os.path.join(data_root, s1_root, row["s1_folder"], row["s1_file"])
    s2_path = os.path.join(data_root, s2_root, row["s2_folder"], row["s2_file"])

    try:
        with rasterio.open(s1_path) as src:
            s1_data = src.read(1)
        with rasterio.open(s2_path) as src:
            # Reads RGB bands (adjust indices if your TIF structure differs)
            s2_data = src.read([2, 3, 4])

        # SAR Validation (Unchanged)
        if s1_data.std() < 0.0001 or s1_data.max() < 0.001:
            return False

        # Optical Validation: Threshold varies by season
        optical_std = np.mean([s2_data[i].std() for i in range(3)])
        season = row.get("id", "").split("_")[0].lower()

        print(f"[VALIDATION] Season: {season}, optical_std: {optical_std:.2f}")

        if season == "winter":
            if optical_std < 2.0:
                print("  -> Rejected (Winter, optical_std < 2.0)")
                return False
        else:
            if optical_std < 3.0:
                print("  -> Rejected (Summer/Other, optical_std < 3.0)")
                return False

        return True

    except Exception:
        return False


def clean_dataset_from_csv(csv_input, csv_output, data_root):
    """
    Cleans an existing CSV by validating each triplet entry.
    """
    df = pd.read_csv(csv_input)

    print(f"Validating {len(df)} triplets...")
    valid_rows = []
    optical_std_stats = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
        triplet = {
            "id": row["id"],
            "s1_folder": row.get("s1_folder", row.get("s1")),
            "s1_file": row["s1_file"],
            "s2_folder": row.get("s2_folder", row.get("s2")),
            "s2_file": row["s2_file"],
            "s2_cloudy_folder": row.get("s2_cloudy_folder", row.get("s2_cloudy")),
            "s2_cloudy_file": row["s2_cloudy_file"],
            "s1_root_folder": row.get("s1_root_folder", "ROIs1868_summer_s1"),
            "s2_root_folder": row.get("s2_root_folder", "ROIs1868_summer_s2"),
        }

        # Specific stats collection for Winter data analysis
        season = triplet["id"].split("_")[0].lower()
        if season == "winter":
            try:
                s2_path = os.path.join(
                    data_root,
                    triplet["s2_root_folder"],
                    triplet["s2_folder"],
                    triplet["s2_file"],
                )
                with rasterio.open(s2_path) as src:
                    s2_data = src.read([2, 3, 4])
                optical_std = np.mean([s2_data[i].std() for i in range(3)])
                optical_std_stats.append(
                    {"id": triplet["id"], "optical_std": optical_std}
                )
            except Exception:
                optical_std_stats.append({"id": triplet["id"], "optical_std": None})

        if validate_triplet(triplet, data_root):
            valid_rows.append(triplet)

    # Export temporary stats for fine-tuning thresholds
    pd.DataFrame(optical_std_stats).to_csv("winter_optical_std_stats.csv", index=False)

    new_df = pd.DataFrame(valid_rows)
    new_df.to_csv(csv_output, index=False, header=True)

    print(f"\n{'='*60}")
    print(
        f"Valid Triplets     : {len(new_df)}/{len(df)} ({len(new_df)/len(df)*100:.1f}%)"
    )
    print(f"File Saved         : {csv_output}")
    print(f"{'='*60}")


def clean_dataset_full_scan(csv_output, data_root):
    """
    Scans all directories, identifies triplets, and runs validation.
    """
    all_triplets = discover_all_triplets(data_root)

    if len(all_triplets) == 0:
        print("No triplets found. Please check your dataset directory structure.")
        return

    print(f"Validating {len(all_triplets)} triplets...")
    valid_rows = []

    for triplet in tqdm(all_triplets, desc="Validating"):
        # We perform the validation check here
        if validate_triplet(triplet, data_root):
            valid_rows.append(triplet)

    # Convert results to DataFrame and save
    new_df = pd.DataFrame(valid_rows)
    columns_order = [
        "id",
        "s1_root_folder",
        "s1_folder",
        "s1_file",
        "s2_root_folder",
        "s2_folder",
        "s2_file",
        "s2_cloudy_root_folder",
        "s2_cloudy_folder",
        "s2_cloudy_file",
    ]

    if not new_df.empty:
        new_df = new_df[columns_order]
        new_df.to_csv(csv_output, index=False, header=True)

    print(f"\n{'='*60}")
    print(
        f"Valid Triplets     : {len(valid_rows)}/{len(all_triplets)} ({len(valid_rows)/len(all_triplets)*100:.1f}%)"
    )
    print(f"File Saved         : {csv_output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("=" * 60)
    print("SEN12 DATASET CLEANING")
    print("=" * 60 + "\n")

    # Update these paths to match your local environment
    OUTPUT_CSV = "data/sen_1_2/cleaned_triplets.csv"
    DATA_ROOT = "data/sen_1_2/"

    clean_dataset_full_scan(OUTPUT_CSV, DATA_ROOT)
