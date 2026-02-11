"""
Script de nettoyage du dataset SEN12 Multisaison

Filtre les images corrompues ou sans contenu significatif.
Critères de validation AJUSTÉS :
- SAR : écart-type > 0.0001 et valeur max > 0.001 (élimine océans/corruption)
- Optique : écart-type moyen RGB > 3.0 (abaissé pour supporter Winter)

Modes de fonctionnement :
- Mode 1 : Nettoyer un CSV existant
- Mode 2 : Scanner tous les dossiers et créer un CSV nettoyé (RECOMMANDÉ)
"""

import rasterio
import numpy as np
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def discover_all_triplets(data_root):
    """
    Découvre tous les triplets disponibles dans le dataset.
    
    Chaque triplet est composé de :
    - S1 : Image SAR (Sentinel-1)
    - S2 : Image optique claire (Sentinel-2)
    - S2_cloudy : Image optique nuageuse (Sentinel-2)
    
    Args:
        data_root (str): Chemin racine du dataset
        
    Returns:
        list: Liste de dictionnaires contenant les informations de chaque triplet
    """
    data_root = Path(data_root)
    
    # Chercher TOUS les dossiers qui matchent les patterns (summer, winter, etc.)
    s1_folders = list(data_root.glob("ROIs*_s1"))
    s2_folders = list(data_root.glob("ROIs*_s2"))
    s2_cloudy_folders = list(data_root.glob("ROIs*_s2_cloudy"))
    
    print("Découverte des triplets disponibles...")
    print(f"  Dossiers S1         : {len(s1_folders)} trouvés")
    print(f"  Dossiers S2         : {len(s2_folders)} trouvés")
    print(f"  Dossiers S2 Cloudy  : {len(s2_cloudy_folders)} trouvés")
    for folder in s1_folders + s2_folders + s2_cloudy_folders:
        print(f"    - {folder.name}")
    print()
    
    s1_files = {}
    for s1_folder in s1_folders:
        if s1_folder.exists():
            for parent_folder in s1_folder.iterdir():
                if parent_folder.is_dir():
                    for tif_file in parent_folder.glob("*.tif"):
                        filename = tif_file.stem
                        # Extraire le patch_id de manière flexible
                        if "_s1_" in filename:
                            patch_id = filename.split("_s1_")[-1]
                            # Ajouter le nom de la saison au patch_id pour éviter les doublons
                            season = s1_folder.name.split("_")[1]  # 'summer' ou 'winter'
                            full_patch_id = f"{season}_{patch_id}"
                            s1_files[full_patch_id] = (s1_folder.name, parent_folder.name, tif_file.name)
    print(f"  S1         : {len(s1_files)} fichiers uniques")
    
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
                            s2_files[full_patch_id] = (s2_folder.name, parent_folder.name, tif_file.name)
    print(f"  S2         : {len(s2_files)} fichiers uniques")
    
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
                            s2_cloudy_files[full_patch_id] = (s2_cloudy_folder.name, parent_folder.name, tif_file.name)
    print(f"  S2 Cloudy  : {len(s2_cloudy_files)} fichiers uniques")
    
    s1_ids = set(s1_files.keys())
    s2_ids = set(s2_files.keys())
    s2_cloudy_ids = set(s2_cloudy_files.keys())
    
    complete_ids = s1_ids & s2_ids & s2_cloudy_ids
    
    print(f"\n  Triplets complets : {len(complete_ids)}")
    
    # Statistiques par saison
    seasons = {}
    for patch_id in complete_ids:
        season = patch_id.split("_")[0]
        seasons[season] = seasons.get(season, 0) + 1
    
    for season, count in sorted(seasons.items()):
        print(f"    - {season.capitalize()}: {count} triplets")
    
    if len(complete_ids) > 0 and len(complete_ids) < 10:
        samples = sorted(list(complete_ids))[:5]
        print(f"  Exemples          : {samples}")
    print()
    
    triplets = []
    for patch_id in sorted(complete_ids):
        s1_root, s1_parent, s1_file = s1_files[patch_id]
        s2_root, s2_parent, s2_file = s2_files[patch_id]
        s2_cloudy_root, s2_cloudy_parent, s2_cloudy_file = s2_cloudy_files[patch_id]
        
        triplets.append({
            'id': patch_id,
            's1_root_folder': s1_root,
            's1_folder': s1_parent,
            's1_file': s1_file,
            's2_root_folder': s2_root,
            's2_folder': s2_parent,
            's2_file': s2_file,
            's2_cloudy_root_folder': s2_cloudy_root,
            's2_cloudy_folder': s2_cloudy_parent,
            's2_cloudy_file': s2_cloudy_file
        })
    
    return triplets


def validate_triplet(row, data_root):
    """
    Valide un triplet selon les critères de qualité.
    
    Critères AJUSTÉS pour multisaison :
    - SAR : écart-type > 0.0001 et max > 0.001 (élimine océans/corruption)
    - Optique : écart-type moyen RGB > 3.0 (abaissé de 10.0 pour Winter)
    
    Args:
        row (dict): Informations du triplet
        data_root (str): Chemin racine du dataset
        
    Returns:
        bool: True si le triplet est valide, False sinon
    """
    # Support du nouveau format (avec root_folders) et de l'ancien format
    s1_root = row.get('s1_root_folder', 'ROIs1868_summer_s1')
    s2_root = row.get('s2_root_folder', 'ROIs1868_summer_s2')
    
    s1_path = os.path.join(data_root, s1_root, row['s1_folder'], row['s1_file'])
    s2_path = os.path.join(data_root, s2_root, row['s2_folder'], row['s2_file'])
    
    try:
        with rasterio.open(s1_path) as src:
            s1_data = src.read(1)
        with rasterio.open(s2_path) as src:
            s2_data = src.read([2, 3, 4])

        # Validation SAR (inchangée)
        if s1_data.std() < 0.0001 or s1_data.max() < 0.001:
            return False

        # Validation Optique : seuil différencié selon la saison

        optical_std = np.mean([s2_data[i].std() for i in range(3)])
        # Détection de la saison
        season = row.get('id', '').split('_')[0].lower()
        print(f"[VALIDATION] Saison: {season}, optical_std: {optical_std:.2f}")
        if season == 'winter':
            if optical_std < 2.0:
                print("  -> Rejeté (Winter, optical_std < 2.0)")
                return False
        else:
            if optical_std < 3.0:
                print("  -> Rejeté (Summer, optical_std < 3.0)")
                return False

        return True

    except Exception as e:
        return False


def clean_dataset_from_csv(csv_input, csv_output, data_root):
    """
    Nettoie un CSV existant en validant chaque triplet.
    
    Args:
        csv_input (str): Chemin du CSV d'entrée
        csv_output (str): Chemin du CSV de sortie
        data_root (str): Chemin racine du dataset
    """
    df = pd.read_csv(csv_input)
    
    print(f"Validation de {len(df)} triplets...")
    valid_rows = []

    optical_std_stats = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validation"):
        triplet = {
            'id': row['id'],
            's1_folder': row.get('s1_folder', row.get('s1')),
            's1_file': row['s1_file'],
            's2_folder': row.get('s2_folder', row.get('s2')),
            's2_file': row['s2_file'],
            's2_cloudy_folder': row.get('s2_cloudy_folder', row.get('s2_cloudy')),
            's2_cloudy_file': row['s2_cloudy_file'],
            's1_root_folder': row.get('s1_root_folder', 'ROIs1868_summer_s1'),
            's2_root_folder': row.get('s2_root_folder', 'ROIs1868_summer_s2'),
        }
        # Collecte optical_std pour Winter
        season = triplet['id'].split('_')[0].lower()
        if season == 'winter':
            try:
                s2_path = os.path.join(data_root, triplet['s2_root_folder'], triplet['s2_folder'], triplet['s2_file'])
                with rasterio.open(s2_path) as src:
                    s2_data = src.read([2, 3, 4])
                optical_std = np.mean([s2_data[i].std() for i in range(3)])
                optical_std_stats.append({'id': triplet['id'], 'optical_std': optical_std})
            except Exception:
                optical_std_stats.append({'id': triplet['id'], 'optical_std': None})
        if validate_triplet(triplet, data_root):
            valid_rows.append(triplet)
    # Export CSV temporaire pour analyse
    pd.DataFrame(optical_std_stats).to_csv('winter_optical_std_stats.csv', index=False)

    new_df = pd.DataFrame(valid_rows)
    new_df.to_csv(csv_output, index=False, header=True)
    
    print(f"\n{'='*60}")
    print(f"Triplets valides   : {len(new_df)}/{len(df)} ({len(new_df)/len(df)*100:.1f}%)")
    print(f"Fichier sauvegardé : {csv_output}")
    print(f"{'='*60}")


def clean_dataset_full_scan(csv_output, data_root):
    """
    Scanne tous les dossiers, trouve les triplets et les valide.
    
    Args:
        csv_output (str): Chemin du CSV de sortie
        data_root (str): Chemin racine du dataset
    """
    all_triplets = discover_all_triplets(data_root)
    
    if len(all_triplets) == 0:
        print("Aucun triplet trouvé. Vérifiez la structure du dataset.")
        return
    
    print(f"Validation de {len(all_triplets)} triplets...")
    valid_rows = []
    optical_std_stats = []
    sar_stats = []
    for triplet in tqdm(all_triplets, desc="Validation"):
        # Collecte optical_std pour Winter
        season = triplet['id'].split('_')[0].lower()
        if season == 'winter':
            try:
                s2_path = os.path.join(data_root, triplet['s2_root_folder'], triplet['s2_folder'], triplet['s2_file'])
                with rasterio.open(s2_path) as src:
                    s2_data = src.read([2, 3, 4])
                optical_std = np.mean([s2_data[i].std() for i in range(3)])
                optical_std_stats.append({'id': triplet['id'], 'optical_std': optical_std})
            except Exception:
                optical_std_stats.append({'id': triplet['id'], 'optical_std': None})
            # Ajout stats SAR
            try:
                s1_path = os.path.join(data_root, triplet['s1_root_folder'], triplet['s1_folder'], triplet['s1_file'])
                with rasterio.open(s1_path) as src:
                    s1_data = src.read(1)
                sar_min = float(np.min(s1_data))
                sar_max = float(np.max(s1_data))
                sar_std = float(np.std(s1_data))
                sar_stats.append({'id': triplet['id'], 'sar_min': sar_min, 'sar_max': sar_max, 'sar_std': sar_std})
            except Exception:
                sar_stats.append({'id': triplet['id'], 'sar_min': None, 'sar_max': None, 'sar_std': None})
        if validate_triplet(triplet, data_root):
            valid_rows.append(triplet)


    new_df = pd.DataFrame(valid_rows)
    csv_data = []
    for _, row in new_df.iterrows():
        csv_data.append([
            row['id'],
            row['s1_root_folder'],
            row['s1_folder'],
            row['s1_file'],
            row['s2_root_folder'],
            row['s2_folder'],
            row['s2_file'],
            row['s2_cloudy_root_folder'],
            row['s2_cloudy_folder'],
            row['s2_cloudy_file']
        ])

    csv_df = pd.DataFrame(csv_data, columns=[
        'id', 
        's1_root_folder', 's1_folder', 's1_file', 
        's2_root_folder', 's2_folder', 's2_file', 
        's2_cloudy_root_folder', 's2_cloudy_folder', 's2_cloudy_file'
    ])
    csv_df.to_csv(csv_output, index=False, header=True)

    print(f"\n{'='*60}")
    print(f"Triplets valides   : {len(valid_rows)}/{len(all_triplets)} ({len(valid_rows)/len(all_triplets)*100:.1f}%)")
    print(f"Fichier sauvegardé : {csv_output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("="*60)
    print("NETTOYAGE DU DATASET SEN12")
    print("="*60 + "\n")
    
    clean_dataset_full_scan('data/sen_1_2/cleaned_triplets.csv', 'data/sen_1_2/')
