# Multimodal SAR-optical fusion for cloud removal in Sentinel-2 imagery

**Cloud removal framework using conditional GANs and multi-seasonal SAR-optical data fusion** *Master 2 Earth Observation and Geoinformatics (OTG) • University of Strasbourg • 2026*

---

## Summary

This personal project implements a multimodal deep learning framework for restoring optical Sentinel-2 imagery obscured by cloud cover. By leveraging the cloud-penetrating capabilities of Sentinel-1 Synthetic Aperture Radar (SAR), the model synthesizes cloud-free RGB patches even under dense atmospheric interference.

The core of the study relies on a modified Pix2pix architecture and an automated data curation pipeline specifically designed to handle seasonal variability, ensuring robust performance across both summer and winter landscapes.

---

## Features

The framework consists of three main technical components:

### 1. Automated Data Curation (`clean_dataset.py`)

- Statistical filtering of SAR/Optical triplets to remove corrupted tiles, sensor artifacts, or non-informative samples.
- Adaptive seasonal contrast thresholds ($\sigma > 10$ for summer, $\sigma > 3$ for winter) to preserve valid textures in dormant landscapes.
- Automated rejection of samples with no backscatter variance based on SAR standard deviation thresholds ($\sigma < 0.0001$).

### 2. Multimodal Pix2Pix Architecture

- 5-channel early fusion generator receiving Sentinel-1 VV/VH polarizations and cloudy Sentinel-2 RGB channels.
- U-Net backbone with skip connections to transfer SAR-derived structural geometries directly to the optical output.
- PatchGAN discriminator ($70 \times 70$) to enforce high-frequency texture realism and reduce blurring effects.

### 3. Multi-seasonal Training Protocol

- Integration of summer and winter samples from the SEN12MS dataset to decouple land structures from seasonal phenology.
- Optimization using Adam with Automatic Mixed Precision (AMP) to accelerate computation on an NVIDIA RTX 3080 GPU.
- Data augmentation including random flips to improve generalization, particularly for the smaller winter subset.

---

## Data Sources

### Remote Sensing Data

#### **Sentinel-1 (SAR)**
- **Source**: ESA Sentinel-1 mission.
- **Format**: Dual-polarization (VV + VH) backscatter.
- **Role**: Provides the geometric backbone and structural information unaffected by weather.

#### **Sentinel-2 (Optical)**
- **Source**: ESA Sentinel-2 mission.
- **Format**: RGB channels (Visible spectrum).
- **Role**: Provides the target spectral reflectance for training and evaluation.

#### **SEN12MS Dataset**
- **Source**: Curated collection of synchronized Sentinel-1 and Sentinel-2 patches.
- **Volume**: Final curated dataset of 4,464 high-quality triplets (4,069 summer and 395 winter samples).

---

## Methodology

### Image-to-Image Translation
The project treats cloud removal as a domain translation problem. The generator is trained to synthesize an image that is indistinguishable from real clear imagery by the discriminator.

### Loss Function
The training utilizes a combined objective function to ensure both adversarial realism and spatial consistency:

$$G^{*}=arg~min_{G}max_{D}\mathcal{L}_{cGAN}(G,D)+\lambda\mathcal{L}_{L1}(G)$$

The $L1$ regularization term ($\lambda=100$) minimizes pixel-wise error, while the GAN loss drives the production of realistic high-frequency details.

---

## Performance

### Quantitative Results
The model reached stable convergence after 200 epochs with the following validation metrics:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **PSNR** | 27.18 dB | High pixel-wise reconstruction accuracy compared to clear ground truth. |
| **SSIM** | 0.791 | Strong preservation of structural features such as urban grids and field layouts. |

### Qualitative Assessment
Visual analysis confirms that the model successfully reconstructs fine-scale ground features even under opaque cloud layers. The early fusion approach effectively uses SAR backscatter as a structural guide to restore missing optical information with high spatial precision.

---

## Author

**Quentin Ledermann** Master 2 OTG - University of Strasbourg (2025-2026)  
Email: quentinledermann@outlook.fr

---

## Acknowledgments

- **ESA** for Sentinel-1 and Sentinel-2 data.
- **Schmitt et al. (2019)** for the SEN12MS dataset.
- **Isola et al. (2017)** for the Pix2Pix architecture.
- **University of Strasbourg** - Master OTG program.

---

**© 2026 - Personal Research Project - University of Strasbourg**
