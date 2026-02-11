# Multimodal SAR-optical fusion for cloud removal in Sentinel-2 imagery

[cite_start]**Cloud removal framework using conditional GANs and multi-seasonal SAR-optical data fusion** *Master 2 Earth Observation and Geoinformatics (OTG) • University of Strasbourg • 2026* [cite: 3, 4, 5]

---

## Summary

[cite_start]This personal project implements a multimodal deep learning framework for restoring optical Sentinel-2 imagery obscured by cloud cover[cite: 1, 6]. [cite_start]By leveraging the cloud-penetrating capabilities of Sentinel-1 Synthetic Aperture Radar (SAR), the model synthesizes cloud-free RGB patches even under dense atmospheric interference[cite: 7, 18, 19].

[cite_start]The core of the study relies on a modified Pix2pix architecture and an automated data curation pipeline specifically designed to handle seasonal variability, ensuring robust performance across both summer and winter landscapes[cite: 2, 8, 24, 25].

---

## Features

The framework consists of three main technical components:

### 1. Automated Data Curation (`clean_dataset.py`)

- [cite_start]Statistical filtering of SAR/Optical triplets to remove corrupted tiles, sensor artifacts, or non-informative samples[cite: 38, 39, 40].
- [cite_start]Adaptive seasonal contrast thresholds ($\sigma > 10$ for summer, $\sigma > 3$ for winter) to preserve valid textures in dormant landscapes[cite: 45, 46, 72].
- [cite_start]Automated rejection of samples with no backscatter variance based on SAR standard deviation thresholds[cite: 43, 44, 72].

### 2. Multimodal Pix2Pix Architecture

- [cite_start]5-channel early fusion generator receiving Sentinel-1 VV/VH polarizations and cloudy Sentinel-2 RGB channels[cite: 7, 75, 78].
- [cite_start]U-Net backbone with skip connections to transfer SAR-derived structural geometries directly to the optical output[cite: 76, 77, 105].
- [cite_start]PatchGAN discriminator ($70 \times 70$) to enforce high-frequency texture realism and reduce blurring effects[cite: 79, 80, 106].

### 3. Multi-seasonal Training Protocol

- [cite_start]Integration of summer and winter samples from the SEN12MS dataset to decouple land structures from seasonal phenology[cite: 24, 33, 35].
- [cite_start]Optimization using Adam with Automatic Mixed Precision (AMP) to accelerate computation on an NVIDIA RTX 3080 GPU[cite: 111, 112].
- [cite_start]Data augmentation including random flips to improve generalization, particularly for the smaller winter subset[cite: 113, 114].

---

## Data Sources

### Remote Sensing Data

#### **Sentinel-1 (SAR)**
- [cite_start]**Source**: ESA Sentinel-1 mission[cite: 15, 18, 246].
- [cite_start]**Format**: Dual-polarization (VV + VH) backscatter[cite: 23, 78].
- [cite_start]**Role**: Provides the geometric backbone and structural information unaffected by weather[cite: 19, 187, 190].

#### **Sentinel-2 (Optical)**
- [cite_start]**Source**: ESA Sentinel-2 mission[cite: 15, 222].
- [cite_start]**Format**: RGB channels (Visible spectrum)[cite: 9, 23, 78].
- [cite_start]**Role**: Provides the target spectral reflectance for training and evaluation[cite: 81, 125].

#### **SEN12MS Dataset**
- [cite_start]**Source**: Curated collection of synchronized Sentinel-1 and Sentinel-2 patches (Schmitt et al., 2019)[cite: 24, 32, 243].
- [cite_start]**Volume**: Final curated dataset of 4,464 high-quality triplets, including 4,069 summer and 395 winter samples[cite: 8, 47].

---

## Methodology

### Image-to-Image Translation
[cite_start]The project treats cloud removal as a domain translation problem[cite: 22]. [cite_start]The generator is trained to synthesize an image that is indistinguishable from real clear imagery by the discriminator[cite: 74, 81].

### Loss Function
[cite_start]The training utilizes a combined objective function to ensure both adversarial realism and spatial consistency[cite: 109]:
$$G^{*}=arg~min_{G}max_{D}\mathcal{L}_{cGAN}(G,D)+\lambda\mathcal{L}_{L1}(G)$$
[cite_start]The $L1$ regularization term ($\lambda=100$) minimizes pixel-wise error, while the GAN loss drives the production of realistic high-frequency details[cite: 110, 121, 179].

---

## Performance

### Quantitative Results
[cite_start]The model reached stable convergence after 200 epochs with the following validation metrics[cite: 120, 180]:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **PSNR** | 27.18 dB | [cite_start]High pixel-wise reconstruction accuracy compared to clear ground truth[cite: 9, 124, 180]. |
| **SSIM** | 0.791 | [cite_start]Strong preservation of structural features such as urban grids and field layouts[cite: 9, 126, 180]. |

### Qualitative Assessment
[cite_start]Visual analysis confirms that the model successfully reconstructs fine-scale ground features even under opaque cloud layers[cite: 10, 26, 186]. [cite_start]The early fusion approach effectively uses SAR backscatter as a structural guide to restore missing optical information with high spatial precision[cite: 82, 190, 203].

---

## Author

[cite_start]**Quentin Ledermann** [cite: 3, 27]  
[cite_start]Master 2 OTG - University of Strasbourg (2025-2026) [cite: 4, 5]  
Email: quentinledermann@outlook.fr

---

## Acknowledgments

- [cite_start]**ESA** for Sentinel-1 and Sentinel-2 data[cite: 15, 222, 246].
- **Schmitt et al. (2019)[cite_start]** for the SEN12MS dataset[cite: 24, 243].
- **Isola et al. (2017)[cite_start]** for the Pix2Pix architecture[cite: 21, 233].
- [cite_start]**University of Strasbourg** - Master OTG program[cite: 5].

---

[cite_start]**© 2026 - Personal Research Project - University of Strasbourg** [cite: 27, 240]
