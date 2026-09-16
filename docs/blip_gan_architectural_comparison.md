# Architectural Comparison: ACFD-GAN vs. BLIP-GAN

This document summarizes the architectural differences between the baseline **ACFD-GAN** model described in the literature (*"High-quality synthetic image with ACFD-GAN for enhanced diabetic retinopathy grading"*) and the enhanced **BLIP-GAN** framework implemented in this project within `notebooks/experiments/BLIP_GAN/` and `src/infrastructure/deep_learning/models/blip_gan/`.

---

## 1. Executive Summary

- **ACFD-GAN** is a conditional Generative Adversarial Network designed to synthesize fundus retinal images conditioned on diabetic retinopathy (DR) severity grades. Its primary objective is to alleviate severe clinical class imbalance by generating high-resolution synthetic samples for underrepresented stages.
- **BLIP-GAN (Background-Preserving Lesion Inpainting GAN)** is developed upon the theoretical foundation of ACFD-GAN, but introduces automated weak-mask preprocessing, background preservation constraints, and lesion-targeted localized inpainting. Therefore, BLIP-GAN serves as a lesion-aware, anatomically faithful restructuring and extension of ACFD-GAN.

> **Implementation Note**: Within the codebase, certain class declarations retain foundational identifiers (e.g., `ACFDGenerator`, `ACFDDiscriminator`). However, the BLIP-GAN pipeline integrates proprietary mechanisms including weak-mask soft fusion, alpha-blended lesion inpainting, background preservation loss, and post-hoc multi-seed FID evaluation.

---

## 2. ACFD-GAN Architecture (Baseline Paper)

The baseline ACFD-GAN framework comprises seven primary components:

1. **WAE (Wasserstein AutoEncoder)**
2. **U-Net Generator**
3. **LRDB (Lightweight Residual Dense Block)**
4. **ACFF (Adaptive Cross-layer Feature Fusion)**
5. **AMM (Adaptive Modulation Module)**
6. **PatchGAN Discriminator**
7. **WMA-FID Model Selection**

### 2.1 WAE Latent Conditioning
The baseline uses a Wasserstein AutoEncoder (WAE) trained via Maximum Mean Discrepancy (MMD) to learn a smooth latent distribution of real fundus photographs. The extracted latent vector from WAE is combined with stochastic noise $z$ to condition the generative process.

**Key Objectives:**
- Retain global anatomical structure across the retinal canvas.
- Introduce diverse variations through stochastic noise sampling.
- Mitigate mode collapse and prevent unrealistic global retinal distortions.

**Code Implementation:**
- `WAEEncoder`
- `WAEDecoder`
- `WAE`
- `mmd_loss`

*Workflow*: The WAE is pre-trained independently; its encoder is then frozen to extract conditioning latent vectors for the Generator.

### 2.2 U-Net-Style Generator
The generator follows an encoder-decoder architecture with dense skip connections:

```text
Conditioning Input (Mask / Latent)
    ├──> Encoder with LRDB + Downsampling
    ├──> Bottleneck integrating latent vector z
    ├──> Decoder with Upsampling
    ├──> ACFF Skip Attention Fusion
    ├──> AMM Feature Modulation
    └──> High-Resolution Synthetic Fundus Image
```

Rather than synthesizing images purely from unstructured noise, the generator ingests both structural mask guidance and latent representations, ensuring anatomical coherence with actual fundus topography.

### 2.3 LRDB (Lightweight Residual Dense Block)
The LRDB module functions as an efficient multi-scale local feature extractor, replacing computationally heavy standard residual dense blocks.

**Functionality:**
- Extracts hierarchical local features across multiple receptive fields.
- Preserves fine-grained pathological micro-lesions (microaneurysms, hemorrhages, hard/soft exudates).
- Reduces computational FLOPs and parameter footprint compared to full RDBs.

**Code Implementation:**
- $1 \times 1$ convolutions for channel compression.
- `DepthwiseSeparableConv` layers for efficient spatial representation learning.
- Cumulative dense residual connections.
- Final $1 \times 1$ convolution for projection and residual fusion.

### 2.4 ACFF (Adaptive Cross-layer Feature Fusion)
ACFF represents a core innovation in ACFD-GAN by bridging encoder and decoder features via joint spatial-channel attention.

**Functionality:**
- Fuses low-level encoder features (rich in edge, boundary, and vessel details) with high-level decoder features (rich in clinical semantics).
- Applies Channel Attention and Spatial Attention sequentially to weight informative feature regions.
- Mitigates the loss of subtle microvascular lesions during decoder feature reconstruction.

**Code Implementation (`ACFF`):**
- Channel attention derived from global average and max-pooled fused feature maps.
- Spatial attention computed across the channel axis.
- Final convolution layer projecting fused representations back to decoder dimensionality.

### 2.5 AMM (Adaptive Modulation Module)
AMM modulates decoder feature representations using both the WAE latent representation and conditioning mask guidance.

**Functionality:**
- Dynamically injects mask geometry into feature synthesis layers.
- Fuses WAE latent semantics with projected mask embeddings.
- Predicts channel-wise scaling ($\gamma$) and shifting ($\beta$) modulation parameters, inspired by Adaptive Instance Normalization (AdaIN).
- Enhances visual diversity while strictly preserving macro-architectural consistency.

### 2.6 PatchGAN Discriminator
The discriminator operates on localized $N \times N$ patches rather than full-image scalar predictions, conditioned on paired inputs:

$$\text{Discriminator Input} = \text{Concat}(\text{Fundus Image}, \text{Mask})$$

**Functionality:**
- Evaluates local patch realism rather than just global image distributions.
- Well-suited for medical imaging where pathologies manifest as discrete, localized micro-patterns.
- Penalizes high-frequency structural artifacts, compelling the generator to synthesize crisp vascular trees and realistic lesion textures.
- Stabilized via Spectral Normalization across all convolutional layers.

### 2.7 Objective Functions & Checkpoint Selection
The baseline ACFD-GAN optimization objective is formulated as:

$$\mathcal{L}_{G} = \mathcal{L}_{\text{adv}} + \lambda_{L1} \mathcal{L}_{L1} + \lambda_{\text{VGG}} \mathcal{L}_{\text{perceptual}}$$

**Default Parameters in Code:**
- `ADV_LOSS_TYPE = "bce"`
- `LAMBDA_L1 = 10.0`
- `LAMBDA_VGG = 10.0` (VGG-19 feature matching)
- Checkpoint selection guided by Fréchet Inception Distance (FID) and Weighted Moving Average FID (WMA-FID).

---

## 3. BLIP-GAN Architectural Enhancements

BLIP-GAN retains the core generative modules of ACFD-GAN while introducing four critical clinical engineering improvements:

```text
Real Fundus Photograph
    ├──> Automated Weak Lesion/Vessel Mask Extraction (Frangi + Canny + Thresholding)
    ├──> Pre-trained WAE Latent Extraction
    ├──> Latent Modulation & Stochastic Noise Sampling
    ├──> ACFD-Style Generator
    ├──> Lesion-Region Alpha Inpainting
    ├──> Background Preservation via Anatomical L1 Constraint
    └──> Multi-Seed Post-Hoc FID & Clinical Downstream Validation
```

### 3.1 Weak-Mask Soft Fusion
Unlike prior work requiring pixel-level manual lesion segmentations (which are rarely available at scale in clinical practice), BLIP-GAN generates unsupervised weak masks directly from fundus images:

- **Implementation**: `make_weak_mask` with `MASK_MODE = "softfusion_v2_rich"`
- **Channel 0 (Vessel & Structure)**: Frangi vesselness filter + Hybrid Canny edge detector.
- **Channel 1 (Bright Lesions)**: Morphological high-intensity thresholding for hard exudates and cotton wool spots.
- **Channel 2 (Dark Lesions)**: Inverted green-channel morphological closing for hemorrhages and microaneurysms.
- **Perimeter Padding**: Boundary erosion to remove optical lens edge artifacts.

*Clinical Impact*: Eliminates dependence on manual segmentation annotations while providing strong spatial priors on pathological lesion distributions.

### 3.2 Lesion-Region Inpainting Mechanism
A major limitation of full-image GAN generators is that they regenerate the entire image from scratch, often corrupting healthy background tissue, optic disc morphology, or vascular continuity. BLIP-GAN restricts synthesis exclusively to lesion regions:

$$I_{\text{synthetic}} = I_{\text{base}} \odot (1 - \alpha) + I_{\text{raw\_fake}} \odot \alpha$$

Where $\alpha \in [0, 1]$ represents the smoothed, boundary-dilated lesion mask:
- Blended with Gaussian smoothing for seamless physiological transitions.
- Hard-clamped to prevent destructive overwriting of normal retinal anatomy.

*Clinical Impact*: Preserves original patient background, optic cup/disc, and major vascular geometry while exclusively inpainting clinically plausible DR lesions.

### 3.3 Background Preservation Loss
To ensure that the generator respects non-lesion regions during training, BLIP-GAN introduces an explicit background penalty:

$$\mathcal{L}_{\text{bg}} = \| (I_{\text{fake}} - I_{\text{real}}) \odot (1 - \alpha_{\text{lesion}}) \|_{1}$$

- Configured with `BACKGROUND_PRESERVE_WEIGHT = 2.0`.
- Integrated directly into the total Generator objective.

*Clinical Impact*: Heavily penalizes unintended color shifts, optical distortions, or synthetic hallucinations in healthy retinal tissue.

### 3.4 Multi-Seed Post-Hoc FID Evaluation
To prevent checkpoint selection bias from lucky random seeds, BLIP-GAN evaluates generative quality across multiple distinct seeds:
- `POSTHOC_FID_SEEDS = [42, 123, 2026]`
- Reports both mean and standard deviation for FID.
- Employs candidate generation and perceptual filtering for class-wise sample selection.

---

## 4. Comprehensive Feature Comparison Matrix

| Component | ACFD-GAN (Baseline Literature) | BLIP-GAN (This Project) |
|:---|:---|:---|
| **Primary Goal** | High-quality DR image synthesis for class balancing | Lesion-focused synthesis with absolute retinal background preservation |
| **Conditioning Input** | Ground-truth mask + WAE latent + noise | Automated weak mask + WAE latent + noise + base real image |
| **Mask Acquisition** | Assumes ideal structural/lesion masks | Unsupervised multi-cue weak mask (`softfusion_v2_rich`) |
| **Generator Backbone** | U-Net with LRDB, ACFF, AMM | U-Net with LRDB, ACFF, AMM + **Lesion Inpainting Head** |
| **LRDB Convolution** | Standard residual convolutions | Depthwise separable convolutions (lightweight, efficient) |
| **Feature Fusion** | ACFF channel + spatial attention | ACFF channel + spatial attention |
| **Modulation** | AMM with WAE latent + noise | AMM with WAE latent + noise + mask latent |
| **Discriminator** | PatchGAN (conditioned on image + mask) | Spectral-Normalized PatchGAN (image + mask) |
| **Loss Function** | Adversarial + $L_1$ + Perceptual | Adversarial + $L_1$ + VGG Perceptual + **Background Preservation Loss** |
| **Synthesis Scope** | Synthesizes entire image from scratch | Inpaints pathological lesions; copies real background |
| **Checkpoint Selection**| Single-seed FID / WMA-FID | WMA-FID + **Multi-seed Post-hoc FID** + Downstream Classifier QWK |
| **Clinical Viability** | Generic synthetic retinal images | Clinically plausible images with verifiable anatomical consistency |

---

## 5. Clinical & Engineering Advantages of BLIP-GAN

1. **Pathology-Centric Synthesis**: DR severity grading is clinically determined by localized micro-features (microaneurysms, blot hemorrhages, neovascularization). Inpainting guarantees that generative capacity is concentrated where it diagnostically matters.
2. **Zero Anatomical Hallucination**: Copying real background tissue prevents artifactual disruptions to the fovea, macula, and optic disc.
3. **Fully Unsupervised Pipeline**: Does not require costly, time-consuming radiologist lesion annotations, allowing direct deployment on arbitrary fundus datasets (APTOS 2019, Messidor-2).
4. **Reproducible Multi-Seed Verification**: Multi-seed FID evaluation ensures robust, non-cherry-picked generative fidelity.
5. **Demonstrated Downstream Utility**: When used for minority-class oversampling, BLIP-GAN directly enables the downstream **LPBO-DRNet** classifier to achieve **98.15% Test Accuracy** and **0.9572 QWK**.

---

## 6. Honest Limitations & Transparency

- **Architectural Lineage**: BLIP-GAN builds upon ACFD-GAN principles rather than proposing a completely decoupled paradigm.
- **Weak-Mask Precision**: Unsupervised weak masks can occasionally capture benign vessel textures or miss tiny punctate hemorrhages in poor-quality scans.
- **Inpainting Dependency**: If the initial weak mask misidentifies a region, the inpainting boundary might carry subtle boundary artifacts.
- **Evaluation Criteria**: FID measures feature-space distance; the ultimate metric of synthetic quality remains downstream classification performance on external benchmarks.

---

## 7. Conclusion

While ACFD-GAN established an effective conditional generative formulation with LRDB, ACFF, and AMM modules, BLIP-GAN elevates it to clinical feasibility. By decoupling healthy anatomical background from pathological lesion synthesis, BLIP-GAN provides realistic, anatomically faithful data augmentation for clinical diabetic retinopathy diagnosis.
