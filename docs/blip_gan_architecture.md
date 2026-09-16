# BLIP-GAN: Architectural Specification

This document presents the detailed architectural specification of **BLIP-GAN** (*Background-Preserving Lesion Inpainting Generative Adversarial Network*), an advanced deep generative framework engineered for high-fidelity fundus image synthesis and clinical lesion inpainting in Diabetic Retinopathy (DR) diagnosis.

---

## 1. System Overview

**BLIP-GAN** addresses the challenge of clinical class imbalance in retinal fundus datasets by synthesizing realistic pathological lesions (such as microaneurysms, hemorrhages, and exudates) while strictly preserving patient-specific anatomical landmarks (such as the optic disc, macula, and vascular tree).

Unlike standard generative models that regenerate the entire image canvas, BLIP-GAN operates as a **lesion-guided inpainting network**. It dynamically isolates pathological regions and synthesizes targeted lesions with high anatomical fidelity, ensuring that synthetic images serve as effective training data for downstream clinical classifiers.

```text
Input Real Fundus Image
    ├──> Automated Weak Mask Extraction (Frangi + Canny + Color Thresholding)
    ├──> Latent Feature Encoding via Pretrained WAE
    ├──> U-Net Inpainting Generator (LRDB + ACFF + AMM)
    ├──> Alpha-Blended Lesion Inpainting Head
    └──> Anatomically Faithful Synthetic Fundus Image
```

---

## 2. Core Architectural Components

The BLIP-GAN framework integrates six specialized deep learning modules:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BLIP-GAN ARCHITECTURE                             │
├───────────────────────────────┬─────────────────────────────────────────────┤
│ 1. WAE Latent Conditioning    │ Wasserstein AutoEncoder with MMD Loss       │
│ 2. LRDB Blocks                │ Lightweight Residual Dense Convolutions     │
│ 3. ACFF Attention Fusion      │ Adaptive Cross-layer Feature Fusion         │
│ 4. AMM Feature Modulation     │ Adaptive Modulation with Mask & Latent      │
│ 5. Alpha Inpainting Head      │ Region-Targeted Pathological Inpainting     │
│ 6. PatchGAN Discriminator     │ Spectral-Normalized Multi-Scale PatchGAN    │
└───────────────────────────────┴─────────────────────────────────────────────┘
```

### 2.1 WAE Latent Conditioning
To capture the global structural distribution of fundus photography, BLIP-GAN employs a **Wasserstein AutoEncoder (WAE)** trained with Maximum Mean Discrepancy (MMD) loss.

- **Encoder (`WAEEncoder`)**: Projects real fundus images into a continuous, smooth 128-dimensional latent space $\mathcal{Z}$.
- **MMD Regularization**: Enforces the latent distribution $q(z)$ to align with a Gaussian prior $p(z) \sim \mathcal{N}(0, I)$ using a multi-scale Inverse Multiquadric (IMQ) kernel:
  $$k(z, z') = \sum_{j} \frac{C_j}{C_j + \|z - z'\|^2}$$
- **Latent Injection**: Combines the learned latent vector with stochastic noise to enable diverse yet structurally coherent image synthesis.

### 2.2 U-Net-Style Generator
The generator follows an encoder-decoder topology with rich skip connections:
1. **Downsampling Path**: Progressively extracts hierarchical feature maps using LRDB blocks and strided convolutions.
2. **Bottleneck Layer**: Fuses spatial features with the WAE latent vector.
3. **Upsampling Path**: Restores spatial resolution via transposed convolutions, guided by ACFF skip connections and AMM modulation layers.

### 2.3 Lightweight Residual Dense Block (LRDB)
The LRDB module functions as a multi-scale local feature extractor tailored for fine microvascular structures:
- **Depthwise Separable Convolutions**: Drastically reduces parameter overhead while maintaining strong representational capacity.
- **Dense Accumulation**: Features from preceding layers are concatenated and fused via $1 \times 1$ convolutions, allowing efficient gradient flow during backpropagation.
- **Residual Bypass**: Preserves high-frequency details including tiny punctate hemorrhages and microaneurysms.

### 2.4 Adaptive Cross-Layer Feature Fusion (ACFF)
The ACFF module connects corresponding stages of the encoder and decoder through dual-attention mechanisms:
- **Channel Attention**: Applies Global Average Pooling and Global Max Pooling across spatial dimensions, passing representations through shared MLPs to emphasize clinically significant feature channels.
- **Spatial Attention**: Gathers cross-channel maximum and mean statistics to generate spatial attention maps, directing generative focus toward lesion candidate regions.
- **Residual Integration**: Ensures that low-level vascular boundaries from the encoder seamlessly merge with high-level semantic features in the decoder.

### 2.5 Adaptive Modulation Module (AMM)
Inspired by Adaptive Instance Normalization (AdaIN), the AMM dynamically modulates decoder feature maps using both latent embeddings and mask geometry:
- Feature normalization: Computes channel-wise statistics via Instance Normalization:
  $$\hat{F} = \frac{F - \mu(F)}{\sigma(F)}$$
- Affine parameter generation: Predicts adaptive scale ($\gamma$) and shift ($\beta$) vectors from the fused latent-mask embedding:
  $$\text{AMM}(F) = \gamma \cdot \hat{F} + \beta$$
- This mechanism enables explicit control over lesion contrast and intensity distributions across different DR grades.

### 2.6 PatchGAN Discriminator with Spectral Normalization
The discriminator evaluates localized $N \times N$ patches rather than outputting a single global scalar:
- **Conditional Input**: Evaluates the concatenation of image and conditioning mask $\text{Concat}(I, M)$.
- **Patch-Based Evaluation**: Enforces crisp, localized boundary realism, matching the localized pathological manifestation of retinal lesions.
- **Spectral Normalization**: Stabilizes adversarial training by constraining the Lipschitz constant of each convolutional layer, preventing mode collapse.

---

## 3. Composite Objective Functions

BLIP-GAN is trained end-to-end using a four-part loss formulation designed for perceptual realism and anatomical preservation:

$$\mathcal{L}_{G} = \mathcal{L}_{\text{adv}} + \lambda_{L1} \mathcal{L}_{L1} + \lambda_{\text{VGG}} \mathcal{L}_{\text{perceptual}} + \lambda_{\text{bg}} \mathcal{L}_{\text{bg}}$$

### 3.1 Adversarial Loss ($\mathcal{L}_{\text{adv}}$)
Enforces generated retinal textures to be indistinguishable from authentic clinical photographs:
$$\mathcal{L}_{\text{adv}} = \mathbb{E} \left[ \log(1 - D(G(z, M), M)) \right]$$

### 3.2 Pixel Reconstruction Loss ($\mathcal{L}_{L1}$)
Encourages global structural and luminance consistency with $\lambda_{L1} = 10.0$:
$$\mathcal{L}_{L1} = \mathbb{E} \left[ \| I_{\text{real}} - I_{\text{fake}} \|_{1} \right]$$

### 3.3 Deep Perceptual Loss ($\mathcal{L}_{\text{perceptual}}$)
Extracts multi-scale representations from pretrained VGG-19 feature stages ($\text{relu1\_2}, \text{relu2\_2}, \text{relu3\_2}, \text{relu4\_2}, \text{relu5\_2}$) to preserve complex retinal textures and structural patterns ($\lambda_{\text{VGG}} = 10.0$):
$$\mathcal{L}_{\text{perceptual}} = \sum_{i=1}^{5} \frac{1}{N_i} \| \phi_i(I_{\text{real}}) - \phi_i(I_{\text{fake}}) \|_{1}$$

### 3.4 Background Preservation Loss ($\mathcal{L}_{\text{bg}}$)
An explicit anatomical constraint ensuring that non-lesion regions remain identical to authentic retinal tissue ($\lambda_{\text{bg}} = 2.0$):
$$\mathcal{L}_{\text{bg}} = \mathbb{E} \left[ \| (I_{\text{fake}} - I_{\text{real}}) \odot (1 - \alpha_{\text{lesion}}) \|_{1} \right]$$

---

## 4. Lesion Inpainting Synthesis Mechanism

During final image generation, BLIP-GAN applies an alpha-blending formulation to ensure seamless integration between synthetic lesions and authentic retinal backgrounds:

$$I_{\text{synthetic}} = I_{\text{real}} \odot (1 - \alpha) + I_{\text{raw\_fake}} \odot \alpha$$

Where $\alpha \in [0, 1]$ is a smooth spatial weight map derived from the pathological lesion channels. This ensures:
1. Healthy background tissue, optic disc margins, and primary vascular arcades are strictly preserved from the original examination.
2. Pathological features (exudates, hemorrhages, microaneurysms) are synthesized with natural biological gradients.
3. Zero artificial boundary artifacts occur at tissue transitions.

---

## 5. Checkpoint Selection & Evaluation Protocol

To ensure consistent generative quality, BLIP-GAN incorporates:
- **Weighted Moving Average FID (WMA-FID)**: Smooths epoch-by-epoch metric fluctuations to select optimal model weights.
- **Multi-Seed Post-Hoc FID**: Evaluates generative fidelity across multiple random seeds (`[42, 123, 2026]`) to guarantee robust, reproducible distribution alignment.
- **Perceptual Candidate Selection**: Evaluates multiple generated variations per mask and selects the most diagnostically faithful sample for minority class augmentation.
