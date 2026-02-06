

# Variational Autoencoder on CelebA (Baseline & β‑VAE)

This project implements and analyzes a convolutional Variational Autoencoder (VAE) and β‑VAE on the CelebA dataset. It focuses on both **practical implementation** (Part 1) and **theoretical understanding** (Part 2).


## 1. Project overview

- **Goal:** Learn a generative latent representation of human faces using VAEs and explore how architecture choices, loss balance (KL vs reconstruction), and preprocessing affect the model.  
- **Key components:**
  - Baseline VAE (β=1) trained on a CelebA subset.  
  - Latent space exploration: reconstructions, interpolations, and traversals.  
  - β‑VAE variants (β>1) to study reconstruction vs disentanglement trade‑offs.


## 2. Dataset and preprocessing

- **Dataset:** CelebA with 202,599 face images and 40 binary attributes per image (e.g. Smiling, Blond_Hair).
- **Files used:**
  - `img_align_celeba/` – aligned face images.  
  - `list_attr_celeba.txt` – 40 attributes per image, used as a 40‑D label vector.  
  - `list_eval_partition.txt` – defines train/val/test splits.  

- **Preprocessing pipeline:**
  - Center‑crop to 178×178 (standard CelebA face crop).
  - Resize to **128×128**.  
  - Convert to tensor and normalize to mean 0.5 / std 0.5 per channel, so pixel values lie in \([-1, 1]\)

- **Splits and subset (hardware‑aware):**
  - Train: **20,000** images (subset of official train).  
  - Validation: **2,000** images (subset of official val).  
  - Test: **2,000** images (subset of official test).  
- **Implementation:** Custom `CelebADataset` class returning `(image, attrs)` where `attrs` is the 40‑D attribute tensor, and `DataLoader`s with batch size 64, `num_workers=0` (Mac).

***

## 3. Baseline VAE (Part 1 – Practice)

### 3.1 Architecture

A convolutional VAE maps \(x \in \mathbb{R}^{3\times128\times128}\) to a **128‑D latent vector** and back.

**Encoder**

- Input: 3×128×128.  
- Conv blocks (kernel=4, stride=2, padding=1, ReLU):
  - 3 → 32 → 64×64  
  - 32 → 64 → 32×32  
  - 64 → 128 → 16×16  
  - 128 → 256 → 8×8  
  - 256 → 512 → 4×4  
- Flatten: 512·4·4 = 8192.  
- Latent heads:
  - `fc_mu`: Linear(8192 → 128)  
  - `fc_logvar`: Linear(8192 → 128)  

**Decoder**

- `fc`: Linear(128 → 512·4·4), reshape to 512×4×4.  
- Deconv blocks (ConvTranspose2d + ReLU):
  - 512 → 256 → 8×8  
  - 256 → 128 → 16×16  
  - 128 → 64 → 32×32  
  - 64 → 32 → 64×64  
  - 32 → 3 → 128×128, followed by **Tanh** to produce outputs in \([-1, 1]\).

**Latent dimension: 128**

- Chosen as a balance between reconstruction quality and capacity to encode multiple semantic factors (pose, expression, hairstyle) for 128×128 faces.

### 3.2 Loss and training setup

The VAE loss per sample is:
\[
\mathcal{L} = \mathbb{E}[\|x - \hat{x}\|^2] + \beta\, \text{KL}(q(z|x)\,\|\,\mathcal{N}(0,I)),
\]
with **β=1** for the baseline.

- **Reconstruction term:** MSE over pixels, summed then averaged per sample.  
- **KL term:**  
  \[
  \text{KL} = -\tfrac{1}{2}\sum (1 + \log\sigma^2 - \mu^2 - \sigma^2).
  \]

**Training config (baseline):**

- Optimizer: Adam, lr = 1e‑3.
- Batch size: 64.  
- Epochs: **10**.  
- Device: Apple M2 (`"mps"` backend).

**Training curves:**

- Plotted total loss, reconstruction loss, and KL divergence over epochs.  
- Observations:
  - Total and reconstruction losses decrease and then flatten, showing convergence.  
  - KL stabilizes after a few epochs, indicating the encoder learns a non‑trivial but regularized latent distribution.

***

## 4. Latent space exploration (Part 1 – Practice)

### 4.1 Reconstructions (test set)

- Take a batch from the **test loader**, pass through encoder + decoder, and visualize **original vs reconstructed** images for at least 8 test faces.  
- Results:
  - Reconstructions are somewhat blurry (typical for VAEs) but preserve coarse structure (pose, face shape, hair region), confirming the latent space encodes meaningful information.

### 4.2 Latent interpolations (3 sequences)

- For three different pairs of test images:
  - Encode them to \(z_1\) and \(z_2\).  
  - Compute interpolated codes \(z_\alpha = (1-\alpha)z_1 + \alpha z_2\) for \(\alpha \in [0,1]\).  
  - Decode each \(z_\alpha\) and display as rows of faces.
- Observations:
  - Smooth morphing of identity, pose, and hairstyle indicates the latent space is continuous and captures meaningful facial factors, not just memorized images.

### 4.3 Latent traversals (single‑dimension variation)

- Fix a reference test image and encode it to \(z\).  
- For selected dimensions (e.g. 8, 25, 68):
  - Vary one coordinate \(z_k\) over a range (e.g. \([-3, 3]\)), keep others fixed, decode, and visualize images as a row.
- Observations:
  - Some dimensions correspond to gradual changes (e.g. slight pose, lighting, or expression variations), supporting the idea of partially disentangled latent factors.

***

## 5. β‑VAE variants (Part 1 – Practice)

### 5.1 β‑VAE loss and training

β‑VAE introduces a tunable β in the KL term:
\[
\mathcal{L}_\beta = \mathbb{E}[\|x - \hat{x}\|^2] + \beta \,\text{KL}(q(z|x)\,\|\,\mathcal{N}(0,I)).
\]

- Two β values used:
  - **β₁ = 4.0** (moderate bottleneck).  
  - **β₂ = 10.0** (strong bottleneck).  
- Each trained from scratch for **5 epochs** with the same architecture, optimizer, and data subset as the baseline.

Training curves (loss, recon, KL) are plotted for each β to show how increased β shifts weight from reconstruction to KL.

### 5.2 Qualitative comparison: reconstructions

- Reconstructions for **β=4** and **β=10** on the test set:
  - β=4: reconstructions become slightly blurrier than β=1 but still recognizable; some factors become more consistently encoded.  
  - β=10: reconstructions are noticeably smoother and more “average,” indicating a strong information bottleneck.

### 5.3 Qualitative comparison: interpolations and traversals

- **Interpolations**:
  - For each β, three interpolation rows are generated as in the baseline.  
  - Higher β often yields smoother, more regular transitions in the latent space (fewer abrupt artifacts), reflecting more structured latents.

- **Traversals across β values**:
  - For selected dimensions (e.g. dim 8), traversals are shown in **rows per β**:
    - Row 1: β=1, row 2: β=4, row 3: β=10.  
  - As β increases, many dimensions show more “single‑factor” effects (e.g. consistent change in expression or pose), which is a hallmark of improved disentanglement at the cost of fine reconstruction detail.

***

## 6. Part 2 – Theory summary

The practical experiments tie into the theory questions:

1. **Latent dimension choice:**  
   - 128‑D latent balances reconstruction and generative diversity; too small → underfitting and blur, too large → unused dimensions and weak KL.

2. **KL vs reconstruction trade‑off:**  
   - β controls emphasis on latent regularization vs pixel fidelity.  
   - High β (β=4,10) increases disentanglement but degrades reconstruction, visible in your β‑VAE results.

3. **Data preprocessing impact:**  
   - Center‑cropping and alignment focus the model on facial content rather than background/position.  
   - Normalization and fixed resizing improve training stability and match decoder output range.

4. **Training stability (posterior collapse):**  
   - Monitored via KL curves and traversals; none of your runs showed KL→0 with flat traversals, suggesting no severe collapse.

5. **Latent space manipulation:**  
   - Current project uses manual traversals and interpolations.  
   - With CelebA attributes, one could extend this by training linear probes or computing mean latent differences to find directions for attributes like “Smiling.”

***

## 7. How to run

1. Download and unpack CelebA into `data/celeba/` with:
   - `img_align_celeba/`  
   - `list_attr_celeba.txt`  
   - `list_eval_partition.txt`[6]

2. Install dependencies:
   - Python, PyTorch, torchvision, pandas, matplotlib.

3. Open the notebook or run the script:
   - Configure `DATA_ROOT`, `IMG_SIZE=128`, `LATENT_DIM=128`.  
   - Run cells in order:
     1. Data & preprocessing.  
     2. Baseline VAE training & plots.  
     3. Latent space exploration (reconstructions, interpolations, traversals).  
     4. β‑VAE training (two β values) & visual comparison.



