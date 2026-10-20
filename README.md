# PlateVision: AI Number Plate Detection and Recognition
### Signals and Image Processing (SIP) Course Project

A complete, production-grade Automated License Plate Recognition (ALPR) system built using **Python**, **Streamlit**, **OpenCV**, **NumPy**, **EasyOCR**, and **Regular Expressions**.

Deployable **100% free of charge (₹0)** on **Streamlit Community Cloud** with public HTTPS access.

---

## 1. System Architecture & Signals and Image Processing Pipeline

```
[Input Vehicle Image (RGB)]
            │
            ▼
[Grayscale Conversion] ────► Y = 0.299R + 0.587G + 0.114B (ITU-R BT.601)
            │
            ▼
[Bilateral Filter] ────────► I_BF(p) = (1/W_p) Σ I(q) g_s(||p-q||) g_r(||I(p)-I(q)||)
            │                (Edge-preserving Gaussian spatial & photometric noise attenuation)
            ▼
[Canny Edge Detection] ────► Sobel gradients G_x, G_y -> Magnitude √(G_x² + G_y²)
            │                Non-maximum suppression -> Dual threshold hysteresis
            ▼
[Morphological Closing] ───► A • B = (A ⊕ B) ⊖ B with rectangular 9x3 structuring element
            │                (Consolidates discrete alphanumeric characters into continuous ROI)
            ▼
[Contour Localization] ────► Connected components -> Aspect ratio filter (1.8 ≤ W/H ≤ 6.5)
            │                Area bounds check (0.2% - 35% of total frame area)
            ▼
[Plate ROI Extraction] ────► Bounding rectangle crop & bilinear interpolation normalization
            │
            ▼
[Otsu Binarization] ───────► Maximize inter-class variance: σ_B²(t*) = ω_0(t)ω_1(t)[μ_0(t) - μ_1(t)]²
            │                Adaptive threshold separating glyphs from background
            ▼
[EasyOCR Text Recognition]─► Deep learning optical character recognition (English Latin alphanumeric)
            │
            ▼
[RegEx Grammar Matching] ──► Indian HSRP: ^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$
                             UK Standard: ^[A-Z]{2}[0-9]{2}[A-Z]{3}$
                             European: ^[A-Z]{1,3}[A-Z]{1,2}[0-9]{1,4}$
```

---

## 2. Zero-Cost (₹0) Deployment Guide to Streamlit Community Cloud

Follow these 3 simple steps to host your application publicly with a permanent HTTPS link:

### Step 1: Create a GitHub Repository
1. Log in to [GitHub](https://github.com) (free account).
2. Click **New Repository**, name it `platevision-alpr`, set it to **Public**, and click **Create repository**.
3. Upload the project files:
   - `app.py`
   - `requirements.txt`
   - `packages.txt`
   - `.streamlit/config.toml`
   - `README.md`

### Step 2: Sign In to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) (free, ₹0).
2. Click **Continue with GitHub** to connect your account. No credit card is required.

### Step 3: Deploy
1. Click **New app**.
2. Select your repository: `your-username/platevision-alpr`.
3. Set Main file path: `app.py`.
4. Click **Deploy!**
5. Within 2-3 minutes, your app is live at a permanent public HTTPS URL (e.g., `https://platevision-alpr.streamlit.app`) accessible by anyone worldwide!

---

## 3. Local Installation & Testing

```bash
# 1. Clone repository
git clone https://github.com/your-username/platevision-alpr.git
cd platevision-alpr

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Streamlit web app
streamlit run app.py
```

---

## 4. Viva / Oral Examination Cheatsheet (Signals & Image Processing)

| Question | Model Viva Answer |
| :--- | :--- |
| **Why use Bilateral filter instead of Gaussian filter?** | Standard Gaussian blurring smooths across edges, blurring license plate boundaries. The Bilateral filter weights pixels by both spatial distance AND intensity difference, smoothing flat noise surfaces while preserving sharp edge transitions. |
| **How does Canny Edge Detection eliminate false edges?** | It uses Non-Maximum Suppression (NMS) to thin thick gradient bands into 1-pixel-wide ridges, followed by Hysteresis Thresholding with two thresholds ($T_L, T_H$). Pixels $> T_H$ are strong edges, and pixels between $T_L$ and $T_H$ are kept only if connected to a strong edge. |
| **What is the mathematical principle of Otsu's thresholding?** | Otsu's method finds an optimal threshold $t^*$ by maximizing the between-class variance $\sigma_B^2(t)$, which is mathematically equivalent to minimizing intra-class variance. It operates completely adaptively without manual tuning. |
| **Why apply Morphological Closing with a rectangular kernel?** | Alphanumeric characters on a license plate are horizontally aligned with small gaps. A rectangular structuring element ($9 \times 3$) dilates horizontally to merge individual characters and the border into a single connected rectangular contour. |
