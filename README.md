# 🎙️ Speech Recognition – Mel Spectrogram & MFCC

Keyword spotting for hands-free phone control, based on the [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/overview) (Kaggle).

**AGH University of Krakow – Faculty of Computer Science**
Machine Learning course, academic year 2025–2026
Supervisor: Dr. Małgorzata Zajęcka

> 🌍 This project was carried out as part of our **one-semester exchange program in Poland** (study abroad semester at AGH University of Krakow).

## 👥 Authors

- Alfra NDJIBOU MADOUNGOU
- Noor FADLANE
- Vignol LEFAKONG TSOMELOU
- Joss DOUNIAMA OKANA
- Hechmi CHEMLI

---

## 📌 Project Overview

The goal is to classify one-second audio clips into **12 classes**:

`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, `silence`, `unknown`

We compare two complementary approaches:

| Approach | Features | Models |
|---|---|---|
| **Deep Learning** | Mel Spectrogram (128 × 32 × 1) | Baseline CNN, Deep CNN, Deep CNN + Dropout |
| **Classical ML** | MFCC (+ mean, std, ZCR, spectral centroid) | SVM, Random Forest, XGBoost, KNN |

## 📊 Results

| Model | Features | Test Accuracy |
|---|---|---|
| 🥇 **Deep CNN + Dropout** | Mel Spectrogram | **88.44%** |
| 🥈 Deep CNN | Mel Spectrogram | 87.49% |
| 🥉 Baseline CNN | Mel Spectrogram | 84.03% |
| XGBoost | MFCC (162 features) | 66.08% |
| SVM (RBF, C=10, γ=0.01) | MFCC (42 features) | 22.81% |
| KNN | MFCC | 11% |

**Main takeaways**

- Mel Spectrograms combined with CNNs clearly outperform MFCC-based classical models (+22 points).
- Dropout (0.5) improves generalization without hurting accuracy.
- `silence` is almost perfectly recognized; `unknown` and acoustically similar pairs (`no`/`go`, `off`/`on`, `down`/`no`) are the main sources of error.

## 🗂️ Dataset

- **Source:** [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/data) on Kaggle
- **Format:** WAV, 16 kHz, 16-bit PCM, mono, 1 second per clip
- **Balanced dataset:** ~2,350–2,380 samples per class (28,418 in total)
- **`silence`:** 1-second random chunks extracted from the `_background_noise_` recordings
- **`unknown`:** random sample drawn from the 20 non-target words
- **Splits:**
  - CNN pipeline: 64% train / 16% validation / 20% test
  - Classical ML pipeline: 80% train / 20% test

> ⚠️ The dataset is **not included** in this repository because of its size. Download it from Kaggle and place it in the `data/` folder (see below).

## 🔧 Feature Extraction

**Mel Spectrogram (CNN)**
- STFT → Mel filter bank (128 bands) → log (dB) compression
- Hop length: 512 samples
- Input shape: `(128, 32, 1)`

**MFCC (classical ML)**
- 20 MFCC coefficients: mean and standard deviation over time
- Zero Crossing Rate and Spectral Centroid
- Extended feature set (162 dimensions) for XGBoost
- `StandardScaler` normalization

## 🧠 CNN Architecture (selected model)

```
Input (128, 32, 1)
 → Conv2D(32, 3×3, ReLU)  → MaxPool2D(2×2)
 → Conv2D(64, 3×3, ReLU)  → MaxPool2D(2×2)
 → Conv2D(128, 3×3, ReLU) → MaxPool2D(2×2)
 → Flatten
 → Dense(128, ReLU) → Dropout(0.5)
 → Dense(12, Softmax)
```

**Training setup:** Adam optimizer, categorical cross-entropy, batch size 32, up to 20 epochs, early stopping (patience = 5, monitoring validation loss, restoring best weights).

## 📁 Repository Structure

> Adjust the file names below to match your repository.

```
.
├── data/                       # Kaggle dataset (not versioned)
├── notebooks/
│   ├── cnn_mel_spectrogram.ipynb    # CNN benchmark on Mel Spectrograms
│   ├── mfcc_svm.ipynb               # SVM / classical ML on MFCC
│   └── mfcc_xgboost.ipynb           # XGBoost on extended MFCC features
├── report/
│   └── Report_Machine_learning.pdf  # Full project report
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Main libraries: `numpy`, `pandas`, `librosa`, `scikit-learn`, `xgboost`, `tensorflow` / `keras`, `matplotlib`, `seaborn`.

### 3. Download the dataset

Download the data from [Kaggle](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/data), extract it, and place it in `data/` (the `train/audio` folder should contain one subfolder per word plus `_background_noise_`).

### 4. Run the notebooks

```bash
jupyter notebook
```

Open the notebook you want to run (CNN or classical ML) and execute the cells in order.

## 🔭 Future Work

- Data augmentation (time stretching, pitch shifting, noise injection, SpecAugment)
- Transformer-based architectures (e.g. Audio Spectrogram Transformer)
- Delta and delta-delta MFCC features for the classical pipeline
- Open-set recognition / confidence thresholding for the `unknown` class

## 📚 References

1. TensorFlow / Kaggle. *TensorFlow Speech Recognition Challenge*, 2018.
2. J. A. Lopez-Olvera et al. *Leveraging MFCC and Mel-Spectrogram Representations for Deep Learning-Based Speech Recognition*. Engineering Proceedings, 123(1):22, 2026.
3. M. Telmem, N. Laaidi, H. Satori. *The Impact of MFCC, Spectrogram, and Mel-Spectrogram on Deep Learning Models for Amazigh Speech Recognition System*. Int. Journal of Speech Technology, 2025.
4. A. Mahmood, U. Köse. *Speech Recognition Based on Convolutional Neural Networks and MFCC Algorithm*. Advances in Artificial Intelligence Research, 1(1):6–12, 2021.
5. D. S. Park et al. *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition*. Interspeech, 2019.
6. S. B. Davis, P. Mermelstein. *Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences*. IEEE TASSP, 28(4):357–366, 1980.
7. V. N. Vapnik. *The Nature of Statistical Learning Theory*. Springer, 1995.

## 📄 License

Educational project developed at AGH University of Krakow during an exchange semester in Poland. Add a license here if you want one (e.g. MIT).
