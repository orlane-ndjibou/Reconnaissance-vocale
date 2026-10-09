# 🎙️ Teaching a Computer to Understand Voice Commands

**Speech recognition project – "yes", "no", "stop", "go"... recognized by AI**

> 🌍 Project carried out during our **one-semester exchange in Poland**, at **AGH University of Krakow** (Faculty of Computer Science), for the **Machine Learning** course – academic year 2025–2026.
> Supervisor: Dr. Małgorzata Zajęcka

---

## 📑 Table of Contents

1. [What is this project about?](#-what-is-this-project-about)
2. [How does it work? (no technical background needed)](#-how-does-it-work)
3. [Results](#-results)
4. [Run the project yourself](#-run-the-project-yourself)
5. [Repository content](#-repository-content)
6. [Technical details (for the curious)](#-technical-details)
7. [Glossary](#-glossary)
8. [Future work](#-future-work)
9. [Authors, references, license](#-authors)

---

## 🎯 What is this project about?

Think of a phone that you can control **only with your voice**: *"stop"*, *"yes"*, *"up"*, *"go"*...
This is useful when your hands are busy (driving, cooking) or for people who cannot easily use a touchscreen.

For this to work, the phone must **recognize which word was spoken** in a very short audio recording (1 second).

**Our goal:** build a program that listens to a 1-second clip and decides which of these **12 categories** it belongs to:

| 10 command words | 2 special categories |
|---|---|
| yes · no · up · down · left · right · on · off · stop · go | **silence** (nobody is speaking) · **unknown** (any other word, e.g. "cat", "tree", "bed") |

**Data used:** about 28,000 short recordings from thousands of different speakers, from the Kaggle competition [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/overview).

We compared **two different approaches** to see which one works best:

| | Approach A – Deep Learning | Approach B – Classical Machine Learning |
|---|---|---|
| **Idea** | A neural network that "looks at" the sound as an image | Classic algorithms working on a short list of numbers describing the sound |
| **Models** | 3 CNNs (neural networks) | SVM, Random Forest, XGBoost, KNN |

---

## 🧩 How does it work?

The pipeline has **3 simple steps**:

```
 🔊 Audio clip          🖼️ Sound "picture"           🧠 AI model           ✅ Answer
 (1 second)      →      (spectrogram or MFCC)    →   (CNN / SVM / ...)  →   "stop"
```

**Step 1 – Record.** Each clip is a 1-second audio file.

**Step 2 – Turn sound into something a computer can analyze.**
A computer cannot "listen" like us, so we convert the sound into numbers:
- **Mel Spectrogram** → an *image* showing which pitches (low/high) are present at each moment. Used by Approach A.
- **MFCC** → a *short list of numbers* summarizing the sound. Used by Approach B.

**Step 3 – Let the AI decide.**
The model has been trained on thousands of examples, so it learned which "pictures" or numbers correspond to which word.

---

## 📊 Results

Accuracy = percentage of test recordings that were correctly recognized (the models never saw them during training).

| Rank | Model | Approach | Accuracy |
|:---:|---|---|:---:|
| 🥇 | **Deep CNN + Dropout** | A – Deep Learning | **88.44%** |
| 🥈 | Deep CNN | A – Deep Learning | 87.49% |
| 🥉 | Baseline CNN | A – Deep Learning | 84.03% |
| 4 | XGBoost | B – Classical ML | 66.08% |
| 5 | SVM | B – Classical ML | 22.81% |
| 6 | KNN | B – Classical ML | 11% |

### 💡 What we learned

- ✅ **Deep Learning wins clearly**: the best neural network is about **22 points better** than the best classical model. Treating sound as an image works very well.
- ✅ **"Dropout" helps**: it is a technique that prevents the network from just memorizing the training examples, and it gave the best score.
- ✅ **"Silence" is easy** to detect (almost 100%).
- ⚠️ **"Unknown" is the hardest category**, because it mixes 20 different words.
- ⚠️ **Similar-sounding words get confused**, like *no / go*, *on / off*, *down / no*.

---

## 🚀 Run the project yourself

**Prerequisites:** Python 3.9+ and Jupyter Notebook.

**1. Download the project**
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

**2. Install the libraries**
```bash
pip install -r requirements.txt
```

**3. Download the audio data** (not included because it is too large)
Get it from [Kaggle](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/data), extract it, and put it in a folder named `data/`.

**4. Open a notebook and run the cells in order**
```bash
jupyter notebook
```

---

## 📁 Repository content

> ✏️ Adapt the file names to your actual repository.

| File / Folder | Content |
|---|---|
| `notebooks/cnn_mel_spectrogram.ipynb` | Approach A: neural networks on spectrograms |
| `notebooks/mfcc_svm.ipynb` | Approach B: SVM on MFCC |
| `notebooks/mfcc_xgboost.ipynb` | Approach B: XGBoost on MFCC |
| `report/Report_Machine_learning.pdf` | Full project report (detailed explanations and figures) |
| `data/` | Audio dataset (to download, not versioned) |
| `requirements.txt` | List of Python libraries |

---

## 🔬 Technical details

<details>
<summary><b>📦 Dataset and preprocessing</b></summary>

- WAV files, 16 kHz, 16-bit PCM, mono, 1 second (16,000 samples)
- 28,418 samples in total, ~2,350–2,380 per class (balanced)
- **`silence`**: random 1-second chunks cut from the `_background_noise_` recordings
- **`unknown`**: random sample taken among the 20 non-target words
- Splits: CNN pipeline = 64% train / 16% validation / 20% test; classical pipeline = 80% / 20%

</details>

<details>
<summary><b>🎛️ Feature extraction</b></summary>

**Mel Spectrogram (CNN input)**
STFT → Mel filter bank (128 bands) → log (dB) scale. Hop length 512 samples. Input shape: `(128, 32, 1)`.

**MFCC (classical ML input)**
20 coefficients, summarized by their mean and standard deviation over time, plus Zero Crossing Rate and Spectral Centroid. Extended version: 162 features. Normalized with `StandardScaler`.

| | MFCC | Mel Spectrogram |
|---|---|---|
| Size | 20–42 values (162 extended) | 128 × 32 = 4,096 values |
| Information | Compressed (some loss) | Complete |
| Best with | Classical ML | CNN |

</details>

<details>
<summary><b>🧠 CNN architecture (selected model)</b></summary>

```
Input (128, 32, 1)
 → Conv2D(32, 3×3, ReLU)  → MaxPool2D(2×2)
 → Conv2D(64, 3×3, ReLU)  → MaxPool2D(2×2)
 → Conv2D(128, 3×3, ReLU) → MaxPool2D(2×2)
 → Flatten
 → Dense(128, ReLU) → Dropout(0.5)
 → Dense(12, Softmax)
```

Training: Adam optimizer, categorical cross-entropy loss, batch size 32, max 20 epochs, early stopping (patience 5, best weights restored).

| Model | Parameters | Accuracy | Loss |
|---|---|---|---|
| Deep CNN + Dropout | 322,892 | 0.8844 | 0.3816 |
| Deep CNN | 322,892 | 0.8749 | 0.4027 |
| Baseline CNN | 756,940 | 0.8403 | 0.5337 |

</details>

<details>
<summary><b>📐 Classical ML models</b></summary>

- **SVM**: RBF kernel, C = 10, γ = 0.01 (found with GridSearchCV, 5-fold cross-validation), 42 features
- **XGBoost**: learning rate 0.05, max depth 8, 500 estimators, 162 features
- **Random Forest** and **KNN** also tested

SVM and KNN performed poorly because averaging MFCC over time loses the order of sounds in a word (e.g. "on" vs "no").

</details>

---

## 📖 Glossary

| Term | Simple explanation |
|---|---|
| **Machine Learning** | Programs that learn from examples instead of being given fixed rules |
| **Spectrogram** | An image of a sound: time on one axis, pitch on the other, brightness = loudness |
| **Mel scale** | A pitch scale that matches how humans hear (we distinguish low pitches better than high ones) |
| **MFCC** | A compact list of numbers describing the "shape" of a sound |
| **CNN** | *Convolutional Neural Network*: a neural network specialized in analyzing images |
| **SVM / XGBoost / KNN** | Classical machine learning algorithms |
| **Dropout** | A trick that randomly switches off parts of the network during training to avoid memorization |
| **Overfitting** | When a model learns the training examples by heart but fails on new ones |
| **Accuracy** | Percentage of correct answers |

---

## 🔭 Future work

- Data augmentation (adding noise, changing speed or pitch, SpecAugment)
- More recent architectures such as Transformers (Audio Spectrogram Transformer)
- Better handling of the `unknown` category
- Using richer MFCC features (delta and delta-delta)

---

## 👥 Authors

- Alfra NDJIBOU MADOUNGOU
- Noor FADLANE
- Vignol LEFAKONG TSOMELOU
- Joss DOUNIAMA OKANA
- Hechmi CHEMLI

## 📚 References

1. TensorFlow / Kaggle. *TensorFlow Speech Recognition Challenge*, 2018.
2. Lopez-Olvera et al. *Leveraging MFCC and Mel-Spectrogram Representations for Deep Learning-Based Speech Recognition*. Engineering Proceedings, 2026.
3. Telmem, Laaidi, Satori. *The Impact of MFCC, Spectrogram, and Mel-Spectrogram on Deep Learning Models for Amazigh Speech Recognition System*. Int. Journal of Speech Technology, 2025.
4. Mahmood, Köse. *Speech Recognition Based on Convolutional Neural Networks and MFCC Algorithm*. Advances in Artificial Intelligence Research, 2021.
5. Park et al. *SpecAugment*. Interspeech, 2019.
6. Davis, Mermelstein. *Comparison of Parametric Representations for Monosyllabic Word Recognition*. IEEE TASSP, 1980.
7. Vapnik. *The Nature of Statistical Learning Theory*. Springer, 1995.

## 📄 License

Educational project developed during an exchange semester in Poland (AGH University of Krakow). Add a license here if you want one (e.g. MIT).
