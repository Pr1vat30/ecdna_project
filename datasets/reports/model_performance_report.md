# eccDNA Model Performance & Configuration Report

## 1. Model Configuration

| Hyperparameter | Value | Description |
|---|---|---|
| **Input Feature Dimension** | 189 | 125 base features + 64 3-mer frequency motifs |
| **Encoder Architecture** | Siamese MLP / 1D-CNN | Dual-encoder structure for representation metric learning |
| **Hidden MLP Layers** | [256, 128] | Layer sizing for the MLP sequence encoder |
| **Embedding Dimension** | 64 | L2-normalized geometric representation dimension |
| **Classifier Hidden Layer** | 128 | Dense layer mapping embeddings to disease logits |
| **Dropout Rate** | 0.3 | Dropout applied across hidden layers |
| **Optimizer** | Adam | lr=1e-3, weight_decay=1e-4 |
| **Loss Balancing** | Dynamic Lambda | Exp decay ($\lambda_{max}=1.0 \to \lambda_{min}=0.1$) |
| **P-K Batch Sampler** | P=7 classes, K=64 samples | Batch Size = 448 sequences |
| **Training Device** | mps / cuda | Metal Performance Shaders (Metal) backend |
| **Total Epochs** | 50 | Number of optimization cycles |

---

## 2. Embedding Space Retrieval Performance (Baseline Models)

| Model Class | Recall@1 | Recall@5 | Mean Average Precision (mAP) |
|---|---|---|---|
| **Siamese MLP Encoder** | 0.3369 | 0.7631 | 0.2960 |
| **Siamese 1D-CNN Encoder** | 0.3260 | 0.7485 | 0.2777 |

---

## 3. Classification Performance (Softmax Head)

### 3.1 Siamese MLP Classifier (Baseline)

| Disease Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Cervical Adenocarcinoma | 0.48 | 0.50 | 0.49 | 1494 |
| Chronic Kidney Disease | 0.20 | 0.17 | 0.18 | 211 |
| Fetal Growth Restriction | 0.46 | 0.55 | 0.50 | 168 |
| Histiocytic Lymphoma | 0.41 | 0.53 | 0.46 | 1496 |
| Hypopharyngeal Squamous Cell Carcinoma | 0.51 | 0.53 | 0.52 | 804 |
| Ovarian Cancer | 0.48 | 0.58 | 0.52 | 1451 |
| Prostate Cancer | 0.34 | 0.14 | 0.19 | 1480 |
|---|---|---|---|---|
| **Accuracy** | | | **0.4430** | 7104 |
| **Macro Average** | 0.41 | 0.43 | 0.41 | 7104 |
| **Weighted Average** | 0.43 | 0.44 | 0.42 | 7104 |

### 3.2 Siamese 1D-CNN Classifier (Baseline)

| Disease Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Cervical Adenocarcinoma | 0.44 | 0.54 | 0.49 | 1494 |
| Chronic Kidney Disease | 0.23 | 0.13 | 0.17 | 211 |
| Fetal Growth Restriction | 0.49 | 0.55 | 0.52 | 168 |
| Histiocytic Lymphoma | 0.40 | 0.43 | 0.42 | 1496 |
| Hypopharyngeal Squamous Cell Carcinoma | 0.47 | 0.47 | 0.47 | 804 |
| Ovarian Cancer | 0.48 | 0.47 | 0.47 | 1451 |
| Prostate Cancer | 0.29 | 0.23 | 0.26 | 1480 |
|---|---|---|---|---|
| **Accuracy** | | | **0.4178** | 7104 |
| **Macro Average** | 0.40 | 0.40 | 0.40 | 7104 |
| **Weighted Average** | 0.41 | 0.42 | 0.41 | 7104 |

---

## 4. Text Table Representation (MLP Baseline)

```
========================================================================
Classe                                  Precision  Recall  F1-score  Support
------------------------------------------------------------------------
Cervical Adenocarcinoma                    0.48     0.50     0.49     1494
Chronic Kidney Disease                     0.20     0.17     0.18      211
Fetal Growth Restriction                   0.46     0.55     0.50      168
Histiocytic Lymphoma                       0.41     0.53     0.46     1496
Hypopharyngeal Squamous Cell Carcinoma     0.51     0.53     0.52      804
Ovarian Cancer                             0.48     0.58     0.52     1451
Prostate Cancer                            0.34     0.14     0.19     1480
------------------------------------------------------------------------
Accuracy                                                     0.44     7104
------------------------------------------------------------------------
Macro avg                                  0.41     0.43     0.41     7104
Weighted avg                               0.43     0.44     0.42     7104
========================================================================
```