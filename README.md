# Lung Cancer CT-Scan Classification using Transfer Learning CNNs and Vision Transformer

Comparative deep learning study for classifying lung CT-scan images into **Benign**, **Malignant** and **Normal** using the IQ-OTH/NCCD Lung Cancer dataset. Six ImageNet-pretrained CNNs are benchmarked against a Vision Transformer (ViT) trained from scratch, all under identical preprocessing, augmentation, optimizer, loss and epoch settings.

**Course:** Machine Learning Theory, School of Systems and Technology, University of Management and Technology (UMT)
**Authors:** Shahla Munawar (S2024332007), Tooba Mehmood (S2024332025)

---

## Results

| Model | Test Loss | Test Accuracy |
|---|---|---|
| **VGG16** | **0.3273** | **86.67%** |
| EfficientNetB0 | 0.3991 | 84.00% |
| ResNet152 | 0.4529 | 82.50% |
| ResNet50 | 0.4752 | 81.00% |
| InceptionV3 | 0.4843 | 77.00% |
| MobileNetV2 | 0.5509 | 72.67% |
| Vision Transformer (from scratch) | - | 45.00% |

**Key findings**
- VGG16 gave the best accuracy and lowest loss, even though it is the oldest and simplest architecture tested.
- Malignant cases were classified with the highest precision (about 0.95 to 1.00) across all CNNs.
- Most errors came from Benign vs Normal confusion. MobileNetV2 was the most biased, with Benign recall of only 0.45.
- The ViT trained without pretraining reached only 45%, showing that transformers need large-scale pretraining on small medical datasets.

### VGG16 per-class report (best model)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Benign | 0.83 | 0.85 | 0.84 |
| Malignant | 0.98 | 0.88 | 0.93 |
| Normal | 0.80 | 0.86 | 0.83 |
| Weighted avg | 0.87 | 0.87 | 0.87 |

---

## Methodology

**Dataset:** IQ-OTH/NCCD Lung Cancer dataset, pre-split and balanced across the three classes.

| Split | Images |
|---|---|
| Train | 3000 (1000 per class) |
| Validation | 600 (200 per class) |
| Test | 600 (200 per class) |

**Preprocessing**
- Images resized to 128 x 128, batch size 32
- Model-specific `preprocess_input` for each backbone
- Augmentation (training only): random horizontal flip, rotation (10%), zoom (10%)
- `tf.data` pipeline with `.cache()` and `.prefetch(AUTOTUNE)`

**Shared classification head (all CNNs)**

```
Backbone (frozen) -> GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(3, Softmax)
```

**Training setup**
- Loss: Sparse Categorical Cross-Entropy (dataset is balanced, so Focal Loss was not needed)
- Optimizer: Adam (default learning rate for CNNs, 1e-4 for ViT)
- Epochs: 30
- Hardware: Google Colab T4 GPU, mixed precision and XLA

**Vision Transformer (from scratch):** patch size 16x16 (64 patches), embedding dim 64, 4 heads, 4 transformer layers, MLP head [128, 64].

---

## Repository Structure

```
.
├── Lung_Cancer_Classification.ipynb   # Full code with outputs
├── report.pdf                         # Detailed project report
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

The dataset is **not included** in this repository because of its size. Download the IQ-OTH/NCCD Lung Cancer dataset from Kaggle (search for "IQ-OTHNCCD lung cancer dataset") and arrange it as:

```
dataset/
├── train/
│   ├── Bengin cases/
│   ├── Malignant cases/
│   └── Normal cases/
├── val/
└── test/
```

Update `TRAIN_DIR`, `VAL_DIR` and `TEST_DIR` in the notebook to match your paths.

> Note: the original dataset folder is spelled "Bengin cases" and the notebook keeps this spelling.

---

## How to Run

**Option 1: Google Colab (recommended)**
1. Open the notebook in Colab and set the runtime to **GPU (T4)**.
2. Upload the dataset as `archive.zip` when prompted.
3. Run all cells.

**Option 2: Local**
```bash
git clone https://github.com/<your-username>/lung-cancer-ct-classification.git
cd lung-cancer-ct-classification
pip install -r requirements.txt
jupyter notebook
```
The notebook uses `google.colab.files` for uploading, so for local use replace that cell with direct dataset paths.

---

## Limitations and Future Work

Results are lower than some published figures (90 to 99%) because this study used a small input size (128x128), frozen backbones, and a fixed 30-epoch budget with no hyperparameter search. Planned improvements:
- Fine-tune the top layers of the backbones
- Increase input resolution
- Pretrain the ViT on a larger corpus, or use a pretrained ViT checkpoint
- Explore Focal Loss on an imbalanced version of the dataset

---

## References

1. Al-Yasriy et al. (2020). Diagnosis of Lung Cancer Based on CT Scans Using CNN. IQ-OTH/NCCD Lung Cancer Dataset.
2. He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
3. Simonyan and Zisserman (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR.
4. Tan and Le (2019). EfficientNet. ICML.
5. Szegedy et al. (2016). Rethinking the Inception Architecture for Computer Vision. CVPR.
6. Sandler et al. (2018). MobileNetV2. CVPR.
7. Dosovitskiy et al. (2021). An Image is Worth 16x16 Words. ICLR.
8. Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV.

---

## Disclaimer

This project is for academic and research purposes only. It is **not** a medical device and must not be used for clinical diagnosis.
