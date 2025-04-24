# Remote Sensing Classification with Imbalance Handling and Explainability

In this repo, we explore the [Skyview Aerial Landscape Images](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset) dataset for remote sensing image classification. Two Jupyter notebooks are included, containing experiments on both the original and long-tailed versions of the dataset using three traditional machine learning methods (SVM, Random Forest, and KNN) and four deep learning models (ResNet50, VGG16, EfficientNet-B0, SE-ResNeXt50_32x4d).  
In order to demonstrate explainable AI techniques, a separate project folder is also provided, which includes visualizations using Grad-CAM and attention rollout.

## Dataset

Skyview contains 12,000 aerial images, equally distributed across 15 land cover categories such as agriculture, forest, beach, residential, and desert. Each image has a resolution of 256×256 pixels. The dataset is constructed from publicly available benchmarks, AID and NWPU-RESISC45, and serves as a balanced and diverse benchmark for classification tasks.

## Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- Albumentations  
- scikit-learn  
- timm  
- OpenCV  
- matplotlib, seaborn

The requirements of explainable_ai could fine in explainable_ai/requirements.txt

## Project Structure

- `ml.ipynb`: Experiments with traditional ML models using handcrafted features (LBP + SIFT + BoVW + PCA).
- `dl.ipynb`: Experiments with pretrained deep learning models and long-tailed class handling.
- `explainability/`: Project folder for visualizing model decisions with Grad-CAM and attention rollout.
- `ml.ipynb`: Experiments with traditional ML models using handcrafted features (LBP + SIFT + BoVW + PCA).
- `dl.ipynb`: Experiments with pretrained deep learning models and long-tailed class handling.
- `explainable_ai/`: Project folder for visualizing model decisions with Grad-CAM and attention rollout.

## Training Configurations

### Machine Learning Models

Class imbalance is addressed using class-weighted training (SVM, RF, XGBoost) and distance-weighted voting (KNN). Evaluation is conducted under two validation regimes: 80/20 train–test split and 5-fold cross-validation.

### Deep Learning Models

- Batch size: 32  
- Epochs: 200  
- Learning rate: 0.0001  
- Optimizer: Adam  
- Image size: 256×256  
- Scheduler: CosineAnnealingLR (T_max=0.8*epochs)

## Full Performance Table

The following table compares all models under different data balancing strategies:
a.80/20 train-test split
b.5-fold cross-validation

| Model                 | Long-Tail Only   | +Oversampling   | +Augmentation   | Original   |
|:----------------------|:-----------------|:----------------|:----------------|:-----------|
| SVM                   | 0.6585ᵃ          | 0.6906ᵃ         | 0.6566ᵃ         | 0.6937ᵃ    |
|                       | 0.6555ᵇ          | 0.6694ᵇ         | 0.6487ᵇ         | 0.6878ᵇ    |
| RF                    | 0.5792ᵃ          | 0.6358ᵃ         | 0.6396ᵃ         | 0.6646ᵃ    |
|                       | 0.5800ᵇ          | 0.6475ᵇ         | 0.6389ᵇ         | 0.6723ᵇ    |
| KNN                   | 0.5830ᵃ          | 0.5245ᵃ         | 0.4623ᵃ         | 0.6100ᵃ    |
|                       | 0.5913ᵇ          | 0.5181ᵇ         | 0.4898ᵇ         | 0.6082ᵇ    |
| ResNet50              | 0.9533           | 0.9417          | 0.9692          | 0.9908     |
| VGG16                 | 0.9171           | 0.9096          | 0.9408          | 0.9788     |
| EfficientNet-B0       | 0.9537           | 0.9487          | 0.9692          | 0.9917     |
| SE-ResNeXt50-32x4d    | 0.9487           | 0.9471          | 0.9675          | 0.9900     |
| PVTv2                 | 0.9358           | 0.9437          | 0.9587          | 0.9846     |
| Explainable ViT model | -                | -               | -               | 0.9683     |

## Results Summary

- Deep learning models achieved up to **99%** accuracy on the balanced dataset.
- Machine learning models peaked at around **65%** accuracy.
- Minority-class augmentation improved deep learning performance by **1.94 percentage points** on average.
- Traditional models struggled to generalize on imbalanced data, while deep models showed stronger robustness.

## Explainable AI

Explainability is performed via:
- **Grad-CAM**: Highlights class-discriminative regions in CNNs.
- **Attention Rollout**: Visualizes global token influence in Transformer-based architectures.
