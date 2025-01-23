# Deep Learning Projects for Biomedical Data Analysis

This repository contains four projects focused on applying deep learning techniques to biomedical data analysis. Each project addresses a specific problem in the field, leveraging state-of-the-art models and methodologies.

---

## **Project 1: PBMC Cell Type Prediction**
**Description**:  
Predicts cell types in Peripheral Blood Mononuclear Cells (PBMC) based on gene expression data. The project explores dimensionality reduction techniques (PCA and UMAP) and evaluates machine learning models (SVM, Random Forest, and XGBoost) for classification.

**Key Features**:
- Data preprocessing and standardization.
- Dimensionality reduction using PCA and UMAP.
- Model evaluation with SVM, Random Forest, and XGBoost.
- Identification of key genes for cell type classification.

---

## **Project 2: Single-Cell RNA-Seq Analysis**
**Description**:  
Predicts cell types from single-cell RNA sequencing (scRNA-seq) data of PBMCs. Two approaches are explored:
1. **Variational Autoencoder (VAE)**: A deep generative model for dimensionality reduction and clustering.
2. **ResNet-based MLP**: A deep learning model with residual connections for cell type classification.

**Key Features**:
- Deep generative modeling for dimensionality reduction.
- ResNet-based architecture for improved classification.
- Analysis of gene expression profiles for cell type prediction.

---

## **Project 3: Breast Cancer Ultrasound Image Classification and Segmentation**
**Description**:  
Classifies and segments breast cancer ultrasound images using Vision Transformer (ViT) and Mask2Former models. The dataset includes images categorized into three classes: normal, benign, and malignant.

**Key Features**:
- Vision Transformer (ViT) for image classification.
- Mask2Former for tumor region segmentation.
- Data preprocessing and augmentation techniques.

---

## **Project 4: DNABERT-CNN for 5' UTR Translation Efficiency Prediction**
**Description**:  
Predicts the translation efficiency of 5' UTR sequences using a hybrid model combining DNABERT (a pre-trained transformer for DNA sequences) and a Convolutional Neural Network (CNN). The dataset includes 280,000 randomly generated 5' UTR sequences.

**Key Features**:
- DNABERT for sequence embedding.
- CNN for feature extraction and regression.
- Prediction of ribosome load (MRL) using polysome profiling and RNA sequencing.

---

## **Directory Structure**
```
Project_1/             # PBMC Cell Type Prediction
Project_2/             # Single-Cell RNA-Seq Analysis
Project_3/             # Breast Cancer Ultrasound Image Classification and Segmentation
Project_4/             # DNABERT-CNN for 5' UTR Translation Efficiency Prediction
README.md              # Overall project documentation
```

---

## **Conclusion**
These projects demonstrate the application of deep learning techniques to various biomedical challenges, including gene expression analysis, cell type classification, medical image segmentation, and sequence-based prediction. Each project highlights the importance of data preprocessing, model selection, and evaluation in achieving robust and interpretable results.

For detailed documentation and code, refer to the respective project folders. Contributions and suggestions are welcome!