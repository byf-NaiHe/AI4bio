# Single-Cell RNA-Seq Analysis: Predicting Cell Types from Gene Expression

## Description
This project focuses on predicting cell types from single-cell RNA sequencing (scRNA-seq) data of peripheral blood mononuclear cells (PBMCs). The dataset contains gene expression profiles of various cell types, and the goal is to classify cells into their respective types using machine learning models. We explore two approaches:
1. **Variational Autoencoder (VAE)**: A deep generative model for dimensionality reduction and clustering.
2. **ResNet-based MLP**: A deep learning model with residual connections for cell type classification.

The dataset used in this project is `pbmc_data.csv`, which contains gene expression profiles of PBMCs. Each row represents a cell, and each column represents a gene. The last two columns contain the cell type labels and their string representations.

---

## Project Overview
This project explores the effectiveness of two machine learning models for analyzing single-cell RNA-seq data:
1. **Variational Autoencoder (VAE)**: Used for dimensionality reduction and clustering of gene expression data.
2. **ResNet-based MLP**: Used for cell type classification.

### Models :
- **VAE**: A deep generative model for learning latent representations of gene expression data.
- **ResNet-based MLP**: A deep learning model with residual connections for classifying cell types.

### Techniques :
- **UMAP**: For visualizing the latent space of the VAE.
- **KMeans Clustering**: For evaluating the clustering performance of the VAE.
- **Cross-Validation**: For evaluating the ResNet-based MLP.

---

## Methodology

### 1. Data Preprocessing
- **Standardization**: Gene expression data is standardized using `StandardScaler`.
- **Label Encoding**: Cell type labels are encoded into integers.
- **Train-Test Split**: The dataset is split into training, validation, and test sets.

### 2. Model Construction
- **VAE**:
  - Encoder: Reduces gene expression data to a latent space.
  - Decoder: Reconstructs gene expression data from the latent space.
  - Loss Function: Combines reconstruction error and KL divergence.
- **ResNet-based MLP**:
  - Residual Blocks: Each block contains two fully connected layers with batch normalization and dropout.
  - Output Layer: Maps features to cell type classes using softmax.

### 3. Evaluation Metrics
- **Adjusted Rand Index (ARI)**: For evaluating clustering performance.
- **Accuracy, Precision, Recall, F1 Score**: For evaluating classification performance.
- **Confusion Matrix**: For visualizing classification results.

---

## Key Findings
- **VAE Model**:
  - Batch normalization is crucial for stabilizing the training process.
  - The dimensionality of the latent space significantly affects clustering performance.
  - The KL divergence weight controls the regularization strength and influences the diversity of generated samples.
- **ResNet-based MLP**:
  - Reducing the number of neurons in the hidden layers effectively prevents overfitting.
  - Increasing the dropout rate improves the model's generalization ability.
  - Increasing the L2 regularization weight has a limited effect on preventing overfitting.
  - A higher learning rate can cause the model to oscillate and fail to converge.

---

## Directory Structure
```
single-cell-rna-seq-analysis/
├── data/                   # Dataset files
├── codes/                  # Code for VAE and ResNet-based MLP
├── results/                # Performance metrics and plots
└── README.md               # Project documentation
```

---

## Future Work
- Experiment with other dimensionality reduction techniques like t-SNE or PCA.
- Incorporate additional features such as pathway analysis or gene ontology.
- Perform hyperparameter tuning for better model performance.
- Explore other deep learning architectures like Graph Neural Networks (GNNs) for single-cell data.

---

Happy coding! 🧬🔬