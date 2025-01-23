# DNABERT-CNN Model for 5' UTR Translation Efficiency Prediction

## Overview
This project implements a deep learning model to predict the translation efficiency of 5' UTR sequences using a combination of DNABERT (a pre-trained transformer model for DNA sequences) and a Convolutional Neural Network (CNN). The model is trained on a dataset of 280,000 randomly generated 5' UTR sequences, each 50 nucleotides long, followed by an eGFP coding sequence and a 3' UTR. The translation efficiency is measured as the ribosome load (MRL) using polysome profiling and RNA sequencing.

## Project Structure
- **data/**: Contains the training and testing datasets (Project_4_data.zip).
- **notebooks/**: Contains the DNABERT pre-trained model and the custom CNN model jupyter notebook.
- **scripts/**: Contains the Python scripts for data processing, model training, and evaluation.
- **results/**: Stores the results of jupyter notebook.

## Data Processing
The dataset consists of 5' UTR sequences and their corresponding translation efficiency labels. The data is processed as follows:
1. **Sequence Tokenization**: Each 5' UTR sequence is tokenized using the DNABERT tokenizer.
2. **Feature Extraction**: DNABERT is used to extract sequence embeddings, which are then passed to the CNN for further processing.
3. **Data Splitting**: The dataset is split into training and testing sets (260,000 sequences for training and 20,000 for testing).

## Model Architecture
The model consists of two main components:
1. **DNABERT**: A pre-trained transformer model that extracts contextual embeddings from the 5' UTR sequences.
2. **CNN**: A 1D convolutional neural network with three convolutional layers, batch normalization, dropout, and fully connected layers for regression.

### CNN Architecture Details:
- **Input**: DNABERT embeddings of the 5' UTR sequences.
- **Convolutional Layers**:
  - Conv1: 120 filters, kernel size 8, stride 4.
  - Conv2: 120 filters, kernel size 8, stride 1.
  - Conv3: 120 filters, kernel size 8, stride 1.
- **Batch Normalization**: Applied after each convolutional layer.
- **Dropout**: 20% dropout after convolutional layers, 50% dropout after the fully connected layer.
- **Fully Connected Layers**:
  - FC1: 120 → 40.
  - FC2: 40 → 1 (output layer for regression).

## Training
The model is trained using the Adam optimizer with a learning rate of 0.001 and L2 regularization (weight decay). The training process includes:
- **Loss Function**: Mean Squared Error (MSE) for regression.
- **Learning Rate Scheduler**: Reduces the learning rate on plateau.
- **Early Stopping**: Stops training if the validation loss does not improve for 3 consecutive epochs.

## Evaluation
The model is evaluated on the test set using the following metrics:
- **Test Loss**: Mean Squared Error (MSE) on the test set.
- **R² Score**: Coefficient of determination to measure the model's predictive power.

## Results
- **Training Loss Curve**: The training loss decreases steadily over epochs, indicating effective learning.
- **Test Loss**: The final test loss is reported, along with the R² score, which measures the model's ability to explain the variance in the data.

## Future Work
- **Model Improvement**: Explore alternative architectures or hyperparameter tuning to improve performance.
- **Data Augmentation**: Incorporate additional data or synthetic sequences to enhance model generalization.
- **Interpretability**: Analyze the model's attention mechanisms to understand which parts of the 5' UTR sequences contribute most to translation efficiency.

## References
- [DNABERT: Pre-trained Bidirectional Encoder Representations from Transformers for DNA Sequences](https://arxiv.org/abs/2106.14424)
- [Human 5’ UTR Design and Variant Effect Prediction from a Massively Parallel Translation Assay](https://www.nature.com/articles/s41586-021-03819-2)

