# Breast Cancer Ultrasound Image Classification and Segmentation

This Project contains code for classifying and segmenting breast cancer ultrasound images using Vision Transformer (ViT) and Mask2Former models. The dataset used is the Breast Ultrasound Dataset, which includes images categorized into three classes: normal, benign, and malignant.

## Introduction

Breast cancer is one of the most common causes of death among women worldwide. Early detection is crucial for reducing mortality rates. This project leverages machine learning models to classify and segment breast cancer ultrasound images, aiding in early diagnosis.

## Dataset

The dataset consists of 780 breast ultrasound images from 600 female patients aged between 25 and 75 years. The images are categorized into three classes:

- **Normal**
- **Benign**
- **Malignant**

Each image is accompanied by a corresponding mask image for segmentation tasks.

You can download dataset from http://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset .

## Model Architecture

### Vision Transformer (ViT)

The Vision Transformer model is used for classifying the ultrasound images into the three categories. The model architecture includes:

- **Patch Embedding**: Converts the image into patches and embeds them into a high-dimensional space.
- **Class Token and Positional Encoding**: Adds a class token and positional information to the patches.
- **Transformer Layers**: Multiple layers of self-attention and feed-forward networks to capture global context.
- **Classification Head**: A final MLP layer for classification.

### Mask2Former

Mask2Former is used for segmenting the tumor regions in the ultrasound images. The model architecture includes:

- **Backbone**: ResNet-50 for feature extraction.
- **Transformer Encoder**: Processes the features using self-attention mechanisms.
- **Mask Head**: Generates segmentation masks for the tumor regions.

## Data Processing

The dataset is preprocessed using the following steps:

- **Image Resizing**: Images are resized to 256x256 pixels.
- **Data Augmentation**: Random horizontal flipping and rotation are applied to the images and masks.
- **Normalization**: Images are normalized using ImageNet's mean and standard deviation.

## Training

The models are trained using the following configurations:

- **Loss Function**: Focal Loss for classification and CrossEntropyLoss for segmentation.
- **Optimizer**: Adam optimizer with a learning rate of 0.0001.
- **Learning Rate Scheduler**: ReduceLROnPlateau for dynamic learning rate adjustment.
- **Early Stopping**: Patience of 5 epochs to prevent overfitting.

## Testing

The models are evaluated on a separate test set using the following metrics:

- **Classification**: Accuracy and Confusion Matrix.
- **Segmentation**: IoU (Intersection over Union) and Dice Score.

## Results

### Vision Transformer

- **Test Accuracy**: ~58% (before data cleaning), ~43% (after data cleaning).
- **Confusion Matrix**: Shows the distribution of predictions across the three classes.

### Mask2Former

- **Average IoU**: 0.6334 (before data cleaning), 0.6645 (after data cleaning).
- **Average Dice Score**: 0.7259 (before data cleaning), 0.7640 (after data cleaning).

## Additional Notes

The dataset contains some issues such as duplicate images, unrelated structures, and ambiguous labels. Cleaning the dataset by removing these problematic images improved the segmentation performance but reduced the classification accuracy due to the smaller dataset size.

## References

1. Pawłowska, Anna, Piotr Karwat, and Norbert Żołek. Letter to the Editor. Re: “[Dataset of breast ultrasound images by W. Al-Dhabyani, M. Gomaa, H. Khaled & A. Fahmy, Data in Brief, 2020, 28, 104863]. *Data in Brief*, vol. 48, 2023, p. 109247. https://doi.org/10.1016/j.dib.2023.109247.
2. Dosovitskiy, Alexey, et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *arXiv preprint arXiv:2010.11929v2* (2021). https://doi.org/10.48550/arXiv.2010.11929.
3. Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & Girdhar, R. (2022). Masked-attention Mask Transformer for universal image segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
