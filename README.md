# Potato Leaf Disease Detection with Deep Learning

This Python project implements a deep learning-based image classification pipeline to identify various potato diseases from leaf or tuber images using transfer learning. It leverages the MobileNetV2 architecture pretrained on ImageNet and fine-tunes it to classify seven categories: Black Scurf, Blackleg, Common Scab, Pink Rot, Dry Rot, Miscellaneous, and Healthy Potatoes.

## Dataset

- **Source**: [Kaggle – Potato Diseases Dataset by mukaffimoin](https://www.kaggle.com/datasets/mukaffimoin/potato-diseases-datasets)
- **Description**: A labeled collection of potato leaf images containing healthy leaves and several disease categories.
- **Access**: The dataset is downloaded via `kagglehub` and automatically extracted in the project workflow.

## Project Overview

This project includes:

- **Exploratory Data Analysis (EDA)**:
  - Distribution of image categories
  - Image resolution analysis across categories
  - Color metrics: brightness, blurriness, contrast, colorfulness, entropy
  - Estimation and visualization of dominant background colors
  - t-SNE dimensionality reduction for feature visualization
- **Data Preparation**:
  - Stratified splitting into training, validation, and test sets
  - Custom preprocessing pipeline including white-background removal
  - Data augmentation using `tf.keras.layers`
  - Computation of class weights to address imbalance
- **Model Architecture**:
  - EfficientNetV2B1 pretrained on ImageNet
  - Global average pooling and dropout
  - Two-phase training: initial freezing, then fine-tuning
- **Evaluation**:
  - Accuracy and loss curves
  - Confusion matrix and classification report
  - Visual comparison of model performance pre- and post-finetuning

## Installation

Install the required Python packages:

```bash
pip install tensorflow numpy pandas opencv-python pillow seaborn scikit-image scikit-learn webcolors matplotlib kagglehub
```

If you are using a GPU, make sure to install the correct version of TensorFlow for GPU support.
[Install TensorFlow](https://www.tensorflow.org/install/pip)

## Results

- High classification accuracy across multiple disease types
- Clear separation of classes in t-SNE projections
- Insightful visualizations for image properties and dataset bias
- Modular and reproducible pipeline suitable for transfer to similar agricultural datasets


## License

This project is released for educational and research purposes. Please credit the dataset creator and cite this repository if used in your work.
