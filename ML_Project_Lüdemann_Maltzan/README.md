# Potato Leaf Disease Detection with Deep Learning

This Python project implements a deep learning-based image classification pipeline to identify various potato diseases from leaf or tuber images using transfer learning. It leverages the MobileNetV2 architecture pretrained on ImageNet and fine-tunes it to classify seven categories: Black Scurf, Blackleg, Common Scab, Pink Rot, Dry Rot, Miscellaneous, and Healthy Potatoes.


## Dataset

- **Source**: [Kaggle – Potato Diseases Dataset by mukaffimoin](https://www.kaggle.com/datasets/mukaffimoin/potato-diseases-datasets)
- **Description**: A labeled collection of potato tuber images containing healthy samples and several disease categories.
- **Two ways to load data**  
  The code supports both local folders and automatic download.  
  ```python
  # Option 1: use a local copy
  local_dataset_path = "path/to/your/potato_dataset"

  # Option 2: leave it as None and the dataset will be
  # downloaded from Kaggle via kagglehub
  local_dataset_path = None
  ```
  When `local_dataset_path` is `None`, the script calls  
  `kagglehub.dataset_download("mukaffimoin/potato-diseases-datasets")` and extracts the archive automatically.  
  If you provide a path, the code skips the download and loads images directly from that folder.


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


## License

This project is released for educational and research purposes. Please credit the dataset creator and cite this repository if used in your work.
