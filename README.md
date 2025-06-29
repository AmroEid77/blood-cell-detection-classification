Of course. Here is the complete content in raw Markdown format. You can copy and paste this directly into a file named README.md.

Generated markdown
# Dual-Stage Deep Learning for Blood Cell Detection and Leukemia Classification

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-LTS-red.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-blueviolet)

This project presents a comprehensive deep learning pipeline for the automated analysis of blood cell images, tackling two distinct but complementary tasks:
1.  **General Blood Component Detection**: Identifying and localizing Red Blood Cells (RBC), White Blood Cells (WBC), and Platelets using various YOLOv8 architectures.
2.  **Leukemia Classification**: Classifying lymphocytes into Benign or Malignant (Acute Lymphoblastic Leukemia - ALL) subtypes using MobileNetV2, InceptionV3, and a custom-designed CNN named "MedNet".

The project emphasizes robust pre-processing, including an advanced nucleus segmentation pipeline, and a rigorous evaluation of different transfer learning and fine-tuning strategies to achieve state-of-the-art performance.

## 🌟 Key Features

- **Dual-Stage Diagnostic Workflow**: A complete pipeline from general cell detection to specific cancer classification.
- **Comparative Model Analysis**:
    - **Detection**: Systematic evaluation of YOLOv8 variants (nano, small, medium, large, x-large) with different fine-tuning strategies.
    - **Classification**: In-depth comparison of MobileNetV2, InceptionV3, and a custom "MedNet" architecture.
- **Advanced Image Segmentation**: A sophisticated pre-processing pipeline using CIELAB color space and K-Means clustering to isolate cell nuclei, significantly improving classification accuracy.
- **Rigorous Evaluation**: Performance is assessed using a comprehensive suite of metrics including mAP, F1-Score, Accuracy, Precision, Recall, Dice Coefficient, and IoU.
- **High-Performance Results**: The custom-built **MedNet** achieves **98.15% accuracy** in cancer classification, while **YOLOv8s** provides an optimal balance of speed and accuracy for detection.

## 🚀 Project Pipeline

The automated analysis workflow is structured in two key stages:

1.  **Stage 1: Object Detection (YOLOv8)**
    -   A blood smear image is fed into the trained YOLOv8 model.
    -   The model detects and localizes all primary blood components: Red Blood Cells, White Blood Cells, and Platelets.


2.  **Stage 2: Leukemia Classification (MedNet)**
    -   The bounding boxes corresponding to White Blood Cells are cropped.
    -   These cropped images are passed to the high-performance MedNet classifier.
    -   MedNet classifies each lymphocyte as *Benign* or one of the three *Malignant* subtypes (Early Pre-B, Pre-B, Pro-B).


## 📊 Models and Results

### Task 1: Blood Cell Detection

Five variants of YOLOv8 were trained with three different unfreezing strategies (unfreezing all layers, last 10, and last 3). The `YOLOv8s-all` model provided the best balance of performance and efficiency.

**Key Findings:**
- **Best F1-Score**: `YOLOv8s-all` (0.8906), making it the most balanced model for general detection.
- **Best mAP@0.5**: `YOLOv8m-all` (0.9350), offering the highest raw detection quality.
- **Best Precision**: `YOLOv8x-last 10` (0.8841), ideal for minimizing false positives.
- **Fine-Tuning Strategy**: Full fine-tuning (unfreezing all layers) consistently yielded the best performance across most model sizes.


### Task 2: Leukemia Classification

Three models were trained on a specialized dataset for Acute Lymphoblastic Leukemia. The training data underwent a sophisticated segmentation pre-processing step to isolate cell nuclei.

**Key Findings:**
- The custom-designed, fine-tuned **MedNet** model significantly outperformed established pre-trained models.
- The advanced segmentation pre-processing step played a crucial role in MedNet's success.
- **MedNet achieved a near-perfect test accuracy of 98.15%**, solidifying its suitability for this critical diagnostic task.

| Metric                        | MobileNetV2 | InceptionV3 | **MedNet (fine-tuned)** |
| ----------------------------- | ----------- | ----------- | ----------------------- |
| **Test Accuracy (%)**         | 82.00%      | 92.92%      | **98.15%**              |
| **F1-score (avg)**            | 0.79        | 0.91        | **0.98**                |
| **Precision (avg)**           | 0.83        | 0.93        | **0.98**                |
| **Recall (avg)**              | 0.78        | 0.90        | **0.98**                |
| **MAE**                       | 0.3108      | 0.1631      | **0.0554**              |
| **False Negative Rate (avg)** | 0.1920      | 0.0972      | **0.0242**              |

<p align="center">
  <img src="https://i.imgur.com/GzB0u4y.png" alt="Classification Model Comparison" width="800"/>
  <br><em>Figure: Comparison of key performance and error metrics for the classification models.</em>
</p>

## 📂 Repository Structure

```
.
├── roboflow_detection.ipynb      # Notebook for YOLOv8 detection models.
├── mobileNet.ipynb               # Notebook for MobileNetV2 classification.
├── inceptionV3.ipynb             # Notebook for InceptionV3 classification.
├── medNet.ipynb                  # Notebook for the custom MedNet classification model.
└── README.md                     # This file.
```

## 🛠️ Getting Started

### Prerequisites

Create a Python environment and install the required libraries.

```bash
pip install tensorflow keras opencv-python scikit-learn pandas seaborn matplotlib imutils pyyaml ultralytics
```

### Datasets

1.  **Classification Dataset: Blood Cell Cancer (ALL)**
    -   **Source**: [GTS.ai Dataset Collection](https://gts.ai/dataset-download/blood-cells-cancer-all-dataset-ai-data-collection/)
    -   Download and extract the dataset. The notebooks assume the data is located at `/content/drive/MyDrive/Blood cell Cancer [ALL]`. You may need to adjust the paths in the notebooks.

2.  **Detection Dataset: Roboflow Universe yolo-yejbs**
    -   **Source**: [Roboflow Universe](https://universe.roboflow.com/tfg-2nmge/yolo-yejbs)
    -   Download the dataset in YOLO format. The detection notebook assumes the data is located in `../data/detection`.

### Running the Notebooks

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/dual-stage-blood-cell-analysis.git
    cd dual-stage-blood-cell-analysis
    ```
2.  Set up the datasets as described above.
3.  Open and run the Jupyter Notebooks (`.ipynb` files) in an environment with the prerequisites installed (Google Colab is recommended for GPU access).
    -   Run `roboflow_detection.ipynb` to train and evaluate the YOLOv8 detection models.
    -   Run `mobileNet.ipynb`, `inceptionV3.ipynb`, and `medNet.ipynb` to train and compare the classification models.

## 結論 (Conclusion)

This study successfully demonstrates the power of a dual-stage deep learning pipeline for enhancing hematological diagnosis.

-   **Task-Specific Solutions are Superior**: For the high-stakes task of cancer classification, a custom-designed model (MedNet) with domain-specific pre-processing (nucleus segmentation) proved significantly more effective than general-purpose, pre-trained models.
-   **Fine-Tuning Strategy is Paramount**: The detection experiments revealed that the method of fine-tuning is as crucial as the choice of model architecture. Comprehensive fine-tuning consistently delivered superior performance.
-   **Optimal Models for a Complete Workflow**: The study identified two ideal models for a complete diagnostic workflow:
    -   **YOLOv8s-all**: An excellent candidate for rapid, initial screening to detect and count all blood cells.
    -   **Fine-tuned MedNet**: The definitive choice for the subsequent, critical step of classifying lymphocytes, offering unparalleled accuracy and reliability.
