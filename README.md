Of course! Here is a comprehensive README.md file generated from the provided notebooks and research paper. It summarizes the project's dual-stage approach, methodologies, results, and instructions for replication.

Dual-Stage Deep Learning for Blood Cell Detection and Leukemia Classification

![alt text](https://img.shields.io/badge/Python-3.10-blue.svg)


![alt text](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)


![alt text](https://img.shields.io/badge/PyTorch-LTS-red.svg)


![alt text](https://img.shields.io/badge/Keras-2.x-red.svg)


![alt text](https://img.shields.io/badge/YOLO-v8-blueviolet)

This project presents a comprehensive deep learning pipeline for the automated analysis of blood cell images, tackling two distinct but complementary tasks:

General Blood Component Detection: Identifying and localizing Red Blood Cells (RBC), White Blood Cells (WBC), and Platelets using various YOLOv8 architectures.

Leukemia Classification: Classifying lymphocytes into Benign or Malignant (Acute Lymphoblastic Leukemia - ALL) subtypes using MobileNetV2, InceptionV3, and a custom-designed CNN named "MedNet".

The project emphasizes robust pre-processing, including an advanced nucleus segmentation pipeline, and a rigorous evaluation of different transfer learning and fine-tuning strategies to achieve state-of-the-art performance.

🌟 Key Features

Dual-Stage Diagnostic Workflow: A complete pipeline from general cell detection to specific cancer classification.

Comparative Model Analysis:

Detection: Systematic evaluation of YOLOv8 variants (nano, small, medium, large, x-large) with different fine-tuning strategies.

Classification: In-depth comparison of MobileNetV2, InceptionV3, and a custom "MedNet" architecture.

Advanced Image Segmentation: A sophisticated pre-processing pipeline using CIELAB color space and K-Means clustering to isolate cell nuclei, significantly improving classification accuracy.

Rigorous Evaluation: Performance is assessed using a comprehensive suite of metrics including mAP, F1-Score, Accuracy, Precision, Recall, Dice Coefficient, and IoU.

High-Performance Results: The custom-built MedNet achieves 98.15% accuracy in cancer classification, while YOLOv8s provides an optimal balance of speed and accuracy for detection.

🚀 Project Pipeline

The automated analysis workflow is structured in two key stages:

Stage 1: Object Detection (YOLOv8)

A blood smear image is fed into the trained YOLOv8 model.

The model detects and localizes all primary blood components: Red Blood Cells, White Blood Cells, and Platelets.

Figure: A sample image with ground-truth bounding box annotations for each object class.

Stage 2: Leukemia Classification (MedNet)

The bounding boxes corresponding to White Blood Cells are cropped.

These cropped images are passed to the high-performance MedNet classifier.

MedNet classifies each lymphocyte as Benign or one of the three Malignant subtypes (Early Pre-B, Pre-B, Pro-B).

Figure: The segmentation pipeline used for classification model training, showing the progression from the original image (a) to the final segmented cell (f).

📊 Models and Results
Task 1: Blood Cell Detection

Five variants of YOLOv8 were trained with three different unfreezing strategies (unfreezing all layers, last 10, and last 3). The YOLOv8s-all model provided the best balance of performance and efficiency.

Key Findings:

Best F1-Score: YOLOv8s-all (0.8906), making it the most balanced model for general detection.

Best mAP@0.5: YOLOv8m-all (0.9350), offering the highest raw detection quality.

Best Precision: YOLOv8x-last 10 (0.8841), ideal for minimizing false positives.

Fine-Tuning Strategy: Full fine-tuning (unfreezing all layers) consistently yielded the best performance across most model sizes.

Figure: Performance comparison of YOLOv8 variants.

Task 2: Leukemia Classification

Three models were trained on a specialized dataset for Acute Lymphoblastic Leukemia. The training data underwent a sophisticated segmentation pre-processing step to isolate cell nuclei.

Key Findings:

The custom-designed, fine-tuned MedNet model significantly outperformed established pre-trained models.

The advanced segmentation pre-processing step played a crucial role in MedNet's success.

MedNet achieved a near-perfect test accuracy of 98.15%, solidifying its suitability for this critical diagnostic task.

Metric	MobileNetV2	InceptionV3	MedNet (fine-tuned)
Test Accuracy (%)	82.00%	92.92%	98.15%
F1-score (avg)	0.79	0.91	0.98
Precision (avg)	0.83	0.93	0.98
Recall (avg)	0.78	0.90	0.98
MAE	0.3108	0.1631	0.0554
False Negative Rate (avg)	0.1920	0.0972	0.0242

Figure: Comparison of key performance and error metrics for the classification models.

📂 Repository Structure
Generated code
.
├── roboflow_detection.ipynb      # Notebook for YOLOv8 detection models.
├── mobileNet.ipynb               # Notebook for MobileNetV2 classification.
├── inceptionV3.ipynb             # Notebook for InceptionV3 classification.
├── medNet.ipynb                  # Notebook for the custom MedNet classification model.
├── assets/                       # Directory for storing images and visual assets.
└── README.md                     # This file.

🛠️ Getting Started
Prerequisites

Create a Python environment and install the required libraries.

Generated bash
pip install tensorflow keras opencv-python scikit-learn pandas seaborn matplotlib imutils pyyaml ultralytics
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
Datasets

Classification Dataset: Blood Cell Cancer (ALL)

Source: GTS.ai Dataset Collection

Download and extract the dataset. The notebooks assume the data is located at /content/drive/MyDrive/Blood cell Cancer [ALL]. You may need to adjust the paths in the notebooks.

Detection Dataset: Roboflow Universe yolo-yejbs

Source: Roboflow Universe

Download the dataset in YOLO format. The detection notebook assumes the data is located in ../data/detection.

Running the Notebooks

Clone the repository:

Generated bash
git clone https://github.com/your-username/dual-stage-blood-cell-analysis.git
cd dual-stage-blood-cell-analysis
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Set up the datasets as described above.

Open and run the Jupyter Notebooks (.ipynb files) in an environment with the prerequisites installed (Google Colab is recommended for GPU access).

Run roboflow_detection.ipynb to train and evaluate the YOLOv8 detection models.

Run mobileNet.ipynb, inceptionV3.ipynb, and medNet.ipynb to train and compare the classification models.

結論 (Conclusion)

This study successfully demonstrates the power of a dual-stage deep learning pipeline for enhancing hematological diagnosis.

Task-Specific Solutions are Superior: For the high-stakes task of cancer classification, a custom-designed model (MedNet) with domain-specific pre-processing (nucleus segmentation) proved significantly more effective than general-purpose, pre-trained models.

Fine-Tuning Strategy is Paramount: The detection experiments revealed that the method of fine-tuning is as crucial as the choice of model architecture. Comprehensive fine-tuning consistently delivered superior performance.

Optimal Models for a Complete Workflow: The study identified two ideal models for a complete diagnostic workflow:

YOLOv8s-all: An excellent candidate for rapid, initial screening to detect and count all blood cells.

Fine-tuned MedNet: The definitive choice for the subsequent, critical step of classifying lymphocytes, offering unparalleled accuracy and reliability.
