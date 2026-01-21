# pneumonia-detection-xray-cnn
CNN model for binary classification for chest x-rays


## Overview
This project implements a **custom convolutional neural network (CNN)** to classify chest X-ray images as **Normal** or **Pneumonia** using a real-world medical imaging dataset. The focus is on **applied machine learning**, model evaluation, and **error analysis for a safety-critical classification task**, where minimizing false negatives is essential.

The project was developed end-to-end, covering data preprocessing, CNN architecture design, training, evaluation, and performance analysis.

## Key Highlights
- Built and trained a **custom CNN from scratch** using TensorFlow  
- Evaluated performance using **precision, recall, F1-score, and confusion matrices**  
- Prioritized **recall** to reduce false negatives in pneumonia detection  
- Achieved **79.3% test accuracy** and **99% recall** on pneumonia cases  
- Conducted error analysis and proposed improvements such as **data augmentation and transfer learning**


## Dataset
- **Source:** Public chest X-ray dataset (Kaggle)  
- **Size:** ~5,800 labeled images  
- **Classes:** Normal, Pneumonia  
- **Challenges:** Class imbalance and high cost of false negatives  

Images were resized, normalized, and split into training, validation, and test sets.


## Model Architecture
The CNN was designed from scratch and includes:
- Multiple convolutional blocks  
- Batch normalization for training stability  
- Dropout layers to mitigate overfitting  
- Fully connected layers for classification  

The architecture emphasizes **generalization and recall**, rather than maximizing raw accuracy.


## Training & Evaluation
The model was trained using supervised learning and evaluated with multiple metrics to capture real-world performance tradeoffs.

**Metrics Used**
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

**Results**
- **Test Accuracy:** 79.3%  
- **Recall (Pneumonia):** 99%  

Confusion matrices were used to analyze misclassification patterns and failure cases.


## Error Analysis & Proposed Improvements
Post-training analysis identified common failure modes and opportunities for improvement, including:
- Data augmentation to improve robustness  
- Transfer learning with pre-trained CNNs (e.g., ResNet, MobileNet)  
- Additional hyperparameter tuning  

These improvements were identified but not implemented in this iteration.


## Technologies Used
- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**


## How to Run
1. Install required Python dependencies.
2. Download the dataset and place it in the `data/` directory.
3. Run the training and evaluation notebooks in order.

