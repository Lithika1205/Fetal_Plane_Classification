# 🩺 Fetal Ultrasound Standard-Plane Classification using Deep Learning, Optimization & Explainable AI

## 📌 Overview

This project presents an end-to-end **AI-based fetal ultrasound standard-plane classification system**. The system classifies fetal ultrasound images into six standard categories using deep learning models and optimized feature-based machine learning.

The project combines **Vision Transformer (ViT), DenseNet121, DINOv2, feature optimization algorithms, classical machine learning classifiers, Explainable AI (XAI), and LangGraph** to build an interpretable and automated fetal ultrasound analysis pipeline.

The final system performs:

**Ultrasound Image → Deep Feature Extraction → Feature Optimization → Classification → Confidence → Explainability → Risk Assessment**

---

## 🎯 Objective

The main objective is to automatically classify fetal ultrasound images into their corresponding anatomical/standard-plane categories while providing interpretable predictions.

The six classification categories are:

1. Fetal abdomen
2. Fetal brain
3. Fetal femur
4. Fetal thorax
5. Maternal cervix
6. Other

---

## 🧠 Project Pipeline

```text
                 FETAL PLANES DB
                       │
                       ▼
                Dataset & EDA
                       │
                       ▼
              Data Preprocessing
                       │
                       ▼
        Patient-wise Train/Val/Test Split
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        ViT        DenseNet121    DINOv2
          │            │            │
          └────────────┼────────────┘
                       ▼
              Deep Model Comparison
                       │
                       ▼
             Best Deep Model Selection
                       │
                       ▼
          DenseNet121 Feature Extraction
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
         DBO          PSO          GWO
          │            │            │
          └────────────┼────────────┘
                       ▼
              Feature Optimization
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
         SVM       Random Forest   XGBoost
          │            │            │
          └────────────┼────────────┘
                       ▼
          9 Optimizer + Classifier
                Combinations
                       │
                       ▼
             Best Pipeline Selection
                       │
                       ▼
              Final Test Evaluation
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Grad-CAM       SHAP       LangGraph
          │            │            │
          └────────────┼────────────┘
                       ▼
              Final Prediction
              + Explanation
              + Risk Assessment
```

---

## 🔬 Deep Learning Models

Three deep learning architectures are evaluated:

### 1. Vision Transformer (ViT)

ViT is used for image classification by representing an image as a sequence of patches and learning global relationships between image regions.

### 2. DenseNet121

DenseNet121 is used for feature extraction and classification. Its learned representations are further optimized using feature-selection algorithms.

### 3. DINOv2 ViT-B/14

DINOv2 is used as a vision foundation model for extracting powerful visual representations from fetal ultrasound images.

---

## ⚙️ Feature Optimization

After extracting DenseNet121 features, three optimization algorithms are evaluated:

### DBO — Dung Beetle Optimizer

DBO is used to search for an effective subset of features that improves classification performance.

### PSO — Particle Swarm Optimization

PSO searches the feature space using a population-based optimization strategy.

### GWO — Grey Wolf Optimizer

GWO performs feature selection by simulating the hunting behavior and hierarchy of grey wolves.

---

## 🤖 Classification Algorithms

The optimized features are evaluated with three classical machine learning classifiers:

* **Support Vector Machine (SVM)**
* **Random Forest**
* **XGBoost**

This produces nine possible combinations:

| Optimizer | Classifier    |
| --------- | ------------- |
| DBO       | SVM           |
| DBO       | Random Forest |
| DBO       | XGBoost       |
| PSO       | SVM           |
| PSO       | Random Forest |
| PSO       | XGBoost       |
| GWO       | SVM           |
| GWO       | Random Forest |
| GWO       | XGBoost       |

The best-performing combination is automatically selected for the final pipeline.

---

## 📊 Evaluation Metrics

The final model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

The project also generates a final confusion-matrix visualization for analyzing class-wise predictions.

---

## 🔍 Explainable AI

A major component of this project is **Explainable AI (XAI)**.

### Grad-CAM

Grad-CAM is applied to the DenseNet121 model to visualize the image regions that contribute to the model's prediction.

```text
Ultrasound Image
       │
       ▼
   DenseNet121
       │
       ▼
 Model Prediction
       │
       ▼
    Grad-CAM
       │
       ▼
Important Image Regions
```

This helps provide visual insight into why a particular class was predicted.

### SHAP

SHAP is applied to the final classifier to analyze the contribution of selected features toward the model's prediction.

```text
Input Features
      │
      ▼
Final Classifier
      │
      ▼
     SHAP
      │
      ▼
Feature Contributions
```

---

## 🧩 LangGraph Agent Pipeline

The project also integrates **LangGraph** to organize the explainability and risk-assessment workflow.

The agent pipeline follows:

```text
Prediction
    │
    ▼
Explainability Agent
    │
    ▼
Risk Assessment Agent
    │
    ▼
Final Output
```

The LangGraph state can contain information such as:

* Image path
* Predicted class
* Predicted class ID
* Confidence
* Grad-CAM output
* SHAP output
* Explanation
* Risk assessment
* Final output

---

## 🚀 End-to-End Prediction

The final demonstration pipeline accepts a fetal ultrasound image and performs:

```text
Input Ultrasound Image
          ↓
DenseNet121
Feature Extraction
          ↓
Selected Features
          ↓
Best Optimizer
          ↓
Best Classifier
          ↓
Predicted Class
          ↓
Confidence Score
          ↓
Grad-CAM + SHAP
          ↓
LangGraph
          ↓
Explanation + Risk Assessment
```

---

## 🗂️ Project Structure

The notebook creates and uses the following project organization:

```text
fetal_project/
│
├── clean_metadata.csv
│
├── features/
│   ├── X_train_features.npy
│   ├── X_val_features.npy
│   ├── X_test_features.npy
│   ├── y_train_features.npy
│   ├── y_val_features.npy
│   ├── y_test_features.npy
│   └── test_feature_metadata.csv
│
├── checkpoints/
│   └── best_densenet121_model.pth
│
├── results/
│   ├── deep_model_comparison.csv
│   ├── best_optimizer_classifier_pipeline.csv
│   ├── final_best_classifier.pkl
│   ├── final_selected_feature_indices.npy
│   ├── final_test_results.csv
│   ├── final_confusion_matrix.csv
│   └── classification reports
│
├── langgraph_output/
│   └── final_langgraph_output.json
│
├── final_prediction/
│   ├── final_image_prediction.json
│   └── generated outputs
│
└── demo_output/
    └── prediction outputs
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning

* PyTorch
* Torchvision
* Vision Transformer
* DenseNet121
* DINOv2

### Machine Learning

* Scikit-learn
* XGBoost

### Optimization

* Dung Beetle Optimizer (DBO)
* Particle Swarm Optimization (PSO)
* Grey Wolf Optimizer (GWO)

### Explainable AI

* Grad-CAM
* SHAP

### Agent Framework

* LangGraph

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib

### Model Persistence

* Joblib

---

## 📦 Installation

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
pip install torch torchvision
pip install shap joblib
pip install langgraph
```

Additional packages may be required depending on the dataset and model implementation.

---

## ▶️ Running the Project

### 1. Open the Notebook

Open:

```text
project-07-final.ipynb
```

in Google Colab or Kaggle Notebook.

### 2. Configure the Dataset

Provide the FETAL PLANES DB dataset and update the dataset paths according to the execution environment.

### 3. Run the Notebook Sequentially

Execute the notebook cells in order because later stages depend on files and models generated by earlier stages.

### 4. Compare Deep Learning Models

Train/evaluate:

```text
ViT
DenseNet121
DINOv2
```

and select the best-performing deep model.

### 5. Extract Features

DenseNet121 features are extracted for subsequent optimization.

### 6. Run Feature Optimization

Evaluate:

```text
DBO
PSO
GWO
```

### 7. Evaluate Classifiers

Evaluate:

```text
SVM
Random Forest
XGBoost
```

for each optimized feature set.

### 8. Select the Best Pipeline

The best optimizer-classifier combination is selected automatically.

### 9. Evaluate the Final Model

Generate:

* Test predictions
* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Classification report

### 10. Generate Explanations

Run:

```text
Grad-CAM
SHAP
```

to explain the predictions.

### 11. Run LangGraph

The final prediction is passed through the LangGraph explainability and risk-assessment pipeline.

---

## 📈 Output

The final system produces:

* Predicted fetal ultrasound class
* Prediction confidence
* Final optimized classifier
* Confusion matrix
* Classification report
* Grad-CAM visualization
* SHAP explanation
* Explainability output
* Risk assessment
* Final pipeline report

---

## ✅ Validation

The notebook contains a final validation stage that checks the availability of required:

* Dataset metadata
* Feature files
* Labels
* DenseNet121 checkpoint
* Model-comparison results
* Optimizer/classifier results
* Final classifier
* XAI outputs
* LangGraph outputs
* Final prediction outputs

This helps verify that the complete project pipeline has been executed successfully.

---

## ⚠️ Important Note

This project is intended for **research and educational purposes**. The classification and risk-assessment outputs should not be treated as a medical diagnosis or a replacement for evaluation by qualified medical professionals.

---

## 🔮 Future Enhancements

* Improve model performance with larger datasets
* Add additional fetal ultrasound datasets
* Improve automated risk assessment
* Deploy the model as a web application
* Develop a REST API for prediction
* Add real-time ultrasound analysis
* Improve XAI visualizations
* Add more medical imaging foundation models
* Integrate clinician feedback
* Develop a complete clinical decision-support interface

---

## 👩‍💻 Author
Lithika

B.Tech – Artificial Intelligence and Data Science
Kongu Engineering College

---

## ⭐ Project Highlights

```text
✔ Fetal Ultrasound Classification
✔ ViT + DenseNet121 + DINOv2
✔ DBO + PSO + GWO Optimization
✔ SVM + Random Forest + XGBoost
✔ 9 Optimizer-Classifier Combinations
✔ Grad-CAM Explainability
✔ SHAP Feature Explainability
✔ LangGraph Agent Workflow
✔ Final End-to-End Prediction Pipeline
✔ Automated Project Validation
```

## 📌 Summary

This project combines **deep learning, feature optimization, machine learning, explainable AI, and agent-based workflows** into a single end-to-end fetal ultrasound classification system. It not only predicts the fetal ultrasound standard plane but also provides model explanations and a structured risk-assessment workflow.
