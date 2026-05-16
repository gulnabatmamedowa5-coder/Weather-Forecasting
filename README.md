# Weather Forecasting Using Pattern Recognition & Machine Learning

**Author**: Gulnabat Mamedova  
**Course**: Principles of Pattern Recognition  
**Date**: May 2026

---

## Project Overview

This project applies multiple Pattern Recognition and Machine Learning techniques to predict daily rainfall in Australia using both numerical weather observations and cloud image classification.

The project combines:
- Classical Machine Learning
- Ensemble Learning
- Unsupervised Clustering
- Deep Learning (CNNs)
- Computer Vision

The dataset contains over **145,000 Australian weather observations** collected between 2007–2017 from 49 locations.

**Best Tuned Model**: Random Forest (F1 = 0.644 | AUC = 0.873)

**Key Finding**: Afternoon humidity, atmospheric pressure, and wind gust speed are the strongest predictors of rainfall.

---

## Dataset

### Weather Dataset
- **File**: `weatherAUS_2(2).csv`
- **Samples**: 145,460 weather observations
- **Target Variable**: `RainTomorrow`

### Cloud Image Dataset
- **Folder**: `CCSN`
- **Images**: 2,543 cloud photographs
- **Classes**:
  - Rain Clouds
  - No-Rain Clouds

---

## Repository Contents

| File | Description |
|---|---|
| `weather_notebook.ipynb` | Main Jupyter Notebook |
| `fig1_kmeans_elbow_method.png` | K-Means elbow curve |
| `fig2_logistic_regression_confusion_matrix.png` | Logistic Regression confusion matrix |
| `fig3_random_forest_confusion_matrix.png` | Random Forest confusion matrix |
| `fig4_SVM_linear_confusion_matrix.png` | Linear SVM confusion matrix |
| `fig5_gradient_boosting_confusion_matrix.png` | Gradient Boosting confusion matrix |
| `fig6_all_model_performance_comparison.png` | Model comparison chart |
| `fig7_effect_of_hyperparameter_tuning.png` | Hyperparameter tuning results |
| `fig8_CNN_cloud_classifier_confusion_matrix.png` | CNN confusion matrix |
| `fig9_CNN_live_predictions.png` | CNN live predictions |
| `fig10_final_performance_heatmap.png` | Final model performance heatmap |

---

## Models Used

### Machine Learning
- Logistic Regression
- Linear SVM
- Random Forest
- Gradient Boosting
- K-Means Clustering

### Deep Learning
- MobileNetV2 CNN (Transfer Learning)

---

## Results

### Model Performance

| Model                     | F1 Score | AUC |
|---|---|---|
| **Random Forest (Tuned)** | **0.644** | **0.873** |
| Logistic Regression       | 0.639 | 0.881 |
| Linear SVM | 0.637        | 0.881 |
| Gradient Boosting         | 0.628 | 0.862 |
| CNN (Cloud Images)        | 0.604 | 0.820 |
| Naive Baseline            | 0.000 | 0.500 |

---

## Key Findings

- Humidity3pm is the strongest predictor of rainfall.
- Linear models performed surprisingly well on weather data.
- Hyperparameter tuning significantly improved Random Forest performance.
- CNNs successfully learned rain-associated cloud structures directly from images.
- Both numerical and visual models discovered the same meteorological patterns.

---

## Technologies & Libraries

### Core Libraries

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn imbalanced-learn tensorflow keras flask statsmodels
```

### Main Python Libraries Used

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- imbalanced-learn (SMOTE)
- SciPy
- Flask
- Statsmodels

---

## Setup Instructions

### Option 1: Conda (Recommended)

```bash
conda create -n weather_ml python=3.10
conda activate weather_ml

pip install numpy pandas matplotlib seaborn scipy scikit-learn imbalanced-learn tensorflow keras flask statsmodels jupyter
```

### Run Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```bash
weather_notebook.ipynb
```
## 📥 Dataset Download Links

### Australian Weather Dataset
Source: Kaggle — Rain in Australia Dataset

[Rain in Australia Dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

### Cloud Image Dataset (CCSN)
Source: CCSN Cloud Dataset

[CCSN Cloud Dataset](https://github.com/poloclub/cloud-classification-cnn)

---
