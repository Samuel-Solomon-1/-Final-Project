# Predictive Maintenance Using Machine Learning

## Project Overview

This project demonstrates a real-world AI application for **predictive maintenance**, using historical sensor data to predict the likelihood of equipment failure. The goal is to help industries take proactive action to prevent costly downtimes and improve operational efficiency.

Using a **Random Forest classifier**, the project walks through all key machine learning stages—from data preprocessing to deployment in a Google Colab environment.

---

## Problem Definition

Predictive maintenance aims to forecast when equipment will fail so that maintenance can be performed just in time. In this project, the task is framed as a **binary classification problem**:
- `0`: Normal operation
- `1`: Imminent equipment failure

---

## Dataset

**Source**: [Predictive Maintenance Dataset (UCI/Kaggle)]  
**Rows**: 124,494  
**Columns**: 12 (including metrics, timestamp, device ID, and failure label)

### Key Features:
- `metric1` to `metric9`: Various anonymized sensor readings  
- `failure`: Target variable (0 or 1)  
- `device`, `date`: Identifiers and timestamps (used for analysis, not modeling)

---

## Tasks Performed

### Task 1: Problem Definition and Dataset Preparation
- Loaded a publicly available predictive maintenance dataset.
- Performed initial data checks and printed column info and shape.
- Verified no missing values and prepared dataset for modeling.

### Task 2: Data Preprocessing and Exploration
- Converted date to datetime.
- Dropped unused columns (`date`, `device`).
- Split data into train-test sets with 80-20 ratio.
- Verified class imbalance in target labels.

### Task 3: Model Selection and Development
- Chose **Random Forest** as the base classifier.
- Applied `GridSearchCV` for hyperparameter tuning.
- Used `class_weight='balanced'` to address imbalance.
- Trained the model using the best parameters.

### Task 4: Model Evaluation and Optimization
- Evaluated using:
  - Classification Report
  - Confusion Matrix
  - ROC-AUC Score
- Result:
  - **Accuracy:** 99% (high due to imbalance)
  - **Recall on Failures:** 0.0 (needs improvement)
  - **AUC Score:** 0.71 (indicates moderate separability)

### Task 5: Model Deployment and Presentation
- Built a simple interactive prediction function.
- Allowed users to input custom sensor readings and get a prediction.
- Showed probability of failure with a user-friendly status indicator.

---

## Model Performance Summary

| Metric         | Value      |
|----------------|------------|
| Accuracy       | 99%        |
| Precision (1)  | 0.00       |
| Recall (1)     | 0.00       |
| AUC Score      | 0.71       |

⚠️ Note: High accuracy is due to class imbalance. Further work (resampling, ensemble methods) is needed to better detect rare failures.

---

## Insights & Future Improvements

- **Class Imbalance** is a major challenge.
  - Suggested: SMOTE, undersampling, anomaly detection.
- **Model Fairness** can be evaluated across devices or dates.
- **Explainability** using SHAP/LIME can improve trust.
- **Deployment** could be extended to a web or edge-based solution.

---

## Try It Yourself

Run the notebook in [Google Colab](https://colab.research.google.com/) and test with your own input:

```python
sample_input = {
    'metric1': 120000000,
    'metric2': 5,
    ...
}
predict_failure(sample_input)
````

---

## Project Structure

```
├── Final Project.ipynb  # Main notebook
└── README.md
```

## Acknowledgments

* Dataset providers (UCI/Kaggle)
* Scikit-learn, Pandas, Seaborn, Matplotlib
* Google Colab for hosting
