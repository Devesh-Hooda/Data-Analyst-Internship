# Data-Analyst-Internship

Here's a comprehensive GitHub README.md file for your project:

```markdown
# Bank Marketing Campaign Analysis

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)

A machine learning pipeline for analyzing bank marketing campaign data, predicting client subscription to term deposits.

## 📌 Overview

This project implements a complete machine learning workflow:
- Data loading with column validation
- Comprehensive data cleaning
- Feature engineering and preprocessing
- Model training and evaluation
- Hyperparameter tuning

Three classification models are compared:
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (with hyperparameter tuning)

## 📊 Results Summary

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.9093   | 0.6486    | 0.4256 | 0.5140   |
| Random Forest        | 0.9112   | 0.6374    | 0.4925 | 0.5556   |
| K-Nearest Neighbors  | -        | -         | -      | 0.5556*  |

*\*After hyperparameter tuning*

**Dataset Statistics:**
- Train set: 32,940 samples
- Test set: 8,236 samples
- Positive class ratio: 11.27%

## 🛠️ Installation
```

1. Clone the repository:
   ```bash
   git clone https://github.com/Devesh-Hooda/Data-Analyst-Internship
   cd "insert folder name"
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy plotly scikit-learn
   ```

## 🚀 Usage

1. Run the Jupyter notebook or Python script:
   ```bash
   jupyter notebook DeveshHooda_Proj2.ipynb
   ```

2. When prompted, upload your CSV file containing bank marketing data.

3. The script will automatically:
   - Validate the dataset columns
   - Clean and preprocess the data
   - Train and evaluate models
   - Display performance metrics and visualizations

## 📂 Expected Data Format

The script expects a CSV file with the following columns (case insensitive):

```
age, job, marital, education, default, housing, loan,
contact, month, day_of_wk, duration, campaign, pdays,
previous, poutcome, emp.var.rate, cons.price.idx,
cons.conf.idx, euribor3m, nr.employed, y
```

## 🔍 Data Processing Pipeline

1. **Data Validation**:
   - Checks for missing/extra columns
   - Provides detailed column information

2. **Data Cleaning**:
   - Handles missing values (median for numerical, mode for categorical)
   - Removes duplicates
   - Standardizes categorical values
   - Special handling for `pdays` (999 → NaN → median)

3. **Feature Engineering**:
   - Creates `previous_contact` flag
   - Encodes target variable (`y`: 'no'→0, 'yes'→1)
   - Separates numerical and categorical features

4. **Preprocessing**:
   - Numerical features: Median imputation + Standard scaling
   - Categorical features: Mode imputation + One-hot encoding

## 🤖 Models Implemented

1. **Logistic Regression**
   - Baseline linear model
   - Max iterations: 1000

2. **Random Forest**
   - 100 estimators
   - Default parameters

3. **K-Nearest Neighbors**
   - Hyperparameter tuning via GridSearchCV
   - Tuned parameters: n_neighbors, weights, metric

## 📈 Performance Metrics

All models are evaluated on:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report


## 📧 Contact

For questions or suggestions, please contact: [dhooda.work@gmail.com](mailto:dhooda.work@gmail.com)
```
