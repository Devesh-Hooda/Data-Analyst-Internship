# Bank Marketing Campaign Analysis

## Executive Summary

This project develops a machine learning pipeline to predict whether bank customers will subscribe to a term deposit based on marketing campaign data. The analysis successfully compares three classification models: Logistic Regression, Random Forest, and K-Nearest Neighbors (with hyperparameter tuning). The project has been adapted from Google Colab to run locally in VSCode, demonstrating portability and deployment readiness.

## Business Problem

Financial institutions rely on direct marketing campaigns to attract customers for term deposits. However, traditional mass marketing approaches are inefficient and costly, with low conversion rates. The key business challenges are:

- Identifying the most likely customers to subscribe to term deposits
- Optimizing marketing spend by targeting high-potential clients
- Improving campaign ROI through data-driven decision making

## Methodology

### Data Processing Pipeline

1. **Data Loading & Validation**
   - Load bankmarketing.csv dataset
   - Validate required columns (21 expected features)
   - Handle column name discrepancies

2. **Data Cleaning**
   - Missing value imputation (median for numerical, mode for categorical)
   - Duplicate removal
   - Categorical standardization
   - Special handling for pdays feature (999 = no previous contact)

3. **Feature Engineering**
   - Target encoding ('no'→0, 'yes'→1)
   - Creation of previous_contact flag
   - Identification of categorical vs numerical features

4. **Preprocessing**
   - Standard scaling for numerical features
   - One-hot encoding for categorical features
   - Train-test split (80/20) with stratification

### Models Implemented

- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble method with 100 estimators
- **K-Nearest Neighbors**: With GridSearchCV hyperparameter tuning
  - Parameters: n_neighbors [3,5,7,9,11], weights ['uniform','distance'], metric ['euclidean','manhattan']

### Evaluation Metrics

All models evaluated on: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report

## Results and Business Recommendations

![](https://github.com/Devesh-Hooda/Data-Analyst-Internship/blob/main/All_Parameters.png)

### Model Performance Comparison

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.9093   | 0.6486    | 0.4256 | 0.5140   |
| Random Forest        | 0.9112   | 0.6374    | 0.4925 | 0.5556   |
| K-Nearest Neighbors* | 0.9027   | 0.5949    | 0.4289 | 0.5109   |

*\*After hyperparameter tuning (best: n_neighbors=9, weights='distance', metric='euclidean')*

**Dataset Statistics:**
- Train set: 32,940 samples 
- Test set: 8,236 samples 

### Key Findings

1. **Random Forest performs best** across all metrics, particularly in F1-Score (0.556)
2. Logistic Regression provides competitive accuracy (0.909) with simpler interpretability
3. KNN underperforms without tuning but improves significantly after optimization

### Business Recommendations

1. **Implement Random Forest for Production**: Offers best predictive performance
2. **Target High-Probability Customers**: Use model predictions to prioritize marketing contacts
3. **Model Monitoring**: Establish continuous performance tracking and model retraining schedule
4. **Cost-Benefit Analysis**: Focus campaigns on top 20% of predicted positive customers

## Further Steps

### Technical Enhancements
- Implement model API for real-time predictions
- Add feature importance analysis for Random Forest
- Explore advanced techniques (SMOTE for class imbalance, ensemble stacking)
- Docker containerization for deployment

### Business Expansion
- A/B testing with different model thresholds
- Customer segmentation analysis
- Multi-channel campaign optimization
- Integration with CRM systems

### Repository Structure
```
├── deveshhooda_proj2.py      # Main analysis script
├── bankmarketing.csv         # Dataset (if public)
├── DeveshHooda_Proj2.ipynb   # Colab notebook version
└── README.md                # This file
```

### Installation & Usage
```bash
# Clone repository
git clone https://github.com/Devesh-Hooda/Data-Analyst-Internship
cd Data-Analyst-Internship

# Install dependencies
pip install pandas numpy plotly scikit-learn

# Run analysis
python deveshhooda_proj2.py
```

### Contact
For questions: gvhooda@gmail.com
