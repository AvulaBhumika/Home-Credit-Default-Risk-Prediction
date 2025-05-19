# Home Credit Default Risk Prediction

## 📌 Project Objective

The goal of this project is to build a machine learning model that predicts whether a loan applicant will default or not. The data is provided by Home Credit and includes demographic, financial, and historical loan information. Due to a high class imbalance, special care is taken to build a robust and fair model.

---

## 🔍 End-to-End Process

### 1. **Data Ingestion and Preprocessing**

* Loaded `application_train.csv` and merged auxiliary tables (bureau, previous\_application, installments\_payments, credit\_card\_balance).
* Aggregated key statistics (mean) from auxiliary datasets for each customer (SK\_ID\_CURR).
* Final dataset contained \~460,000 records with 120 features after cleaning.

### 2. **Feature Selection**

* Dropped irrelevant columns and those with >40% missing values.
* Selected important categorical variables (e.g., `NAME_CONTRACT_TYPE`, `CODE_GENDER`, `NAME_INCOME_TYPE`).
* Selected relevant numerical features including credit and demographic features.
* Applied Label Encoding on categorical features.

### 3. **Train-Test Split**

* Stratified split on the `TARGET` column to maintain class imbalance.
* 80% training and 20% testing data.

### 4. **Model Building**

* Models Used:

  * RandomForestClassifier
  * XGBoostClassifier
  * LightGBMClassifier
* Created a Stacking Ensemble:

  * Base learners: RF, XGBoost, LightGBM
  * Final estimator: RandomForestClassifier
  * Used `scale_pos_weight` for XGBoost and LightGBM due to class imbalance

### 5. **Evaluation Metrics**

* Confusion Matrix
* Classification Report
* ROC AUC Score
* Feature Importances (averaged from base learners)
* ROC Curve Plot


  ![image](https://github.com/user-attachments/assets/9c072f65-1165-487e-8638-eea6cb1227e8)


---

## 📊 Model Comparison Report

| Model             | Accuracy  | Precision | Recall   | F1-Score | ROC-AUC   |
| ----------------- | --------- | --------- | -------- | -------- | --------- |
| Random Forest     | 91.4%     | 28%       | 6%       | 10%      | 0.66      |
| XGBoost           | 91.6%     | 30%       | 5.7%     | 9.5%     | 0.675     |
| LightGBM          | 91.2%     | 27%       | 5.5%     | 9%       | 0.67      |
| **Stacked Model** | **91.3%** | **32%**   | **5.4%** | **8%**   | **0.678** |

✅ **Best Model for Production**: The **Stacked Ensemble** due to its relatively higher recall and ROC-AUC. Even though precision and recall are low (due to class imbalance), it performs best among all tested models.

---

## 🚧 Challenges Faced

### 1. **Severe Class Imbalance**

* Only \~8% of records belong to the positive class (`TARGET = 1`).
* Applied:

  * `scale_pos_weight` in XGBoost and LightGBM
  * `class_weight='balanced'` in RandomForestClassifier
  * Stratified train-test split

### 2. **High Cardinality and Mixed Data Types**

* Several categorical columns had too many unique values or dirty formats.
* LabelEncoding was used instead of OneHotEncoding to prevent dimensionality explosion.

### 3. **Missing Values**

* Some columns (e.g., `EXT_SOURCE_3`) had missing values.
* Used only features with less than 40% missing values.

### 4. **Model Interpretability**

* Extracted feature importance from all base learners and averaged them.
* Visualized Top 20 important features to interpret model decisions.

---

## ✅ Final Model Metrics

* **Confusion Matrix:**

```
[[55943   595]
 [ 4711   254]]
```

* **ROC AUC Score:** 0.678

---

## 📌 Conclusion

While absolute recall and precision for defaulters remain low due to class imbalance, the ensemble model shows improvements in ROC-AUC and balanced performance. With further steps like SMOTE or threshold tuning, performance can be improved.

---

## 🛠️ Tools & Libraries

* Python, Pandas, NumPy
* Scikit-learn, XGBoost, LightGBM
* Seaborn, Matplotlib

**~AvB**
