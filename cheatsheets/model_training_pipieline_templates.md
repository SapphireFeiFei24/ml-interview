# Model Training Pipeline Templates
## 1. Problem Framing
* What is the task?
  * Classification / regression / ranking
* What is the target variable
  * binary 0/1, numeric score
* What metric makes sense and why?
  * Check with label distribution
    * F1, ROC-AUC for skewed binary classification
* Any obvious leakage risks
  * Aggregation data

## 2. Data Sanity
* Data size, feature size `df.shape`
* Target distribution `df[target].value_counts()`
* Missing values `df.isnull().mean()`
* Train/val split logic
* Time leakage? User leakage?

## 3. (Always) Baseline First
> Baseline = insurance policy
* Simple features
* Simple model
* Get any score

## Feature Ideas
### Text
* length
* token count
* TF-IDF
* Embeddings (only if easy)
### Metadata
* time of day/day of week
* counts/flags
* user-level aggregates(careful!)

## Model Upgrade
* Logistic -> LightGBM
* Linear -> XGBoost
* CPU > GPU unless justified

## Evaluation & Sanity
* Compare vs baseline
* Overfitting
* Feature importance
* Example predictions
