# Interview Cheatsheet
## Pipeline
> Input: raw training data
> Ouput: ready-to-train data

### Step 1. Dataset Exploration
> Check the data before preprocessing
#### Questions to Answer
> Check the data, understand the feature semantics
* Q1: What's data size
  * To understand the ideal feature size, AND
  * what's the ideal model complexity
* Q2: What are the features
  * Ids, Locations: Emb
  * Count based: Empty? Category? Skewed?
  * Numeric: Empty? Skewed?
  * (str)Category: Empty? Dimension?
  * Aggregated features: Empty?
    * Valid?: Potential leakage. Drop or recalculate if not.
  * Text: Empty?
    * Cap and Embedding?
    * Size, Sentimental, Other scores -> Transfer to numeric if meaningful
  * Time-based
    * Relevant time: time_since_xxx
    * Sin+Cos: Periodical, e.g. time of the day
    * Explicit: is_holiday, is_weekend
* Q3. Any data cleaning needed?
  * Empty ids
  * Invalid aggregated features
  * Noisy data, data that might lead to leakage
    * Temporal leakage
    * Target leakage: features derived from engagement itself
    * Cross-entity leakage: user stats computed using test users
    * Post-ranking leakage: only available after ranking/exposure
  * Data with high missing rate: mostly empty, no longer meaningful
* Q4. What's the scenario?
  * How to split data?
    * Time sensitive: fraud, rcmd, search
      * Split by date
    * Pattern Recognition: Random splitting
* Q5: What's the Label
  * Skewed? need resampling
  * Delayed feedback?

### Step 2. Dataset Cleaning and Splitting
> Do it before preprocessing so that,
> * Prevent data leakage for avg/max like features
> * Speedup process
1. Clean those invalid data records based on features and some sementic logic
2. Split data according to semantics

### Step 3. Feature Engineering
> For each of the feature, do:
> 1. Check semantics, distribution
> 2. Fill empty features: 0, "UNK"
> 3. Transform: most_common, cap, log, encode
> 4. (optional) Normalization: Standardization, min-max
> 5. (optional) Cardinality reduction

#### Distribution
* Heavy tail
  * rule-of-thumb: p99 >> p90(3x-10x)

#### Normalization
* Fit during training, fix during testing

### Step 4. Other Complex Features
* Sentimental score over raw text
* Count->Signal: is_popular_user, is_popular_tweet
  * Coldstart
* Time -> Signal: is_holiday

#### Cold-start handling
* Default features
* Rule-based fallbacks
* Backoff models

#### Crossed features
* user_activity_level x content_freshness

# Rule-of-thumbs
## Label skew
`Positive rate = #positive/#total`
* 30%-50%: balanced
* 5%-30%: mild skew, ACC becomes meaningless
  * Stratified split: making sure each split has appx the same label distri as the full dataset
  * PR-AUC: Precision over Recall auc, focus on true positive class performance
* 1% - 5%: skewed
  * Reweighting, calibration
* 0.1%-1%: highly skewed
  * Sampling + ranking loss
* < 0.1%: extreme
  * No longer a binary classification, reframe as ranking
  * Two-stage models, recall-first

## Feature missing rate
> Keep if missingness is informative
* < 5%: Safe
* 5% - 20%: Likely, add missing indicator
* 20% - 40%: Depends
* 40% - 70%: Rarely, cold0-start only
* \> 70%: Almost always useless

## Feature count vs Data size
### Linear/Logistic Regression
```python
N >= 10~20 x D
```
### Tree-based
More forgiving, can trimming

### DNN
```python
N >= 5-10 x #parameters
```

### Other
*As skew increases, effective data shrinks. Positives - not total rows, determin how many features I can trust.*

