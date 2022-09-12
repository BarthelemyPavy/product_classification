# Algorithm

## Data Exploration

Take a look to **[this notebook](../../../notebooks/EDA.ipynb)** to deep dive into my exploration

## Divide and Rule

[Notebook](../../../product_classification/split_dataset.ipynb)

About train, test and validation step, we have two choices:

- Stratify split based on each category (as single label classification)
- Iterative split from this [Paper](http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf)

As we could see from exploratory analysis, less than 1% of product belongs to several categories so I choose to use **stratify split**

## Deal with unbalance data

In order to tackle unbalance classes problem I combined several approach.

#### Baseline and CNN approach
- Remove categories with few samples (not enough to be used)
- Down sampling on categories with too much examples

#### CNN approach
- Apply weights to loss function
- Use focal loss to add more weight on uncorrected predictions

## Baseline Approach

[Notebook](../../../product_classification/baseline.ipynb)
### Input data

- price
- brand_name and merchant name as categorical data (One-Hot Encoded)
- product_name process with (Stemmatization + TF-IDF)

### Classifier
- LGBM 

## CNN text classifier
[Notebook](../../../product_classification/cnn.ipynb)

### Input data
- brand_name and merchant name as categorical data (One-Hot Encoded)
- product_name embedded with fasttext

### Classifier

- [Text CNN Classifier](https://arxiv.org/pdf/1408.5882.pdf)

## Evaluation

As we don't define a preference between recall and precision we can take a look to F1-score:

- samples avg
- weighted average
- micro avg

I used also hamming_loss to follow validation set performance on CNN approach.

We could see that CNN may be a better approach (if we taking account only metrics performances) even if precision seems to be bad, it's because focal loss parameters needs to be re-calibrate to reduce the importance of incorrect predictions (High recall, Medium/Bad precision).

In term of computation time:

- LGBM is 2x faster than CNN
