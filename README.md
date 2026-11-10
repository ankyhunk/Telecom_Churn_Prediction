# Telecom Churn Prediction

Predicting which telecom customers are likely to **churn** (switch to another operator), and finding the main indicators of churn. Kaggle competition from the upGrad & IIIT-B Machine Learning programme.

## Problem statement

In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.

For many incumbent operators, retaining high profitable customers is the number one business goal. To reduce customer churn, telecom companies need to predict which customers are at high risk of churn. In this project, you will analyze customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn, and identify the main indicators of churn.

In this competition, your goal is to build a machine learning model that is able to predict churning customers based on the features provided for their usage.

### Customer behaviour during churn

Customers usually do not decide to switch to another competitor instantly, but rather over a period of time (this is especially applicable to high-value customers). In churn prediction, we assume that there are three phases of customer lifecycle:

- **The 'good' phase:** In this phase, the customer is happy with the service and behaves as usual.
- **The 'action' phase:** The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than the 'good' months. It is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor's offer/improving the service quality etc.)
- **The 'churn' phase:** In this phase, the customer is said to have churned. In this case, since you are working over a four-month window, the first two months are the 'good' phase, the third month is the 'action' phase, while the fourth month (September) is the 'churn' phase.

## Dataset

From the Kaggle competition *kagglelabstest2021* (not included in this repository):

| File | Rows | Description |
|---|---|---|
| `train.csv` | 69,999 | Customer usage over June–August, plus the `churn_probability` label |
| `test.csv` | 30,000 | Same features, no label (used for the submission) |
| `sample.csv` | 30,000 | Submission format: `id`, `churn_probability` |
| `data_dictionary.csv` | 36 | Description of the columns |

## Approach

1. **Missing data:** visualised with `missingno`; columns with too many missing values dropped, the rest filled with 0
2. **Outliers:** capped with the k-sigma method
3. **Scaling:** `StandardScaler`
4. **Feature selection:** feature importances and **PCA** (keeping 90% of the variance)
5. **Models:**
   - XGBoost in a pipeline (impute → scale → PCA → XGBoost), tuned with `RandomizedSearchCV`
   - Random Forest tuned with `GridSearchCV`

## Results

| Model | Train accuracy | Test accuracy |
|---|---|---|
| XGBoost + PCA pipeline | 93.8% | 91.8% |
| Random Forest | 93.7% | 93.5% (precision 73%, recall 60%) |

**Strongest predictors of churn:**

1. Total outgoing minutes in August (the 'action' month)
2. Total off-network minutes in August
3. Total recharge amount in August

A drop in usage and recharges during the action month is the clearest early warning sign.

## How to run

```bash
pip install numpy pandas matplotlib seaborn missingno scikit-learn xgboost
jupyter notebook "Telecom Churn Prediction.ipynb"
```

Download the competition files from Kaggle and set the paths in the "Loading dependencies & datasets" section (the notebook has both Kaggle and local paths).

## Technologies

Python, Jupyter Notebook, pandas, NumPy, Matplotlib, seaborn, missingno, scikit-learn, XGBoost

## Acknowledgements

Based on the starter notebook by Akshay Sehgal for the upGrad & IIIT-B Machine Learning programme.
