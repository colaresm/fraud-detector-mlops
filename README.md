# Fraud-detector-mlops

```
docker run -p 5000:5000 mlflow-server
docker run -p 5000:5000 mlflow/mlflow
```

# Dataset

This dataset was synthetically generated to simulate a loan risk classification scenario. The target variable (`loan_risk`) has three categories: `low`, `moderate`, and `high`.

The explanatory variables are:

- `monthly_income` (float): applicant's monthly income in Brazilian reais, generated from a normal distribution with a mean of 5000 and a standard deviation of 1500, with values truncated below 800. Example generation:  
  `monthly_income = np.random.normal(loc=5000, scale=1500, size=n_samples).clip(800, None)`

- `credit_score` (int): applicant's credit score, ranging from 300 to 1000, generated using uniformly random integers. Example:  
  `credit_score = np.random.randint(300, 1001, size=n_samples)`

- `current_debt` (float): total current debts of the client in Brazilian reais, generated from a normal distribution with a mean of 10000 and a standard deviation of 5000, with negative values truncated to zero. Example:  
  `current_debt = np.random.normal(loc=10000, scale=5000, size=n_samples).clip(0, None)`

- `late_payments` (int): number of late payments in the last 12 months, generated as integers between 0 and 10. Example:  
  `late_payments = np.random.randint(0, 11, size=n_samples)`

The target variable `loan_risk` is calculated using a weighted rule based on the variables above, where the risk increases with higher debt relative to income, lower credit score, and more late payments. The formula used is:

```
risk_score = (debt / income) * 0.4 + (10 - credit_score / 1000 * 10) * 0.3 + late_payments * 0.3
```


With the classification:

- If `risk_score < 4`, risk = "low"  
- If `risk_score < 7`, risk = "moderate"  
- Otherwise, risk = "high"  

This rule reflects that risk is higher when debt is high relative to income, the credit score is low, and there are many previous late payments.
