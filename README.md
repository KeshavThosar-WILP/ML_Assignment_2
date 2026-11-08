# ML Assignment 2

## Problem Statement
Analyze customer behavior and predict if they are more likely to drop their subscriptions (churn) or not.

## Dataset Description
**Dataset:** Telecom Customer Churn  
**Link:** https://www.kaggle.com/datasets/alipakzad/telecom-customer-churn  
**Description:** (As mentioned in the Data Card of the Kaggle page)  

The dataset contains information about customers and their churn status.
| Column Name | Description |
|---|---|
| customerID | Unique identifier for each customer. |
| gender | Gender of the customer (Male, Female). |
| SeniorCitizen | Whether the customer is a senior citizen or not (1: Yes, 0: No). |
| Partner | Whether the customer has a partner or not (Yes, No). |
| Dependents | Whether the customer has dependents or not (Yes, No). |
| tenure | Number of months the customer has stayed with the company. |
| PhoneService | Whether the customer has a phone service or not (Yes, No). |
| MultipleLines | Whether the customer has multiple lines or not (Yes, No, No phone service). |
| InternetService | Type of internet service the customer has (DSL, Fiber optic, No). |
| OnlineSecurity | Whether the customer has online security or not (Yes, No, No internet service). |
| OnlineBackup | Whether the customer has online backup or not (Yes, No, No internet service). |
| DeviceProtection | Whether the customer has device protection or not (Yes, No, No internet service). |
| TechSupport | Whether the customer has tech support or not (Yes, No, No internet service). |
| StreamingTV | Whether the customer has streaming TV or not (Yes, No, No internet service). |
| StreamingMovies | Whether the customer has streaming movies or not (Yes, No, No internet service). |
| Contract | The contract term of the customer (Month-to-month, One year, Two year). |
| PaperlessBilling | Whether the customer has paperless billing or not (Yes, No). |
| PaymentMethod | The payment method of the customer (Electronic check, Mailed check, Bank transfer, Credit card). |
| MonthlyCharges | The amount charged to the customer monthly. |
| TotalCharges | The total amount charged to the customer. |
| **Churn (Target)** | Whether the customer churned or not (Yes, No). |


## Github Repository Link
https://github.com/KeshavThosar-WILP/ML_Assignment_2

## Models Used
### Evaluation Metrics
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.735983 | **0.838950** | 0.501678 | 0.799465 | 0.616495 | 0.458077 |
| Decision Tree | 0.716111 | 0.664783 | 0.470455 | 0.553476 | 0.508600 | 0.312868 |
| KNN | 0.700497 | 0.764405 | 0.459459 | 0.727273 | 0.563147 | 0.374031 |
| Naive Bayes | 0.652236 | 0.789194 | 0.421622 | **0.834225** | 0.560144 | 0.372004 |
| Support Vector Machine | **0.757275** | 0.824509 | **0.530189** | 0.751337 | **0.621681** | **0.465573** |

### Observations
| ML Model Name | Observation about model |
|---|---|
| Logistic Regression | Highest AUC score. Good model to determine decision boundaries |
| Decision Tree | Slightly higher precision than KNN but low consistency (lower F1 score as compared to KNN) |
| KNN | High recall. Can be used over Decision tree if goal is to minimize missed churn rates |
| Naive Bayes | Highest recall. Detected most churners but at the cost of high false positive rate |
| Support Vector Machine | Overall best performing model across most evaluation metrics in determining customer churn |
