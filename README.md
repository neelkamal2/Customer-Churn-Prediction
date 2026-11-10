# Customer Churn Prediction Using Machine Learning

A machine learning project that predicts whether a telecommunications
customer is likely to leave the service (**Churn = Yes**) or remain
(**Churn = No**). Five classification algorithms are trained and
compared using the same train/test split.

## Project Overview

Customer churn is an important business problem for telecom companies.
Identifying customers who may leave can help a company understand
customer behavior and plan retention activities.

This project uses the Telco Customer Churn dataset and compares five
machine learning models:

1.  Logistic Regression
2.  Decision Tree
3.  Random Forest
4.  K-Nearest Neighbors (KNN)
5.  Support Vector Machine (SVM)

The project is implemented in Python and was run in Spyder using Python
3.13.9 (Anaconda).

## Dataset

-   **Dataset file:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
-   **Rows:** 7,043 customers
-   **Columns in the original dataset:** 21
-   **Target column:** `Churn`
-   **Target values:** `No` and `Yes`

The original columns include customer information, services subscribed
to, contract and payment details, tenure, and charges.

### Dataset preparation

The supplied run output reports:

-   Original dataset shape: **7,043 rows × 21 columns**
-   Duplicate rows: **0**
-   No null values were reported in the initial null-value check.
-   `customerID` was removed because it is an identifier rather than a
    model input.
-   `TotalCharges` was converted to numeric. Values that could not be
    converted were treated as missing and filled with `0`.
-   `Churn` was encoded as `No = 0` and `Yes = 1`.
-   After removing `customerID`, the dataset contained **7,043 rows × 20
    columns**.
-   Categorical input columns were converted into numeric indicator
    columns using one-hot encoding (`pd.get_dummies` with
    `drop_first=True`).
-   The resulting model input contained **30 features**.

## Exploratory Analysis and Visualizations

The project screenshots are stored in the `images/` folder. The plots
show:

-   The distribution of customers who churned and those who did not.
-   Customer tenure distribution by churn status.
-   Churn patterns by contract type and payment method.
-   Churn counts by gender and internet service.
-   Monthly charges compared with churn status.
-   Accuracy and F1-score comparisons across the five models.
-   Confusion matrices for each model.
-   ROC curve comparison.

### Observations from the supplied charts

-   **Contract type:** Month-to-month customers show a much higher churn
    percentage (about 43%) than one-year customers (about 11%) and
    two-year customers (about 3%).
-   **Payment method:** Electronic-check customers have the highest
    churn percentage in the chart (about 45%); the other payment methods
    are substantially lower.
-   **Internet service:** The fiber-optic group shows a notably larger
    number of churned customers than the DSL and no-internet groups in
    the count chart.
-   **Tenure:** Churn is concentrated more heavily among customers with
    shorter tenure, while longer-tenure customers are more frequently
    represented among non-churners.
-   **Monthly charges:** The median monthly charge appears higher for
    customers who churned than for customers who stayed.
-   **Gender:** The gender-wise count chart shows broadly similar churn
    counts for female and male customers.

These are descriptive patterns in the supplied charts; they do not by
themselves establish that any one factor causes churn.

## Preprocessing and Model Training

The data was split into training and testing sets using an **80:20
split**, `random_state=42`, and stratification by the target:

  Split              Rows
  --------------- -------
  Training data     5,634
  Testing data      1,409

`StandardScaler` was fitted on the training data and used to scale the
training and testing features for Logistic Regression, KNN, and SVM.
Decision Tree and Random Forest were trained using the unscaled one-hot
encoded features.

The models use the settings defined in the Python script, including
class weighting for Logistic Regression, Decision Tree, Random Forest,
and SVM.

## Model Results

The following results are from the supplied program output on the
1,409-row test set. Precision, recall, and F1-score refer to the
positive class (`Churn = 1`).

  -----------------------------------------------------------------------------
  Model            Accuracy    Precision       Recall     F1-score          AUC
  ------------ ------------ ------------ ------------ ------------ ------------
  Logistic           0.7402       0.5069   **0.7861**       0.6164   **0.8414**
  Regression                                                       

  Decision           0.7246       0.4882       0.7754       0.5992       0.8279
  Tree                                                             

  Random         **0.7637**   **0.5409**       0.7246   **0.6194**       0.8400
  Forest                                                           

  KNN                0.7473       0.5253       0.5000       0.5123       0.7718

  SVM                0.7473       0.5162       0.7674       0.6172       0.8185
  -----------------------------------------------------------------------------

### Result interpretation

-   **Random Forest is the selected best model by F1-score**, with an
    F1-score of **0.6194**. It also has the highest accuracy
    (**0.7637**) and precision (**0.5409**) among the five models.
-   **Logistic Regression has the highest churn recall (0.7861)** and
    the highest AUC (**0.8414**). It identifies the largest proportion
    of actual churners in this comparison, although its precision is
    lower than Random Forest's.
-   **SVM** also has relatively high churn recall (**0.7674**) and an
    F1-score of **0.6172**.
-   **KNN** has the lowest churn recall (**0.5000**) and F1-score
    (**0.5123**) in this experiment.
-   The model choice depends on the business objective. If missing a
    customer who is likely to churn is especially costly, recall may be
    important; if balancing precision and recall is the goal, the
    F1-score is useful.

The script selects the model with the highest F1-score, which is Random
Forest for the reported run.

## Visual Results

The paths below assume the screenshots are uploaded into a folder named
`images` at the repository root. Keep the image filenames exactly as
shown, or update the paths if you rename them.

### Model performance

![Accuracy comparison](images/Accuracy%20Comparison.png)

![F1-score comparison](images/F1%20Score%20Comparison%20of%20Models.png)

![ROC curve comparison](images/Roc%20Curve%20Comparison.png)

### Customer churn analysis

![Customer churn
distribution](images/Customer%20Churn%20Distribution.png)

![Customer tenure
distribution](images/Customer-Tenure%20Distribution.png)

![Churn by contract
type](images/Churn%20Percentage%20by%20Contract%20Type.png)

![Churn by payment
method](images/Churn%20Percentage%20by%20Payment%20Method.png)

![Gender-wise customer churn](images/Gender-wise%20Customer%20Churn.png)

![Internet service-wise customer
churn](images/Internet%20Service-wise%20Customer%20Churn.png)

![Monthly charges versus
churn](images/Monthly%20Charges%20vs%20Churn.png)

### Confusion matrices

![Logistic Regression confusion
matrix](images/Logistic%20Regression%20Confusion%20Matrix.png)

![Decision Tree confusion
matrix](images/Decision%20Tree%20Confusion%20Matrix.png)

![Random Forest confusion
matrix](images/Random%20Forest%20Confusion%20Matrix.png)

![KNN confusion matrix](images/KNN%20Confusion%20Matrix.png)

![SVM confusion matrix](images/SVM%20Confusion%20Matrix.png)

## Technologies and Libraries

-   Python 3.13.9
-   Pandas
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Spyder IDE (Anaconda)

## Repository Structure

Arrange the repository like this:

``` text
Customer-Churn-Prediction/
│
├── customer_churn_prediction1.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
├── README.md
└── images/
    ├── Accuracy Comparison.png
    ├── Churn Percentage by Contract Type.png
    ├── Churn Percentage by Payment Method.png
    ├── Customer Churn Distribution.png
    ├── Customer-Tenure Distribution.png
    ├── Decision Tree Confusion Matrix.png
    ├── F1 Score Comparison of Models.png
    ├── Gender-wise Customer Churn.png
    ├── Internet Service-wise Customer Churn.png
    ├── KNN Confusion Matrix.png
    ├── Logistic Regression Confusion Matrix.png
    ├── Monthly Charges vs Churn.png
    ├── Random Forest Confusion Matrix.png
    ├── Roc Curve Comparison.png
    └── SVM Confusion Matrix.png
```

## How to Run the Project

1.  Download or clone this repository.

2.  Install the required libraries:

    ``` bash
    pip install -r requirements.txt
    ```

3.  Keep the CSV dataset in the same folder as the Python script.

4.  Open `customer_churn_prediction1.py` in Spyder.

5.  **Update the dataset path in the script** to match your computer.
    For example, if the CSV is in the same folder as the script, use:

    ``` python
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    ```

6.  Run the Python file. The program prints the evaluation metrics and
    displays the comparison charts and confusion matrices.

## Notes

-   Results can vary if the dataset, preprocessing, model settings, or
    train/test split are changed.
-   The reported metrics are from this particular test split; they are
    not a guarantee of performance on future customers.
-   Before keeping the dataset in a public repository, check that its
    terms permit redistribution. If redistribution is not permitted,
    remove the CSV from the public repository and provide instructions
    for obtaining it from its authorized source instead.

## Author

**Neelkamal**
