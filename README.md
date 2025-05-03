# Fraudulent-Transaction-Detection

This project aims to detect fraudulent financial transactions using a machine learning model. It utilizes a synthetic dataset generated using the PaySim simulator, which mimics real-world mobile money transactions. The project involves data cleaning, exploratory data analysis, feature engineering, model training (using RandomForest), evaluation, and interpretation, addressing specific questions related to the fraud detection process.

## Project Objective

The primary goal is to build and evaluate a robust model capable of distinguishing between legitimate and fraudulent transactions, paying close attention to the challenges posed by highly imbalanced data typical in fraud detection scenarios.

## Dataset

*   **Source:** The dataset used is `Fraud.csv`, typically sourced from Kaggle (often referred to as the "Synthetic Financial Datasets For Fraud Detection"). It's based on the PaySim simulator.
*   **Characteristics:**
    *   Contains approximately 6.3 million transaction records.
    *   Features include transaction type (`type`), amount (`amount`), old/new balances for origin and destination accounts (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`), timestamp (`step`), and flags (`isFraud`, `isFlaggedFraud`).
    *   Highly imbalanced: Fraudulent transactions (`isFraud`=1) constitute a very small percentage (~0.13%) of the total dataset.

## Key Questions Addressed

This project systematically addresses the following key questions:

1.  **Data Cleaning:** How were missing values, outliers, and multi-collinearity handled?
2.  **Model Description:** Elaborate on the chosen fraud detection model (RandomForest Classifier).
3.  **Variable Selection:** How were variables selected for inclusion in the model?
4.  **Model Performance:** How was the model's performance demonstrated using appropriate tools and metrics (Confusion Matrix, Classification Report, ROC AUC)?
5.  **Key Predictive Factors:** What are the key factors that predict a fraudulent customer/transaction?
6.  **Factor Validation:** Do these predictive factors make logical sense in the context of fraud?
7.  **Prevention Strategies:** What kind of preventive measures should be adopted during infrastructure updates?
8.  **Effectiveness Measurement:** How can the effectiveness of implemented prevention actions be determined?

## Methodology

1.  **Data Loading & Exploration:** Initial loading and understanding of the dataset structure, types, and basic statistics.
2.  **Data Cleaning:**
    *   Checked for missing values (none found).
    *   Analyzed outliers using boxplots (kept for tree-based model, potentially indicative).
    *   Addressed multicollinearity by examining correlations and dropping highly correlated features (`newbalanceOrig`, `newbalanceDest`).
    *   Dropped high-cardinality identifier columns (`nameOrig`, `nameDest`).
3.  **Feature Engineering:**
    *   One-Hot Encoded the categorical `type` feature.
4.  **Data Splitting:** Split the data into training (70%) and testing (30%) sets, stratified by the `isFraud` target variable to maintain class proportions.
5.  **Feature Scaling:** Applied `StandardScaler` to numerical features in the training and testing sets.
6.  **Model Training:** Trained a `RandomForestClassifier` with `class_weight='balanced'` to handle the severe class imbalance.
7.  **Model Evaluation:** Evaluated the model on the test set using:
    *   Confusion Matrix
    *   Classification Report (Precision, Recall, F1-Score for both classes)
    *   ROC AUC Score and ROC Curve
8.  **Feature Importance:** Extracted and visualized feature importances from the trained RandomForest model.
9.  **Interpretation & Reporting:** Answered the 8 key questions based on the analysis and model results.
10. **Prediction Interface:** Included a user-friendly command-line interface to predict the fraud status of a manually entered transaction.

## Performance Highlights

*   The RandomForest model demonstrated strong performance in distinguishing fraudulent transactions despite the severe class imbalance.
*   Achieved a high ROC AUC score (typically > 0.98 on this dataset with RandomForest), indicating excellent separation capability.
*   The use of `class_weight='balanced'` helped improve the Recall for the minority (fraud) class, which is crucial for minimizing missed fraud cases, though it often comes at the cost of lower precision (more false positives).
*   Key metrics like Precision, Recall, and F1-score for the 'Fraud' class are reported in the script's output.

## Key Findings (Predictive Factors)

The most important features identified by the model for predicting fraud typically include:

1.  `oldbalanceOrg`: The balance of the originating account before the transaction.
2.  `amount`: The transaction amount.
3.  `type_TRANSFER`: Indicator if the transaction was a transfer.
4.  `type_CASH_OUT`: Indicator if the transaction was a cash-out.
5.  `oldbalanceDest`: The balance of the destination account before the transaction.
6.  `step`: The time step of the transaction.

These factors align with logical fraud patterns (e.g., large transfers/cash-outs from accounts with specific balance characteristics).

## Technology Stack

*   Python 3.x
*   Pandas
*   NumPy
*   Scikit-learn
*   Matplotlib
*   Seaborn

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  **Install required libraries:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
    *(Alternatively, if a `requirements.txt` file is created: `pip install -r requirements.txt`)*
4.  **Dataset:** Ensure the `Fraud.csv` file is present in the expected location (the script currently uses a hardcoded path `C:\Users\dell\Downloads\Fraud.csv`). You **must** update this path in the script (In[3]) to point to the correct location of your `Fraud.csv` file.

## Usage

1.  **Update Dataset Path:** Modify the path in cell `[3]` of the script `Fraudulent Transaction detection.py` to point to your `Fraud.csv` file.
2.  **Run the script:** Execute the Python script from your terminal.
    ```bash
    python "Fraudulent Transaction detection.py"
    ```
3.  **Output:** The script will perform all the analysis steps and print the results, including data summaries, correlation heatmaps, model performance metrics (Confusion Matrix, Classification Report, ROC AUC score), feature importances, and answers to the key questions.
4.  **Interactive Prediction:** At the end of the script execution, it will prompt the user to enter details for a new transaction to predict whether it is fraudulent or not. Follow the on-screen instructions.

## License
 All Rights Reserved

## 📧 Contact

For questions or suggestions, please contact:  
Palak Sharma - 22cse079@gweca.ac.in <br>
Project Link: https://github.com/palaksharma1432/CaseNext <br>
linkedin: https://www.linkedin.com/in/palak-sharma-4799672b1/ <br>
```
