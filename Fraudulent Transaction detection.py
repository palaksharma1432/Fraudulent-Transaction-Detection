#!/usr/bin/env python
# coding: utf-8

# # Fraudulent Transaction Detection

# # Tasks to be performed:
#     
# ## 1. Data cleaning including missing values, outliers and multi-collinearity.  
# 
# ## 2. Describe your fraud detection model in elaboration.
# 
# ## 3. How did you select variables to be included in the model?  
# 
# ## 4. Demonstrate the performance of the model by using best set of tools.
# 
# ## 5. What are the key factors that predict fraudulent customer? 
# 
# ## 6. Do these factors make sense? If yes, How? If not, How not?
# 
# ## 7. What kind of prevention should be adopted while company update its infrastructure?
# 
# ## 8. Assuming these actions have been implemented, how would you determine if they work? 
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# # Loading data:

# In[3]:


data = pd.read_csv(r"C:\Users\dell\Downloads\Fraud.csv")


# #  Initial Data Exploration 

# In[4]:


print("First 5 rows:")
print(data.head())
print("Data Info:")
data.info()
print("Descriptive Statistics:")
print(data.describe())
print("Target variable distribution (isFraud):")
print("IN PERCENTAGE:")
print(data['isFraud'].value_counts(normalize=True) * 100) 
print("IN ACTUAL NUMBERS:")
print(data['isFraud'].value_counts()) 


# 
# # Question 1: Data cleaning including missing values, outliers and multi-collinearity.
# 

# # 1.1 Missing Values
(we dont need to handle missing values (like by imputation or removal) because we don't have missing values
# In[5]:


missing_values = data.isnull().sum()
print(missing_values)
if missing_values.sum() == 0:
    print("No missing values found in the dataset.")
else:
    print("Missing values identified. Strategy needed (e.g., imputation, removal).")


# # 1.2 Outliers
(Here we are focusing on key numerical columns like 'amount', balances)
# In[7]:


num_cols_outliers = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
plt.figure(figsize=(15, 5 * len(num_cols_outliers) // 2))
for i, col in enumerate(num_cols_outliers):
    plt.subplot(len(num_cols_outliers) // 2 + 1, 2, i + 1)
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

print("Observations on Outliers:")
print("- 'amount' shows a significant number of high-value outliers.")
print("- Balance columns also exhibit outliers, often representing large sums.")

(Strategy: For tree-based models like RandomForest, outliers are generally less problematic than for linear models. We will keep them for now as extreme values might be indicative of fraud, but this should be noted.)

# # 1.3 Multi-collinearity

# In[8]:


numerical_cols = data.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(10, 8))
correlation_matrix = data[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

print("High correlation pairs which have absolute value > 0.8:")
high_corr = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates()
print(high_corr[ (abs(high_corr) > 0.8) & (abs(high_corr) < 1.0) ]) 

print("Observations on Multi-collinearity (Based on Correlation Matrix):")
print("- High correlation observed between 'oldbalanceOrg' and 'newbalanceOrig'.")
print("- High correlation observed between 'oldbalanceDest' and 'newbalanceDest'.")

(Strategy: High correlation suggests multicollinearity. While RandomForest can handle some correlation, severe cases can affect feature importance interpretation. We will remove one feature from each highly correlated pair based on the correlation matrix.)

(so now as a decision we will drop 'newbalanceOrig' and 'newbalanceDest' before modeling.)
# # Applying the decision: Drop columns

# In[9]:


data_cleaned = data.drop(['newbalanceOrig', 'newbalanceDest'], axis=1)
print(f"Dropped 'newbalanceOrig' and 'newbalanceDest'. \nNew shape: {data_cleaned.shape}")


# # 1.4 Additional cleaning steps based on data exploration

# In[10]:


print("Dropping 'nameOrig' and 'nameDest' due to high cardinality.")
data_cleaned = data_cleaned.drop(['nameOrig', 'nameDest'], axis=1)
print(f"Dropped 'nameOrig' and 'nameDest'. \nNew shape: {data_cleaned.shape}")


print("Encoding categorical feature 'type' using One-Hot Encoding.")
data_cleaned = pd.get_dummies(data_cleaned, columns=['type'], prefix='type', drop_first=True) 
print("Encoded 'type'. New shape and columns:")
print(data_cleaned.head())
print(data_cleaned.columns)


# # Question 2: Describe your fraud detection model in elaboration.
# 

# # Model Chosen: RandomForest Classifier
# ## Why RandomForest?
# 1. Strong Performance: It tends to perform well with a range of datasets with minimal hyperparameter tuning (though tuning can enhance performance).
# 2. Non-linear Modeling: It can reveal high-order, non-linear interactions amongst features and the fraud target variable.
# 3. Feature Importance: It has a feature to measure how important every feature is to fraud prediction.
# 4. Handling Class Imbalance (with settings): Settings like `class_weight='balanced'` can be applied in the model to correct the inherent class imbalance present in fraud detection.
# 5. Scalability: It is although high-resource-hungry on enormous datasets but parallel processing ability (`n_jobs=-1`). 
# 6. Lower Risk of Overfitting: Since it's making a mean of outcomes from multiple uncorrelated trees, there is less possibility for overfitting compared to a single tree decision. 
# ## Model Setup Overview: 
# - Input Features: Numerical and categorical selected features from sanitized data (after multicollinearity and high cardinality). 
# - Target Variable: `isFraud` (binary: 0 = non-fraudulent, 1 = fraudulent). 
# - Handling Imbalance: We will utilize the `class_weight='balanced'` parameter within the `RandomForestClassifier` to automatically adjust weights proportionally to class counts in the dataset (`n_samples / (n_classes * np.bincount(y))`). 
# - Data Split: The data is split into calibration (training) and validation (test) sets. Stratification will ensure the original proportion of fraud transactions remains in both sets.
# - Evaluation: Due to the high class imbalance, simple accuracy won't cut it. Our interest will be in:
# * Confusion Matrix: To quantify the trade-off between detecting fraud (True Positives) and falsely labeling legitimate transactions (False Positives).
# * Precision, Recall, F1-Score: Particularly for the fraud class (fraud=1). Recall (Sensitivity or True Positive Rate) is critical in fraud detection.

# # Preparing Data for Modeling

# In[11]:


X = data_cleaned.drop('isFraud', axis=1)
y = data_cleaned['isFraud']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print("Features columns:")
print(X.columns.tolist())


# # Spliting data into Training (Calibration) and Testing (Validation) sets

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")
print("Fraud proportion in Training set:")
print(y_train.value_counts(normalize=True))
print("Fraud proportion in Testing set:")
print(y_test.value_counts(normalize=True))


# # Feature Scaling
We scale only numerical features. (RandomForest is less sensitive to it, but it can sometimes help.)
# In[14]:


numerical_features = X.select_dtypes(include=np.number).columns.tolist()
ohe_features = X.filter(like='type_').columns.tolist() 
numerical_to_scale = [col for col in numerical_features if col not in ohe_features]

print(f"Numerical features to scale: {numerical_to_scale}")

Using StandardScaler
# In[16]:


scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_to_scale] = scaler.fit_transform(X_train[numerical_to_scale])
X_test_scaled[numerical_to_scale] = scaler.transform(X_test[numerical_to_scale]) 

print("Numerical features are scaled using StandardScaler.")
print("Scaled Training Data Head:")
print(X_train_scaled.head())


# # Train RandomForest Model
# 

# In[19]:


rf_model = RandomForestClassifier(
    n_estimators=100,       
    random_state=42,
    class_weight='balanced', 
    n_jobs=-1,               
    max_depth=None,          
    min_samples_split=2,     
    min_samples_leaf=1      
)

Training the model using scaled data
# In[20]:


print("Training the model... (This may take several minutes on the full dataset)")
rf_model.fit(X_train_scaled, y_train)
print("Model training completed.")


# # Question 3: How did you select variables to be included in the model?
# 

# # Variable selection involved several steps:
# 
# ## 1.  Initial Inclusion: 
# Started with all columns provided in the dataset.
# 
# ## 2.  Domain Understanding & Initial Assessment: 
# Reviewed the data dictionary and initial data exploration (`info()`, `describe()`, `head()`). Identified numerical, categorical, and potential identifier columns.
# 
# ## 3. Dealing with Many Unique Categorical Features: 
# Columns such as `nameOrig` and `nameDest` show account IDs. They have a large number of different values. Adding them directly (like one-hot encoding) would greatly increase the number of features and might cause overfitting or memory problems. Without advanced techniques (like frequency encoding or target encoding), they were considered too complex to include directly and were removed.
# 
# ## 4.  Categorical Feature Encoding: 
# The `type` column represents the transaction type. This is a nominal categorical feature with a limited number of unique values. One-Hot Encoding was used to convert these categories into numerical format suitable for the RandomForest model. `drop_first=True` was used to avoid perfect multicollinearity among the encoded columns.
# 
# ## 5. Checking Multicollinearity: 
# We looked at the correlation matrix for the numerical features. We found that `oldbalanceOrg`/`newbalanceOrig` and `oldbalanceDest`/`newbalanceDest` were highly correlated. To avoid problems with multicollinearity (especially when understanding the importance of features), we removed `newbalanceOrig` and `newbalanceDest`, keeping the 'old' balance features that show the state *before* the transaction.
# 
# ## 6. Considering Redundent Features: 
# The `isFlaggedFraud` column is marked by the system when a transfer is over 200k. It doesn't catch much actual fraud, as shown in early data analysis. It will be checked for its usefulness; if it seems unimportant, we might remove it later. The `step` column shows time units and is kept because time patterns can help identify fraud.
# ## 7.  Final Feature Set: 
# The resulting set of features used for modeling includes:
#     *   `step`
#     *   `amount`
#     *   `oldbalanceOrg`
#     *   `oldbalanceDest`
#     *   `isFlaggedFraud`
#     *   One-hot encoded `type` columns (e.g., `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`)
# 
# This process combines data-driven checks (cardinality) with practical considerations for modeling. Feature importance derived from the trained model (see Question 5) provides post-hoc validation of these choices.
# 

# 
# # Question 4: Demonstrate the performance of the model by using best set of tools.
# 

# # Making predictions on the test set using scaled test data

# In[21]:


y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1] 


# # 4.1 Confusion Matrix

# In[22]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud (0)', 'Fraud (1)'], yticklabels=['Not Fraud (0)', 'Fraud (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")
print("\nInterpretation:")
print(f"- The model correctly identified {tp} fraudulent transactions (TP).")
print(f"- It missed {fn} fraudulent transactions (FN).")
print(f"- It correctly identified {tn} non-fraudulent transactions (TN).")
print(f"- It incorrectly flagged {fp} non-fraudulent transactions as fraud (FP).")


# # 4.2 Classification Report

# In[23]:


print(classification_report(y_test, y_pred, target_names=['Not Fraud (0)', 'Fraud (1)']))
print("Interpretation:")
print("- Focus on the 'Fraud (1)' row.")
print("- Precision (Fraud=1): Of all transactions flagged as fraud, what proportion was actually fraud?")
print("- Recall (Fraud=1): Of all actual fraudulent transactions, what proportion did the model catch? (This is often critical)")
print("- F1-Score (Fraud=1): The harmonic mean of precision and recall for the fraud class.")
print("- Support: The actual number of instances for each class in the test set.")


# # 4.3 ROC AUC Score

# In[24]:


roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

Ploting ROC Curve
# In[26]:


fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity or Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

ROC AUC measures the model's ability to distinguish between fraud and non-fraud cases across all thresholds. A value closer to 1 is better.
# # Question 5: What are the key factors that predict fraudulent customer?
finding feature importance from trained dataset
# In[27]:


importances = rf_model.feature_importances_
feature_names = X_train_scaled.columns 

feature_importance_data = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_data = feature_importance_data.sort_values(by='Importance', ascending=False)

print("Feature Importances (Most important first):")
print(feature_importance_data)

Ploting feature importances
# In[29]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_data.head(10)) 
plt.title('Top 10 Feature Importances from RandomForest Model')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# In[30]:


print("Key Factors (based on importance scores):")
top_features = feature_importance_data['Feature'].head(5).tolist() 
print(f"- The most significant predictors of fraud appear to be: {', '.join(top_features)}")
print("- Other features also contribute, but these seem to have the highest predictive power according to the model.")


# # Question 6: Do these factors make sense? If yes, How? If not, How not?

# # Let's analyze the typical top factors often found in this dataset and see if they make sense:
# 
# Common Top Factors & Why They Make Sense:
# 
# ## 1.  `oldbalanceOrg` (Originator's Balance Before Transaction): 
# This is often very important. A big money transfer from an account that has no or very little money is very suspicious. Cheaters often take all the money from accounts fast or use hacked accounts that have certain balance features. This makes a lot of sense.
# 
# ## 2.  `amount` (Transaction Amount): 
# Fake transactions, especially big ones that try to empty accounts, usually have higher amounts than normal user activity. So, the transaction amount is an important sign. This also makes sense.
# 
# ## 3.  `type_TRANSFER` / `type_CASH_OUT` (Transaction Type): 
# The model probably sees some types of transactions as risky. `TRANSFER` (moving money to another account, often in a bad way) and `CASH_OUT` (taking out money) are common ways used in financial fraud to get money fast. Legal transactions are usually made up of `PAYMENT`, `CASH_IN`, or `DEBIT`. Therefore, it makes sense that the type of transaction is an important clue.
# 
# ## 4.  `step` (Time Step): 
# This represents a unit of time (e.g., hour). If fraud occurs in specific patterns related to time (e.g., late at night, specific intervals after account compromise), `step` can capture this. Velocity attacks (many transactions in a short period) could also be implicitly linked to `step`. This makes sense as fraudulent activities might not follow typical user time patterns.
# 
# ## 5.  `oldbalanceDest` (Destination Balance Before Transaction):
# Looking at the recipient account *before* the transaction can provide useful information. For instance, a new account or one that quickly gets a lot of money might be involved in fraud (like money laundering). This is important for tracking where money goes.
# 
# ## 6.  `isFlaggedFraud`: 
# This tool usually catches only a few transfers over 200k. When it does catch something (value=1), it may strongly relate to real fraud for those transactions. Even though its overall importance is low, it could still be useful for identifying a few frauds accurately. Because it has a narrow focus, its lower ranking makes sense.
# 
# So as conclusion we can say:
# Yes, the main factors found by the RandomForest model (like transaction amount, account balances before a transaction, certain risky transaction types, and timing patterns) clearly match common signs of financial fraud. They show how fraudsters illegally move and take money.

# # Question 7: What kind of prevention should be adopted while company update its infrastructure?

# ## When updating infrastructure, the company has an opportunity to embed fraud prevention measures more deeply. Based on the model's insights and general best practices, consider adopting:
# 
# ## 1.  Real-Time Transaction Monitoring & Scoring:
# 
#  a) Integrate the machine learning model (or a production-ready version) directly into the transaction processing flow.
#  
#  b) Score transactions in real-time based on the identified key factors (`amount`, `type`, balances, time patterns, etc.).
#     *   Flag high-risk transactions for review or automated blocking *before* completion.
# 
# ## 2.  Enhanced Authentication & Verification:
# 
#  a) Implement Adaptive Multi-Factor Authentication (MFA). Instead of static MFA for everyone, trigger stronger  
#     authentication (e.g., SMS code, biometric) based on the risk score of the transaction or context (new device, unusual
#     location, high-risk transaction type like `TRANSFER`).
#     
#  b) Step-up verification for high-amount `TRANSFER` and `CASH_OUT` transactions, especially if `oldbalanceOrg` is unusual or
#     the destination account is new/suspicious.
# 
# ## 3.  Behavioral Biometrics & Anomaly Detection:
# 
#  a) Go beyond static rules. Monitor user behavior patterns (typing speed, navigation, typical transaction 
#     times/amounts/locations).
#     
#  b) Use anomaly detection techniques to flag significant deviations from a user's established "normal" behavior, which could 
#     indicate account takeover.
# 
# ## 4.  Velocity Checks and Thresholds:
# 
#  a) Use dynamic speed checks: Keep track of how many and the total value of transactions in certain time periods (using  
#     `step` or real timestamps). Mark any unusual increases.
#     
#  b) Create flexible limits for transaction amounts that can be customized based on user groups or their past activity, 
#     particularly for `TRANSFER` and `CASH_OUT`.
# 
# ## 5.  Account & Device Profiling:
#  a) Build profiles for users including typical devices, locations, transaction patterns.
#  
#  b) Leverage device fingerprinting to detect suspicious devices or configurations often used by fraudsters.
# 
# ## 6.  Data Enrichment:
#    
#  a) Use outside data sources (when allowed and useful) like IP address trust levels, location information, and fraud lists 
#     to improve the model's features.
# 
# ## 7.  Focus on High-Risk Transaction Types:
# 
#  a) Make stricter rules and checks for `TRANSFER` and `CASH_OUT` transactions, as the model suggests. Maybe add delays or 
#     extra checks for large transfers to new recipients.
# 
# ## 8.  Feedback Loop & Model Retraining:
# 
#  a) Make sure the system can gather feedback on reported transactions, whether they were actual fraud or mistakes.
#  
#  b) Use this feedback to regularly retrain and update the fraud detection model to adapt to evolving fraud tactics.
# 
# ## 9.  Secure Infrastructure Design:
# 
#  a) Make sure the systems behind the model are safe by using secure coding, managing weaknesses, and controlling access to  
#     stop fraud.
# 

# # Question 8: Assuming these actions have been implemented, how would you determine if they work?

# ### To determine if the implemented prevention actions are effective, a combination of quantitative metrics and qualitative feedback is needed. Key approaches include:
# 
# ## 1.  Tracking Key Performance Indicators (KPIs):
# 
# ##### a) Fraud Rate: 
# Check the overall fraud rate (like the value of bad transactions divided by total transaction value, or the number of bad transactions divided by total transactions). Compare rates before and after changes. Look closely at rates for specific transaction types (`TRANSFER`, `CASH_OUT`). A big drop means success.
# 
# ##### b) Detection Rate (Model Recall/Sensitivity):
# How well is the new system finding fraud? Check the True Positive Rate (TP / (TP + FN)). This should ideally go up or stay high..
# 
# ##### c) False Positive Rate:
# How often are real transactions getting flagged or blocked? Check (FP / (FP + TN)). A big rise here may mean that good customers are facing too many obstacles, and adjustments may be needed.
# 
# ##### d) Precision of Alerts: 
# What percentage of the flagged transactions are real fraud? (True Positives / (True Positives + False Positives)). If this percentage goes up, it means the system is getting better.
# 
# ##### e) Average Fraud Loss per Incident:
# Has the financial impact of *successful* fraudulent transactions decreased?
# 
# ##### f) Alert Review Time: 
# If humans check the alerts, keep track of how long they take on each one. Improved efficiency might show good results.
# 
# ## 2.  A/B Testing (Champion-Challenger):
# 
# a) If feasible, roll out new rules/models to a portion of the traffic (Challenger group) while keeping the old system for another portion (Champion group).
# 
# b) Directly compare the KPIs (fraud rate, false positives, etc.) between the two groups over a defined period. This provides strong evidence of the new system's impact.
# 
# ## 3.  Cohort Analysis:
# 
# a) Compare how customers who joined *after* the new rules behave and their fraud experiences to those who joined before.
# 
# ## 4.  Monitoring Customer Impact:
# 
# a) Customer Friction:
# Track metrics related to user experience, such as:
# 
#    1.  Transaction abandonment rates (especially if new checks are added).
#         
#    2.  Customer support contacts related to blocked transactions or verification issues.
#         
#    3.  Customer satisfaction surveys (NPS, CSAT).
#         
# b) A successful system reduces fraud without putting too much pressure on honest customers.
# 
# ## 5.  Analysis of Fraud Types:
# 
# a) Are the types of fraud being caught changing? Are fraudsters shifting tactics in response to the new measures? This requires ongoing analysis of confirmed fraud cases.
# 
# ## 6.  Regular Model Performance Monitoring:
# 
# a) Continuously monitor the deployed model's performance metrics (AUC, Precision, Recall) on new, incoming data (out-of-time validation). Performance degradation might signal concept drift (fraud patterns changing) and the need for retraining.
# 
# By systematically tracking these metrics and comparing them to baseline or control groups, the company can quantitatively assess the effectiveness of the implemented fraud prevention strategies and make data-driven adjustments as needed.
# 

# # Section for External Prediction (User-Friendly: Options + Custom Input)
# 

# In[34]:


def predict_transaction(model, scaler, training_columns, numerical_to_scale):
    user_input = {}
    transaction_details = {} 

   
    step_options = {1: 10, 2: 150, 3: 400}
    type_options = {1: 'CASH_OUT', 2: 'PAYMENT', 3: 'CASH_IN', 4: 'TRANSFER', 5: 'DEBIT'}
    amount_options = {1: 50.00, 2: 5000.00, 3: 150000.00, 4: 300000.00}
    oldbalanceOrg_options = {1: 0.00, 2: 1000.00, 3: 50000.00, 4: 200000.00}
    oldbalanceDest_options = {1: 0.00, 2: 5000.00, 3: 100000.00, 4: 1000000.00}

    CUSTOM_KEY = 0 

    
    def get_user_selection(prompt, options_dict, feature_name, allow_custom=False, require_non_negative=False):
        print("-" * 40)
        print(prompt)
        print("-" * 40)

        
        for key, value in options_dict.items():
            display_value = f"{value:,.2f}" if isinstance(value, float) else value
            print(f"  [{key}] {display_value}")

       
        if allow_custom:
            print(f"  [{CUSTOM_KEY}] Enter a custom value for {feature_name}")

       
        while True:
            try:
                choice_str = input(f"Your choice for {feature_name}? (Enter number): ")
                choice = int(choice_str)

                if choice == CUSTOM_KEY and allow_custom:
                   
                    while True:
                        custom_value_str = input(f"  -> Enter your custom {feature_name}: ")
                        try:
                            custom_value = float(custom_value_str)
                            if require_non_negative and custom_value < 0:
                                print(f"  -> Sorry, the {feature_name} cannot be negative. Please try again.")
                            else:
                                print(f"  -> Custom {feature_name} entered: {custom_value:,.2f}")
                                return custom_value 
                        except ValueError:
                            print(f"  -> That doesn't look like a valid number for {feature_name}. Please try again.")

                elif choice in options_dict:
                    selected_value = options_dict[choice]
                    display_value = f"{selected_value:,.2f}" if isinstance(selected_value, float) else selected_value
                    print(f"  -> You selected: {display_value}")
                    return selected_value 
                else:
                    valid_keys = sorted(list(options_dict.keys()) + ([CUSTOM_KEY] if allow_custom else []))
                    print(f"  -> Hmm, that's not a valid option number. Please choose from {valid_keys}.")

            except ValueError:
                print("  -> Please enter only the number corresponding to your choice.")
            except KeyboardInterrupt:
                 print("\nOperation cancelled by user.")
                 return None 
   
   
    inputs_complete = False
    while not inputs_complete:
        transaction_details.clear() 

       
        step_val = get_user_selection("Step 1: Select the transaction time step", step_options, "Time Step")
        if step_val is None: return 
        user_input['step'] = step_val
        transaction_details['Time Step'] = step_val

       
        type_val = get_user_selection("Step 2: Select the type of transaction", type_options, "Transaction Type")
        if type_val is None: return 
        user_input['type'] = type_val
        transaction_details['Transaction Type'] = type_val

      
        amount_val = get_user_selection("Step 3: Select or enter the transaction amount", amount_options, "Amount", allow_custom=True, require_non_negative=True)
        if amount_val is None: return 
        user_input['amount'] = amount_val
        transaction_details['Amount'] = f"{amount_val:,.2f}"

       
        oldbalanceOrg_val = get_user_selection("Step 4: Select or enter the ORIGIN account balance BEFORE the transaction", oldbalanceOrg_options, "Origin Balance", allow_custom=True, require_non_negative=True)
        if oldbalanceOrg_val is None: return
        user_input['oldbalanceOrg'] = oldbalanceOrg_val
        transaction_details['Origin Balance (Before)'] = f"{oldbalanceOrg_val:,.2f}"

      
        oldbalanceDest_val = get_user_selection("Step 5: Select or enter the DESTINATION account balance BEFORE the transaction", oldbalanceDest_options, "Destination Balance", allow_custom=True, require_non_negative=True)
        if oldbalanceDest_val is None: return 
        user_input['oldbalanceDest'] = oldbalanceDest_val
        transaction_details['Destination Balance (Before)'] = f"{oldbalanceDest_val:,.2f}"

       
        print("\n" + "="*50)
        print("Please confirm the transaction details you entered:")
        print("="*50)
        for key, value in transaction_details.items():
            print(f"- {key}: {value}")
        print("="*50)

        while True:
            confirm = input("Is this correct? (yes/no/cancel): ").lower().strip()
            if confirm == 'yes':
                inputs_complete = True
                break
            elif confirm == 'no':
                print("\nOkay, let's enter the details again.")
                break 
            elif confirm == 'cancel':
                print("\nPrediction cancelled.")
                return 
            else:
                print("Please type 'yes', 'no', or 'cancel'.")
        

  
    if not inputs_complete:
         return 

    print("\nGreat! Processing the transaction details...")

   
    user_input['isFlaggedFraud'] = 1 if user_input['amount'] > 200000 else 0
    
  
    input_df = pd.DataFrame([user_input])

    
    type_prefix = 'type'
    chosen_type = user_input['type']
    input_df[f'{type_prefix}_{chosen_type}'] = 1
    input_df = input_df.drop('type', axis=1)

   
    for col in training_columns:
        if col.startswith(f'{type_prefix}_') and col not in input_df.columns:
            input_df[col] = 0

    
    try:
        input_df = input_df.reindex(columns=training_columns, fill_value=0)
    except Exception as e:
         print(f"\n[Error] Could not align input columns with model expectations: {e}")
         return

    
    input_df_scaled = input_df.copy()
    cols_to_scale_in_input = [col for col in numerical_to_scale if col in input_df_scaled.columns]
    if cols_to_scale_in_input:
         try:
             input_df_scaled[cols_to_scale_in_input] = scaler.transform(input_df[cols_to_scale_in_input])
          
         except Exception as e:
             print(f"\n[Error] Could not scale input data for the model: {e}")
             return


    try:
        prediction = model.predict(input_df_scaled)
        probability = model.predict_proba(input_df_scaled)

        print("\n" + "="*50)
        print("      --- Prediction Result ---")
        print("="*50)
        if prediction[0] == 1:
            print("🚨 Prediction: This transaction looks potentially FRAUDULENT.")
        else:
            print("✅ Prediction: This transaction looks likely NOT Fraudulent.")

        print("-" * 50)
        print(f"Confidence Score (Fraud Risk): {probability[0][1]:.2%}")
        print(f"(Score closer to 100% means higher fraud risk)")
        print("="*50)

    except Exception as e:
        print(f"\n[Error] Could not make a prediction: {e}")
        print("Please ensure the model and input data are compatible.")


# # Calling the function

# In[35]:


print("\n--- Let's Predict Fraud for a New Transaction ---")
print("For each step below, you can either choose a preset option")
print("or enter your own value where indicated.")

if 'rf_model' in locals() and 'scaler' in locals() and 'X_train_scaled' in locals() and 'numerical_to_scale' in locals():
     final_training_columns = X_train_scaled.columns.tolist()
     predict_transaction(rf_model, scaler, final_training_columns, numerical_to_scale)
else:
     print("\n[Setup Issue] Skipping prediction as the model or necessary data isn't ready.")
     print("Please ensure the entire script, including model training, has run successfully.")

print("\n--- End of External Prediction Section ---")


# In[ ]:





# In[ ]:




