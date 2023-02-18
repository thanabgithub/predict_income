import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, Normalizer, QuantileTransformer

import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns


col_names = ['age', 'workclass', 'fnlwgt','education', 'education-num', 
'marital-status', 'occupation', 'relationship', 'race', 'sex',
'capital-gain','capital-loss', 'hours-per-week','native-country', 'income']
df = pd.read_csv('adult.data',header = None, names = col_names)
# print(df.head())
#Clean columns by stripping extra whitespace for columns of type "object"
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].str.strip()
print(df.head())

#1. Check Class Imbalance
print(df["income"].value_counts()/df.shape[0])



# Identify categorical variables
cat_vars = df.select_dtypes(include=['object']).columns

# Convert categorical variables to dummy variables
dummy_vars = pd.get_dummies(df[cat_vars], drop_first=True)

# Combine dummy variables with original data
df = pd.concat([df, dummy_vars], axis=1)

# Drop original categorical variables
df.drop(cat_vars, axis=1, inplace=True)
print(df.head())
df_cor = df.corr()
# Define the custom color map for the heatmap
colors = sns.color_palette(["#D43D3D", "white", "#4878D0"])
cmap = sns.blend_palette(colors, as_cmap=True)

sns.heatmap(df_cor, cmap=cmap, center=0)
plt.show()

#2. Create feature dataframe X with feature columns and dummy variables for categorical features
num_feature_cols = ['age','capital-gain', 'capital-loss', 'hours-per-week', 'hours-per-week']


sex_cols = df.filter(like="sex").columns.tolist()
education_cols = df.filter(like="education").columns.tolist()
race_cols = df.filter(like="education").columns.tolist()

feature_cols = num_feature_cols + sex_cols + education_cols + race_cols

#4. Create output variable y which is binary, 0 when income is less than 50k, 1 when it is greather than 50k



#5a. Split data into a train and test set

X = df[feature_cols]
y = df["income_>50K"]
# Normalizer, QuantileTransformer
scaler = QuantileTransformer()

# fit and transform the data
X[num_feature_cols] = scaler.fit_transform(X[num_feature_cols])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)

#5b. Fit LR model with sklearn on train set, and predicting on the test set
log_reg = LogisticRegression(C=0.05, penalty='l1', solver='liblinear')

log_reg.fit(X_train, y_train)

#6. Print model parameters and coefficients
print(f'Model Parameters, Intercept: {log_reg.intercept_}')

print(f'Model Parameters, Coeff: {log_reg.coef_}')

print(classification_report(y_test, log_reg.predict(X_test)))
#7. Evaluate the predictions of the model on the test set. Print the confusion matrix and accuracy score.

tn, fp, fn, tp = confusion_matrix(y_test, log_reg.predict(X_test)).ravel()

# Print confusion matrix
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# Compute confusion matrix
cm = confusion_matrix(y_test, log_reg.predict(X_test))

# Convert confusion matrix to numpy array
cm_df = pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])

accuracy = accuracy_score(y_test, log_reg.predict(X_test))

print('Confusion Matrix on test set: ', cm_df)
print('Accuracy Score on test set: ', accuracy)

# 8.Create new DataFrame of the model coefficients and variable names; sort values based on coefficient
# create dataframe of coefficients and variable names
coefs = pd.DataFrame({'Variable': ['Intercept'] + feature_cols,
                      'Coef': [log_reg.intercept_[0]] + list(log_reg.coef_[0])})

# sort dataframe by coefficient
coefs = coefs.sort_values(by='Coef', ascending=False)
print(coefs)
#9. barplot of the coefficients sorted in ascending order
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(coefs['Variable'], coefs['Coef'], color='b')
ax.set_xlabel('Coefficient')
ax.set_ylabel('Variable')
ax.set_title('Logistic Regression Coefficients')
plt.show()
#10. Plot the ROC curve and print the AUC value.
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]

# calculate the AUC score
auc = roc_auc_score(y_test, y_pred_prob)
print(auc)
# calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# plot the ROC curve
fig, ax = plt.subplots()

ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC={:.2f})'.format(auc))
plt.show()
