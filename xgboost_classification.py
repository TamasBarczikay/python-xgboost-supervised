# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import optuna


# Load dataset and select relevant variables
file_path = r'C:\WORK\Projects\CSP\miami-housing.csv'  # Use raw string to handle backslashes
data = pd.read_csv(file_path)
data = data[['PARCELNO', 'SALE_PRC', 'age', 'structure_quality', 'CNTR_DIST', 'LND_SQFOOT', 'TOT_LVG_AREA']]

# Drop duplicates based on PARCELNO, keep the first occurrence
data = data.drop_duplicates(subset='PARCELNO', keep='first')

# Number of observations and data types
print(f'We have {data.shape[0]} observations after dropping duplicates.')

print("\nData Types:")
print(data.dtypes)

# Get baseline summary statistics
print("\nBaseline Summary Statistics:")
print(data.describe().round(1))

# Bar chart: Mean SALE_PRC by structure_quality (also count occurences)
structure_quality_counts = data['structure_quality'].value_counts().reset_index()
structure_quality_counts.columns = ['structure_quality', 'count']
structure_quality_counts = structure_quality_counts.sort_values(by='structure_quality')

print("\nOccurrences in structure_quality:")
print(structure_quality_counts)

mean_prices_by_quality = data.groupby('structure_quality')['SALE_PRC'].mean().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(
    x='structure_quality',
    y='SALE_PRC',
    data=mean_prices_by_quality,
    palette='viridis'
)
plt.title('Mean Sale Price by Structure Quality')
plt.xlabel('Structure Quality')
plt.ylabel('Mean Sale Price ($)')
plt.grid(True)
plt.show()

# Create a new binary variable for superior quality
data['superior_quality'] = (data['structure_quality'] == 5).astype(int)



## XGBOOST - classification
# Drop the original structure_quality column from features
X = data.drop(columns=['superior_quality', 'structure_quality', 'PARCELNO'])  # Exclude target, categorical column, and ID
y = data['superior_quality']  # Binary target variable

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize XGBoost for classification
xgboost_classifier = xgb.XGBClassifier(
    objective='binary:logistic',  # Logistic regression objective
    n_estimators=100,             # Number of trees
    max_depth=6,                  # Tree depth
    learning_rate=0.1,            # Step size shrinkage
    random_state=42               # Reproducibility
)

# Train the classifier
xgboost_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = xgboost_classifier.predict(X_test)



# Evaluate model performance
accuracy = (y_test == y_pred).mean()
precision = precision_score(y_test, y_pred, zero_division=0)  # Avoid division by zero warnings
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Print metrics
print(f"Baseline Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(xgboost_classifier, X_test, y_test)
plt.title('Confusion Matrix')
plt.show()




## XGBOOST BRUTAL
# Define the objective function for Optuna
def classification_objective(trial):
    # Define parameter grid
    param = {
        'objective': 'binary:logistic',  # Classification objective
        'eval_metric': 'logloss',       # Log loss for binary classification
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300]),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2]),
        'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8]),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
    }

    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(**param)

    # Train the model
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Predict probabilities and evaluate with accuracy
    preds = model.predict(X_test)
    accuracy = (y_test == preds).mean()  # Accuracy as the optimization target
    return accuracy

# Run Optuna optimization to maximize accuracy
classification_study = optuna.create_study(direction='maximize')
classification_study.optimize(classification_objective, n_trials=50)

# Print the best hyperparameters
print("\nBest Hyperparameters:")
print(classification_study.best_params)

# Train the final model with optimized hyperparameters
optimized_classification_params = classification_study.best_params
optimized_classification_params['objective'] = 'binary:logistic'
optimized_classification_params['eval_metric'] = 'logloss'

final_classifier = xgb.XGBClassifier(**optimized_classification_params)
final_classifier.fit(X_train, y_train)

# Import necessary metric functions
# Evaluate the final model
y_pred = final_classifier.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Print metrics
print(f"\nFinal Model Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(final_classifier, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

# Feature importance plot
xgb.plot_importance(final_classifier, importance_type='weight', max_num_features=10, height=0.4)
plt.title('Feature Importance (Classification)')
plt.show()

# Save optimization results as HTML
from optuna.visualization import plot_optimization_history, plot_param_importances
opt_hist = plot_optimization_history(classification_study)
opt_hist.write_html("classification_optimization_history.html")
print("Optimization history plot saved as 'classification_optimization_history.html'")

param_imp = plot_param_importances(classification_study)
param_imp.write_html("classification_param_importances.html")
print("Parameter importances plot saved as 'classification_param_importances.html'")




# Ensure the final model is assigned to the variable `model`
model = final_classifier  # Assign the optimized classifier to `model`

# Generate probabilities using the logistic regression model
probabilities = model.predict_proba(X_test)[:, 1]  # Extract probabilities for the positive class

# Combine SALE_PRC and probabilities into a DataFrame for plotting
plot_data = pd.DataFrame({
    'SALE_PRC': X_test['SALE_PRC'],  # Ensure SALE_PRC is in the test set
    'Probability': probabilities
})

# Sort by SALE_PRC for a smooth curve
plot_data = plot_data.sort_values(by='SALE_PRC')

# Plot the S-shaped curve
plt.figure(figsize=(8, 6))
sns.lineplot(x=plot_data['SALE_PRC'], y=plot_data['Probability'], color='blue', linewidth=2)
plt.title('Probability of Superior Quality vs Sale Price')
plt.xlabel('Sale Price ($)')
plt.ylabel('Probability of Superior Quality')
plt.grid(True)
plt.show()






'''
## XGBoost Binary Classification: Parameter Explanation

The following parameter grid is used for **XGBoost** hyperparameter tuning in a **binary classification task**. Each parameter has a specific role in controlling the model's behavior, particularly for optimizing predictive accuracy and generalization.

```python
param = {
    'objective': 'binary:logistic',  # Classification objective
    'eval_metric': 'logloss',       # Log loss for binary classification
    'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300]),
    'max_depth': trial.suggest_int('max_depth', 3, 7),
    'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2]),
    'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8]),
    'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
    'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
}


# Parameter Descriptions

## 1. `objective: 'binary:logistic'`
- **Purpose**: Sets the model's task as binary classification.
- **Output**: Probabilities for the two classes in the range [0, 1].
- **Usage**: A threshold (e.g., 0.5) is applied to convert probabilities into binary predictions.

## 2. `eval_metric: 'logloss'`
- **Purpose**: Specifies the evaluation metric as logarithmic loss.
- **Measurement**: Measures the model's predictive accuracy for probabilities.
- **Goal**: Lower values indicate better performance, particularly suitable for classification problems with probabilistic outputs.

## 3. `n_estimators`
- **Purpose**: Specifies the number of boosting rounds or trees.
- **Values**: `[100, 200, 300]`.
- **Effect**: A higher number of estimators reduces bias but may increase overfitting if not regularized.

## 4. `max_depth`
- **Purpose**: Defines the maximum depth of each decision tree.
- **Values**: `[3, 4, 5, 6, 7]`.
- **Effect**: Deeper trees capture more complex patterns but may overfit on small datasets.

## 5. `learning_rate`
- **Purpose**: Determines the step size for updating weights in each boosting round.
- **Values**: `[0.01, 0.05, 0.1, 0.2]`.
- **Effect**: Smaller values slow down learning, requiring more trees, but often lead to better generalization.

## 6. `subsample`
- **Purpose**: Controls the fraction of training data sampled for each tree.
- **Values**: `[0.6, 0.7, 0.8]`.
- **Effect**: Smaller values introduce more randomness, helping to prevent overfitting.

## 7. `reg_alpha`
- **Purpose**: L1 regularization term, penalizing the absolute values of leaf weights.
- **Values**: Continuous range `[0.01, 10.0]` (log-scaled).
- **Effect**: Encourages sparsity in the model, reducing the number of features used.

## 8. `reg_lambda`
- **Purpose**: L2 regularization term, penalizing the squared values of leaf weights.
- **Values**: Continuous range `[1.0, 10.0]`.
- **Effect**: Helps smooth the model and reduce overfitting by shrinking large weights.

'''
