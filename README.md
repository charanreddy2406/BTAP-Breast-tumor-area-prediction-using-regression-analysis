# BTAP-Breast-tumor-area-prediction-using-regression-analysis
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load the Wisconsin breast cancer dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Select key features for analysis
selected_features = [
    'mean radius', 'mean texture', 'mean perimeter', 
    'mean area', 'mean smoothness', 'mean compactness'
]

# Create target variable: Using mean area as the target for regression
X = df[selected_features].drop('mean area', axis=1)
y = df['mean area']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=0.1),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}
predictions = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                              cv=5, scoring='r2')
    
    results[name] = {
        'MSE': mse,
        'R2 Score': r2,
        'CV R2 Mean': cv_scores.mean(),
        'CV R2 Std': cv_scores.std()
    }

# Create visualization functions
def plot_model_comparison():
    results_df = pd.DataFrame(results).T
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=results_df.index, y=results_df['R2 Score'])
    plt.title('Model Performance Comparison (R² Score)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def plot_residuals(model_name):
    residuals = y_test - predictions[model_name]
    
    plt.figure(figsize=(12, 4))
    
    # Residual plot
    plt.subplot(121)
    sns.scatterplot(x=predictions[model_name], y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Residual Plot - {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    
    # QQ plot
    plt.subplot(122)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot - {model_name}')
    
    plt.tight_layout()
    return plt

def plot_feature_importance(model_name):
    if model_name == 'Random Forest':
        importance = models[model_name].feature_importances_
    else:
        importance = models[model_name].coef_
    
    feat_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': abs(importance)
    })
    feat_importance = feat_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_importance)
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    return plt

# Generate evaluation report
def print_evaluation_report():
    print("Model Evaluation Report")
    print("=" * 50)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"MSE: {metrics['MSE']:.4f}")
        print(f"R² Score: {metrics['R2 Score']:.4f}")
        print(f"Cross-validation R² (mean): {metrics['CV R2 Mean']:.4f}")
        print(f"Cross-validation R² (std): {metrics['CV R2 Std']:.4f}")

# Example usage of visualization functions
plot_model_comparison()
plot_residuals('Random Forest')
plot_feature_importance('Random Forest')
print_evaluation_report()
