import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Union, Optional
import joblib

class ModelTrainer:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.regression_models = {}
        self.classification_models = {}
        self.problem_type = self._detect_problem_type()

    def _detect_problem_type(self) -> str:
        """Detect if the problem is regression or classification."""
        # Check if the target variable is numeric and has many unique values (regression) or few (classification)
        if pd.api.types.is_numeric_dtype(self.y):
             unique_values = self.y.nunique()
             # Heuristic: if unique values are few, classify. Otherwise, assume regression for numeric data.
             if unique_values <= 20: # Increased threshold for classification
                 return 'classification'
             return 'regression'
        else:
             # Non-numeric target is typically classification
             return 'classification'

    def train_regression_models(self) -> Dict[str, Dict[str, float]]:
        """Train specified regression models and return their metrics."""
        models = {
            'Linear Regression': LinearRegression(),
            'Lasso': Lasso(alpha=1.0), # Default alpha
            'Ridge': Ridge(alpha=1.0), # Default alpha
            'Polynomial Regression (degree 2)': make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        }
        metrics = {}

        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            self.regression_models[name] = model

            y_pred = model.predict(self.X_test)
            metrics[name] = {
                'R2 Score': r2_score(self.y_test, y_pred),
                'MSE': mean_squared_error(self.y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred))
            }
        return metrics

    def train_classification_model(self) -> Dict[str, Union[float, str, np.ndarray]]:
        """Train Logistic Regression and return its metrics."""
        model = LogisticRegression(random_state=42)
        model.fit(self.X_train, self.y_train)
        self.classification_models['Logistic Regression'] = model

        y_pred = model.predict(self.X_test)
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Classification Report': classification_report(self.y_test, y_pred),
            'Confusion Matrix': confusion_matrix(self.y_test, y_pred)
        }
        return metrics

    def evaluate_model(self, model_name: str, model_type: str) -> Dict:
        """Evaluate a specific model and return metrics."""
        if model_type == 'regression':
            model = self.regression_models[model_name]
            y_pred = model.predict(self.X_test)
            metrics = {
                'R2 Score': r2_score(self.y_test, y_pred),
                'MSE': mean_squared_error(self.y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred))
            }
            if hasattr(model, 'coef_'): # Linear models
                 metrics['Coefficients'] = model.coef_
            elif hasattr(model.named_steps.get('linearregression'), 'coef_'): # Pipeline with LinearRegression
                 metrics['Coefficients'] = model.named_steps['linearregression'].coef_

        elif model_type == 'classification':
            model = self.classification_models[model_name]
            y_pred = model.predict(self.X_test)
            metrics = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Classification Report': classification_report(self.y_test, y_pred),
                'Confusion Matrix': confusion_matrix(self.y_test, y_pred)
            }
        else:
             raise ValueError("Invalid model type specified.")

        return metrics

    def create_confusion_matrix_plot(self, model_name: str) -> go.Figure:
        """Create a confusion matrix plot for classification models."""
        if self.problem_type != 'classification' or model_name not in self.classification_models:
            raise ValueError("Confusion matrix is only available for trained classification models")

        model = self.classification_models[model_name]
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        labels = sorted(list(self.y.unique()))

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f'Predicted {l}' for l in labels],
            y=[f'Actual {l}' for l in labels],
            colorscale='Blues',
            text=cm, # Add text annotations
            texttemplate="%{text}",
            textfont={"size":10}
        ))

        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            height=400
        )

        return fig

    def create_prediction_plot(self, model_name: str, model_type: str) -> go.Figure:
        """Create a plot comparing actual vs predicted values or probability distribution."""
        if model_type == 'regression':
            if model_name not in self.regression_models:
                 raise ValueError(f"Regression model {model_name} not trained.")
            model = self.regression_models[model_name]
            y_pred = model.predict(self.X_test)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=self.y_test,
                y=y_pred,
                mode='markers',
                name='Predictions'
            ))

            # Add perfect prediction line
            min_val = min(min(self.y_test), min(y_pred)) if len(self.y_test) > 0 and len(y_pred) > 0 else 0
            max_val = max(max(self.y_test), max(y_pred)) if len(self.y_test) > 0 and len(y_pred) > 0 else 1
            if min_val != max_val:
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash')
                ))

            fig.update_layout(
                title=f'Actual vs Predicted - {model_name}',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values',
                height=500
            )

        elif model_type == 'classification':
            if model_name not in self.classification_models:
                 raise ValueError(f"Classification model {model_name} not trained.")
            model = self.classification_models[model_name]
            fig = go.Figure()

            # For classification, show probability distribution for binary classification
            if hasattr(model, 'predict_proba') and len(model.classes_) == 2:
                y_prob = model.predict_proba(self.X_test)[:, 1]
                fig.add_trace(go.Histogram(
                    x=y_prob,
                    name='Prediction Probabilities'
                ))

                fig.update_layout(
                    title=f'Prediction Probabilities - {model_name}',
                    xaxis_title='Probability of Positive Class',
                    yaxis_title='Count',
                    height=500
                )
            else:
                # For multi-class or models without predict_proba, show predicted vs actual
                y_pred = model.predict(self.X_test)
                pred_df = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_pred})
                fig = px.scatter(pred_df, x='Actual', y='Predicted', title=f'Actual vs Predicted - {model_name}')

        else:
            raise ValueError("Invalid model type specified.")

        return fig

    def save_model(self, model_name: str, model_type: str, path: str):
        """Save a trained model to disk."""
        if model_type == 'regression':
            model = self.regression_models[model_name]
        elif model_type == 'classification':
            model = self.classification_models[model_name]
        else:
             raise ValueError("Invalid model type specified.")
        joblib.dump(model, path)

    def load_model(self, model_name: str, model_type: str, path: str):
        """Load a trained model from disk."""
        model = joblib.load(path)
        if model_type == 'regression':
            self.regression_models[model_name] = model
        elif model_type == 'classification':
            self.classification_models[model_name] = model
        else:
             raise ValueError("Invalid model type specified.")

    def get_feature_importance(self, model_name: str, model_type: str) -> pd.DataFrame:
        """Get feature importance for tree-based models."""
        if model_type == 'regression' and model_name in self.regression_models:
             model = self.regression_models[model_name]
        elif model_type == 'classification' and model_name in self.classification_models:
             model = self.classification_models[model_name]
        else:
            raise ValueError("Model not found or invalid type.")

        # Handle pipelines containing a model with feature_importances_
        if isinstance(model, Pipeline) and hasattr(model.steps[-1][1], 'feature_importances_'):
             importance = model.steps[-1][1].feature_importances_
             features = pd.DataFrame({
                'Feature': self.X.columns,
                'Importance': importance
             })
             return features.sort_values('Importance', ascending=False)
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            features = pd.DataFrame({
                'Feature': self.X.columns,
                'Importance': importance
            })
            return features.sort_values('Importance', ascending=False)
        else:
            raise ValueError("Model does not support feature importance") 