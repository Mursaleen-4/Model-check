import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder
)
from typing import Dict, List, Tuple, Union
import joblib

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = df.select_dtypes(include=['object']).columns
        self.encoders = {}
        self.scalers = {}
        self.preprocessing_history = []

    def detect_column_types(self) -> Dict[str, List[str]]:
        """Detect and categorize columns by their data type."""
        return {
            'numeric': list(self.numeric_columns),
            'categorical': list(self.categorical_columns)
        }

    def encode_categorical(self, method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical variables using the specified method.
        
        Args:
            method: 'onehot' or 'label'
        """
        if method == 'onehot':
            for col in self.categorical_columns:
                self.df[col] = self.df[col].astype(str)  # Convert to string
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_data = encoder.fit_transform(self.df[[col]])
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=[f"{col}_{cat}" for cat in encoder.categories_[0]],
                    index=self.df.index
                )
                self.df = pd.concat([self.df.drop(col, axis=1), encoded_df], axis=1)
                self.encoders[col] = encoder
                
        elif method == 'label':
            for col in self.categorical_columns:
                self.df[col] = self.df[col].astype(str)  # Convert to string
                encoder = LabelEncoder()
                self.df[col] = encoder.fit_transform(self.df[col])
                self.encoders[col] = encoder

        self.preprocessing_history.append(f"Encoded categorical variables using {method} encoding")
        return self.df

    def normalize_numeric(self, method: str = 'standard') -> pd.DataFrame:
        """
        Normalize numeric variables using the specified method.
        
        Args:
            method: 'standard', 'minmax', or 'robust'
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid normalization method")

        self.df[self.numeric_columns] = scaler.fit_transform(self.df[self.numeric_columns])
        self.scalers['numeric'] = scaler
        
        self.preprocessing_history.append(f"Normalized numeric variables using {method} scaling")
        return self.df

    def save_preprocessors(self, path: str):
        """Save the preprocessors to disk."""
        preprocessors = {
            'encoders': self.encoders,
            'scalers': self.scalers
        }
        joblib.dump(preprocessors, path)

    def load_preprocessors(self, path: str):
        """Load the preprocessors from disk."""
        preprocessors = joblib.load(path)
        self.encoders = preprocessors['encoders']
        self.scalers = preprocessors['scalers']

    def get_preprocessing_summary(self) -> Dict:
        """Get a summary of all preprocessing operations performed."""
        return {
            'numeric_columns': list(self.numeric_columns),
            'categorical_columns': list(self.categorical_columns),
            'preprocessing_history': self.preprocessing_history
        }

    def preprocess_data(self,
                       encode_categorical: bool = True,
                       normalize_numeric: bool = True,
                       encoding_method: str = 'onehot',
                       normalization_method: str = 'standard') -> pd.DataFrame:
        """
        Perform all preprocessing operations in sequence.
        
        Args:
            encode_categorical: Whether to encode categorical variables
            normalize_numeric: Whether to normalize numeric variables
            encoding_method: Method for categorical encoding
            normalization_method: Method for numeric normalization
        """
        if encode_categorical:
            self.encode_categorical(method=encoding_method)
        if normalize_numeric:
            self.normalize_numeric(method=normalization_method)
            
        return self.df 