import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_history = []

    def get_missing_values_info(self) -> Dict:
        """Get information about missing values in the dataset."""
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        return {
            'count': missing_values.to_dict(),
            'percentage': missing_percentage.to_dict()
        }

    def handle_missing_values(self, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values based on the specified strategy.
        
        Args:
            strategy: 'auto', 'drop', 'mean', 'median', 'mode', or 'zero'
        """
        if strategy == 'auto':
            # For numeric columns, use mean
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
            
            # For categorical columns, use mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        elif strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'mean':
            self.df = self.df.fillna(self.df.mean())
        elif strategy == 'median':
            self.df = self.df.fillna(self.df.median())
        elif strategy == 'mode':
            self.df = self.df.fillna(self.df.mode().iloc[0])
        elif strategy == 'zero':
            self.df = self.df.fillna(0)
            
        self.cleaning_history.append(f"Handled missing values using {strategy} strategy")
        return self.df

    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows from the dataset."""
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed_rows = initial_rows - len(self.df)
        self.cleaning_history.append(f"Removed {removed_rows} duplicate rows")
        return self.df

    def convert_data_types(self) -> pd.DataFrame:
        """Convert data types to appropriate formats."""
        # Convert numeric columns to appropriate types
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self.df[col].dtype == np.float64:
                if (self.df[col] % 1 == 0).all():
                    self.df[col] = self.df[col].astype(np.int64)
        
        # Convert date columns
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                except:
                    pass
        
        self.cleaning_history.append("Converted data types to appropriate formats")
        return self.df

    def get_cleaning_summary(self) -> Dict:
        """Get a summary of all cleaning operations performed."""
        return {
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'cleaning_history': self.cleaning_history
        }

    def clean_data(self, 
                  handle_missing: bool = True,
                  remove_duplicates: bool = True,
                  convert_types: bool = True) -> pd.DataFrame:
        """
        Perform all cleaning operations in sequence.
        
        Args:
            handle_missing: Whether to handle missing values
            remove_duplicates: Whether to remove duplicates
            convert_types: Whether to convert data types
        """
        if handle_missing:
            self.handle_missing_values()
        if remove_duplicates:
            self.remove_duplicates()
        if convert_types:
            self.convert_data_types()
            
        return self.df 