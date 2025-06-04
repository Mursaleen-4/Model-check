import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Optional

class DataVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = df.select_dtypes(include=['object']).columns

    def create_correlation_heatmap(self) -> go.Figure:
        """Create a correlation heatmap for numeric columns."""
        corr_matrix = self.df[self.numeric_columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title='Correlation Heatmap',
            xaxis_title='Features',
            yaxis_title='Features',
            height=600
        )
        
        return fig

    def create_distribution_plots(self, columns: Optional[List[str]] = None) -> go.Figure:
        """Create distribution plots for specified numeric columns."""
        if columns is None:
            columns = self.numeric_columns[:4]  # Default to first 4 numeric columns
            
        n_cols = len(columns)
        fig = make_subplots(rows=n_cols, cols=1, subplot_titles=columns)
        
        for i, col in enumerate(columns, 1):
            fig.add_trace(
                go.Histogram(x=self.df[col], name=col),
                row=i, col=1
            )
            
        fig.update_layout(
            title='Distribution of Numeric Features',
            height=300 * n_cols,
            showlegend=False
        )
        
        return fig

    def create_box_plots(self, columns: Optional[List[str]] = None) -> go.Figure:
        """Create box plots for specified numeric columns."""
        if columns is None:
            columns = self.numeric_columns[:4]  # Default to first 4 numeric columns
            
        fig = go.Figure()
        
        for col in columns:
            fig.add_trace(go.Box(
                y=self.df[col],
                name=col
            ))
            
        fig.update_layout(
            title='Box Plots of Numeric Features',
            yaxis_title='Value',
            height=500
        )
        
        return fig

    def create_scatter_matrix(self, columns: Optional[List[str]] = None) -> go.Figure:
        """Create a scatter plot matrix for specified numeric columns."""
        if columns is None:
            columns = self.numeric_columns[:4]  # Default to first 4 numeric columns
            
        fig = px.scatter_matrix(
            self.df,
            dimensions=columns,
            title='Scatter Plot Matrix'
        )
        
        fig.update_layout(height=800)
        return fig

    def create_categorical_plots(self, columns: Optional[List[str]] = None) -> go.Figure:
        """Create bar plots for categorical columns."""
        if columns is None:
            columns = self.categorical_columns[:4]  # Default to first 4 categorical columns
            
        n_cols = len(columns)
        fig = make_subplots(rows=n_cols, cols=1, subplot_titles=columns)
        
        for i, col in enumerate(columns, 1):
            value_counts = self.df[col].value_counts()
            fig.add_trace(
                go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    name=col
                ),
                row=i, col=1
            )
            
        fig.update_layout(
            title='Distribution of Categorical Features',
            height=300 * n_cols,
            showlegend=False
        )
        
        return fig

    def create_pair_plot(self, columns: Optional[List[str]] = None) -> go.Figure:
        """Create a pair plot for specified numeric columns."""
        if columns is None:
            columns = self.numeric_columns[:4]  # Default to first 4 numeric columns
            
        fig = px.scatter_matrix(
            self.df,
            dimensions=columns,
            title='Pair Plot'
        )
        
        fig.update_layout(height=800)
        return fig

    def create_time_series_plot(self, date_column: str, value_column: str) -> go.Figure:
        """Create a time series plot for a specific column."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.df[date_column],
            y=self.df[value_column],
            mode='lines+markers',
            name=value_column
        ))
        
        fig.update_layout(
            title=f'Time Series Plot: {value_column}',
            xaxis_title='Date',
            yaxis_title='Value',
            height=500
        )
        
        return fig

    def create_3d_scatter(self, x: str, y: str, z: str, color: Optional[str] = None) -> go.Figure:
        """Create a 3D scatter plot."""
        fig = px.scatter_3d(
            self.df,
            x=x,
            y=y,
            z=z,
            color=color,
            title=f'3D Scatter Plot: {x} vs {y} vs {z}'
        )
        
        fig.update_layout(height=800)
        return fig 