import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add custom modules to path
sys.path.append(str(Path(__file__).parent))

from utils.data_cleaner import DataCleaner
from utils.preprocessor import DataPreprocessor
from utils.visualizer import DataVisualizer
from models.model_trainer import ModelTrainer

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better buttons and headings
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: transparent;
        color: #fff;
        border: 2px solid;
        border-image: linear-gradient(90deg, #ff00c1, #00ffff, #ffff00, #ff00c1) 1;
        border-radius: 30px;
        padding: 0.5em 1.5em;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 1.2em 0 1.2em 0;
        box-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff, inset 0 0 10px #00ffff;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    .stButton>button:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, rgba(255,0,193,0.5), rgba(0,255,255,0.5), rgba(255,255,0,0.5), rgba(255,0,193,0.5));
        z-index: -1;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .stButton>button:hover {
        color: #fff;
        box-shadow: 0 0 20px #00ffff, 0 0 30px #00ffff, 0 0 40px #00ffff, inset 0 0 20px #00ffff;
    }
    .stButton>button:hover:before {
         opacity: 1;
    }
    .stCheckbox>div {
        font-size: 1.05rem;
        font-weight: 500;
    }
    .stSelectbox>div {
        font-size: 1.05rem;
        font-weight: 500;
    }
    /* Remove default Streamlit tab styling */
    .stTabs [data-baseweb="tab-list"] {
        display: none;
    }
    /* Style for top navigation container */
    .top-nav-container {
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📊 AI-Powered Data Analysis Dashboard")
st.markdown("""
    Upload your dataset and let our AI analyze it for you! This dashboard provides:
    - Automated data cleaning and preprocessing
    - Smart model selection and training
    - Interactive visualizations
    - Prediction capabilities
""")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaner' not in st.session_state:
    st.session_state.cleaner = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = None

# Sidebar (only for file upload)
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.data = df
            st.session_state.cleaner = DataCleaner(df)
            st.session_state.preprocessor = DataPreprocessor(df)
            st.session_state.visualizer = DataVisualizer(df)

            st.success("File uploaded successfully!")

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Main content area
if st.session_state.data is not None:

    # Top Navigation Selectbox
    sections = ["Data Overview", "Data Cleaning", "Data Preprocessing", "Model Training"]
    selected_section = st.selectbox(
        "Jump to section:",
        sections,
        key='section_selector' # Add a key to the selectbox
    )

    # --- Section Content (Conditional Rendering) ---

    if selected_section == "Data Overview":
        st.markdown("## 📈 Data Overview")
        st.markdown("---")

        # Basic information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Rows", st.session_state.data.shape[0])
        with col2:
            st.metric("Number of Columns", st.session_state.data.shape[1])
        with col3:
            st.metric("Memory Usage", f"{st.session_state.data.memory_usage().sum() / 1024:.2f} KB")

        # Data preview
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head())

        # Basic statistics
        st.subheader("Basic Statistics")
        st.dataframe(st.session_state.data.describe())

        # Improved Visualizations
        st.subheader("Key Data Visualizations")
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = st.session_state.data.select_dtypes(include=['object', 'category']).columns.tolist()

        if numeric_cols:
            st.markdown("**Distribution of Numeric Columns**")
            st.caption("These histograms show how your numeric data is distributed. Peaks indicate common values, while spread shows variability.")
            for col in numeric_cols[:3]:
                st.plotly_chart(
                    px.histogram(st.session_state.data, x=col, nbins=20, title=f"Distribution of {col}"),
                    use_container_width=True
                )
        if len(numeric_cols) > 1:
            st.markdown("**Correlation Heatmap**")
            st.caption("This heatmap shows how strongly numeric columns are related. Values close to 1 or -1 mean strong relationships.")
            st.plotly_chart(
                st.session_state.visualizer.create_correlation_heatmap(),
                use_container_width=True
            )
        if categorical_cols:
            st.markdown("**Top Categories in Categorical Columns**")
            st.caption("These bar charts show the most common values in your categorical columns.")
            for col in categorical_cols[:2]:
                vc = st.session_state.data[col].value_counts().head(10)
                st.plotly_chart(
                    px.bar(x=vc.index, y=vc.values, labels={'x': col, 'y': 'Count'}, title=f"Top Values in {col}"),
                    use_container_width=True
                )

    elif selected_section == "Data Cleaning":
        st.markdown("## 🧹 Data Cleaning")
        st.markdown("---")

        # Missing values information
        missing_info = st.session_state.cleaner.get_missing_values_info()
        st.subheader("Missing Values Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.write("Missing Values Count")
            st.dataframe(pd.DataFrame(missing_info['count'], index=['Count']).T)
        with col2:
            st.write("Missing Values Percentage")
            st.dataframe(pd.DataFrame(missing_info['percentage'], index=['Percentage']).T)

        # Cleaning options
        st.subheader("Cleaning Options")

        col1, col2, col3 = st.columns(3)
        with col1:
            handle_missing = st.checkbox("Handle Missing Values", value=True)
        with col2:
            remove_duplicates = st.checkbox("Remove Duplicates", value=True)
        with col3:
            convert_types = st.checkbox("Convert Data Types", value=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("Clean Data"):
            with st.spinner("Cleaning data..."):
                cleaned_df = st.session_state.cleaner.clean_data(
                    handle_missing=handle_missing,
                    remove_duplicates=remove_duplicates,
                    convert_types=convert_types
                )
                # Update session state
                st.session_state.data = cleaned_df
                st.session_state.visualizer = DataVisualizer(cleaned_df)
                st.success("Data cleaned successfully!")
                # Show cleaning summary
                summary = st.session_state.cleaner.get_cleaning_summary()
                st.write("Cleaning Summary:")
                for operation in summary['cleaning_history']:
                    st.write(f"- {operation}")

    elif selected_section == "Data Preprocessing":
        st.markdown("## 🔧 Data Preprocessing")
        st.markdown("---")

        # Column type detection
        column_types = st.session_state.preprocessor.detect_column_types()

        col1, col2 = st.columns(2)
        with col1:
            st.write("Numeric Columns")
            st.write(column_types['numeric'])
        with col2:
            st.write("Categorical Columns")
            st.write(column_types['categorical'])

        # Preprocessing options
        st.subheader("Preprocessing Options")

        col1, col2 = st.columns(2)
        with col1:
            encode_categorical = st.checkbox("Encode Categorical Variables", value=True)
            if encode_categorical:
                encoding_method = st.selectbox(
                    "Encoding Method",
                    ['onehot', 'label']
                )
        with col2:
            normalize_numeric = st.checkbox("Normalize Numeric Variables", value=True)
            if normalize_numeric:
                normalization_method = st.selectbox(
                    "Normalization Method",
                    ['standard', 'minmax', 'robust']
                )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                processed_df = st.session_state.preprocessor.preprocess_data(
                    encode_categorical=encode_categorical,
                    normalize_numeric=normalize_numeric,
                    encoding_method=encoding_method if encode_categorical else 'onehot',
                    normalization_method=normalization_method if normalize_numeric else 'standard'
                )

                # Update session state
                st.session_state.data = processed_df
                st.session_state.visualizer = DataVisualizer(processed_df)

                st.success("Data preprocessed successfully!")

                # Show preprocessing summary
                summary = st.session_state.preprocessor.get_preprocessing_summary()
                st.write("Preprocessing Summary:")
                for operation in summary['preprocessing_history']:
                    st.write(f"- {operation}")

    elif selected_section == "Model Training":
        st.markdown("## 🤖 Model Training")
        st.markdown("---")

        # Target variable selection
        target_col = st.selectbox(
            "Select Target Variable",
            st.session_state.data.columns
        )

        if st.button("Train Models"):
            with st.spinner("Training models..."):
                # Prepare data
                X = st.session_state.data.drop(columns=[target_col])
                X = X.select_dtypes(include=[np.number])
                y = st.session_state.data[target_col]
                # Drop rows where y is NaN
                not_nan_mask = ~y.isna()
                X = X.loc[not_nan_mask]
                y = y.loc[not_nan_mask]

                # Initialize and train models
                st.session_state.model_trainer = ModelTrainer(X, y)

                st.subheader("Model Training Results")

                if st.session_state.model_trainer.problem_type == 'regression':
                    st.markdown("### Regression Results")
                    regression_metrics = st.session_state.model_trainer.train_regression_models()

                    for model_name, metrics in regression_metrics.items():
                        st.write(f"**{model_name}**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R2 Score", f"{metrics['R2 Score']:.4f}")
                        with col2:
                            st.metric("MSE", f"{metrics['MSE']:.4f}")
                        with col3:
                            st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                        # Optional: Add prediction plot for each regression model
                        # try:
                        #     fig_pred = st.session_state.model_trainer.create_prediction_plot(model_name, 'regression')
                        #     st.plotly_chart(fig_pred, use_container_width=True)
                        # except Exception as e:
                        #      st.warning(f"Could not create prediction plot for {model_name}: {e}")

                elif st.session_state.model_trainer.problem_type == 'classification':
                    st.markdown("### Classification Results")
                    classification_metrics = st.session_state.model_trainer.train_classification_model()

                    st.write("**Logistic Regression**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{classification_metrics['Accuracy']:.4f}")
                    with col2:
                        st.write("Classification Report")
                        st.text(classification_metrics['Classification Report'])

                    st.write("Confusion Matrix")
                    st.plotly_chart(
                        st.session_state.model_trainer.create_confusion_matrix_plot('Logistic Regression'),
                        use_container_width=True
                    )
                    # Optional: Add prediction plot for classification (e.g., probability histogram)
                    # try:
                    #     fig_pred = st.session_state.model_trainer.create_prediction_plot('Logistic Regression', 'classification')
                    #     st.plotly_chart(fig_pred, use_container_width=True)
                    # except Exception as e:
                    #      st.warning(f"Could not create prediction plot for Logistic Regression: {e}")

                st.success("Models trained successfully!")

else:
    st.info("👆 Please upload a CSV or Excel file to begin analysis.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit") 