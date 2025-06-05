# AI-Powered Data Analysis Dashboard

An intelligent data analysis and machine learning dashboard built with Streamlit that provides automated data cleaning, analysis, and predictive modeling capabilities.

## Live Demo

Check out the live application at: [AI Data Analysis Dashboard](https://model-check.streamlit.app)

## Features

- 📊 Interactive data upload and cleaning
- 🔍 Automated data preprocessing
- 🤖 Smart model selection and training
- 📈 Comprehensive data visualization
- 🎯 Prediction interface with visual analytics

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
├── app.py                 # Main Streamlit application
├── utils/
│   ├── data_cleaner.py   # Data cleaning functions
│   ├── preprocessor.py   # Data preprocessing functions
│   └── visualizer.py     # Visualization functions
├── models/
│   └── model_trainer.py  # ML model training and selection
└── requirements.txt      # Project dependencies
```

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies 