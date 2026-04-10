# House Price Predictor

A machine learning project to predict house prices using Python, scikit-learn, and Streamlit. This project includes data preprocessing, model training, evaluation, and a web app for predictions and visualizations.

## Features
- Data cleaning and preprocessing
- Model training and evaluation
- Streamlit web app for predictions
- Visualization of results and investment analysis

## Project Structure
```
├── 2_train.py           # Model training script
├── 3_evaluate.py        # Model evaluation script
├── app.py               # Streamlit web application
├── preprocess.py        # Data preprocessing
├── requirements.txt     # Python dependencies
├── data/                # Raw and cleaned datasets
├── models/              # Saved model files
├── outputs/             # Generated plots and evaluation results
├── pages/               # Streamlit multipage scripts
```

## Setup
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd house-price-predictor
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
  ```bash
  python preprocess.py
  ```
  ```bash
  python 2_train.py
  ```
  ```bash
  python 3_evaluate.py
  ```
  ```bash
  streamlit run app.py
  ```

Made with ❤️ by Archismaanshreyas

