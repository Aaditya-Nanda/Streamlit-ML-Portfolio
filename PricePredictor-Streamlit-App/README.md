# Price Predictor Streamlit App

A simple machine learning web application that predicts house prices based on their size. This project is part of the [Streamlit-ML-Portfolio](https://github.com/Aaditya-Nanda/Streamlit-ML-Portfolio).

## Description

This application provides a hands-on demonstration of a complete, albeit simple, machine learning workflow. It features an interactive user interface built with **Streamlit** where users can input the size of a house and receive an estimated price prediction from a trained **Linear Regression** model.

The app also visualizes the prediction on a scatter plot alongside the original training data, offering a clear view of how the model's estimate compares to the dataset.

##  Features

- **Interactive UI:** A clean and simple user interface for entering house size.
- **Price Prediction:** Uses a `scikit-learn` Linear Regression model to predict prices in real-time.
- **Data Visualization:** Displays an interactive scatter plot using **Plotly** to show the relationship between house size and price, highlighting the user's prediction.

## 🛠️ Technologies Used

This project is built entirely in Python and relies on the following libraries:

- **Streamlit:** For creating the interactive web application UI.
- **scikit-learn:** For training the Linear Regression model.
- **Pandas & NumPy:** For data manipulation and generation.
- **Plotly:** For creating interactive data visualizations.

## Setup and Installation

To run this application on your local machine, please follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/Aaditya-Nanda/Streamlit-ML-Portfolio.git](https://github.com/Aaditya-Nanda/Streamlit-ML-Portfolio.git)
cd Streamlit-ML-Portfolio/PricePredictor-Streamlit-App
```
# Create a virtual environment
```
python -m venv venv
```

# Activate the environment (use the command for your OS)
# On Windows:
```
.\venv\Scripts\activate
```
# On macOS/Linux:
```
# source venv/bin/activate
```
# Install the required packages
```
pip install -r requirements.txt
```
# Run the Streamlit app
```
streamlit run app.py
```


