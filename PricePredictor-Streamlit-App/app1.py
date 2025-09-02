import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# --- 1. Data Generation ---
# This function creates a sample dataset of house sizes and prices.
def generate_house_data(n_samples=100):
    np.random.seed(50)
    size = np.random.normal(1400, 50, n_samples)
    price = size * 50 + np.random.normal(0, 50, n_samples)
    return pd.DataFrame({'size': size, 'price': price})

# --- 2. Model Training ---
# This function trains a linear regression model on the generated data.
def train_model():
    df = generate_house_data(n_samples=100)
    X = df[['size']]  # Features (must be 2D)
    y = df['price']   # Target variable

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# --- 3. Streamlit App ---
st.title("House Price Prediction App")
st.write("Enter a house size to predict its price.")

# Train the model
model = train_model()

# User input for house size
size = st.number_input('House size (in sq. ft.)', min_value=500, max_value=1560, value=1500)

# Prediction button
if st.button('Predict Price'):
    # Predict the price using the trained model
    predicted_price = model.predict([[size]])
    
    # Display the prediction in a formatted success message
    st.success(f'Estimated Price: ${predicted_price[0]:,.2f}')

    # --- 4. Data Visualization ---
    # Generate the original dataset for plotting
    df = generate_house_data()

    # Create a scatter plot of the original data
    fig = px.scatter(df, x="size", y="price", title="Size vs. House Price")
    
    # Add the new prediction as a red marker to the plot
    fig.add_scatter(
        x=[size], 
        y=[predicted_price[0]],
        mode='markers',
        marker=dict(size=15, color='red'),
        name='Your Prediction'
    )
    
    # Display the interactive plot in the app
    st.plotly_chart(fig)