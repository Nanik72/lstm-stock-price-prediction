📈 LSTM Stock Price Prediction
This project uses a Long Short-Term Memory (LSTM) neural network to predict future stock prices based on historical closing prices. It takes the past 100 days of stock data and learns patterns to forecast the next day’s price.
The goal of this project is to understand how deep learning models can be applied to time-series problems like stock market forecasting.

🚀 Project Overview
Predict future stock closing prices using deep learning.
Use LSTM layers to learn long-term trends in historical data.
Prepare training data using a sliding window approach.
Train the model using MinMax scaling, dropout layers, and multiple LSTM layers.
Evaluate predictions by comparing them with actual prices.
This project is built as a Jupyter Notebook so you can run all steps step-by-step.

🧰 Technologies Used
Technology	           Purpose
Python	               Main programming language
Pandas	               Data handling, cleaning, train/test split
NumPy                  Numerical operations, array reshaping
Matplotlib	           Visualization of price trends and predictions
Scikit-learn	         MinMaxScaler for normalization
TensorFlow / Keras	   Building and training the LSTM deep learning model

🧠 How It Works
1️⃣ Data Preparation
    Load the stock dataset (CSV).
    Select the Close price column.
    Split into 70% training and 30% testing.
    Normalize the prices between 0 and 1 using MinMaxScaler.

2️⃣ Sequence Generation (Sliding Window)
    The model learns using a window of 100 previous days:
    Past 100 days  →  predict next day
    Example:
    Days 1–100 → Predict day 101
    Days 2–101 → Predict day 102
    This allows the model to understand time patterns.
3️⃣ Model Architecture
    Your LSTM model consists of multiple stacked layers:
    LSTM (50 units)
    LSTM (60 units)
    LSTM (80 units)
    LSTM (120 units)
    Dropout layers to prevent overfitting
    Dense layer (1 unit) to predict a single next-day closing price
    The model is compiled using:
    Optimizer: Adam
    Loss: Mean Squared Error (MSE)
    Metric: Mean Absolute Error (MAE)
4️⃣ Training
    The model is trained on:
    x_train – sequences of 100 days
    y_train – next-day values
    Validation is done on (x_test, y_test) to measure prediction accuracy.
5️⃣ Testing & Visualization
    After training:
    The model makes predictions on test data.
    Predictions are inverse-transformed back to actual price scale.
    A graph compares Actual Price vs Predicted Price to visualize performance.

📊 Results
The model typically achieves:
MAE around 9 (depends on stock volatility)
Smooth predictions that follow general market trends
Good trend capture but may lag during sudden movements
This performance is normal for single-feature LSTM stock models.

📚 Project Structure
├── LSTM_model.ipynb        # Main Jupyter Notebook
├── keras_model.h5          # Saved LSTM model
├── data/                   # Folder containing stock CSV files
└── README.md               # Project documentation

📥 How to Run
Install required libraries:
pip install numpy pandas matplotlib scikit-learn tensorflow
Open the notebook:
jupyter notebook LSTM_model.ipynb
Run all cells step-by-step.

🌱 Future Improvements
Add more features (Open, High, Low, Volume)
Predict next 5 days instead of 1 day
Use GRU or Transformer models
Hyperparameter tuning
Add sentiment analysis from news or Twitter

❤️ Acknowledgements
Thanks to the open-source community for providing tools like TensorFlow, Pandas, and Matplotlib that made this project possible.
