# 📈 AI-Powered Stock Price Forecasting System

An **end-to-end stock forecasting system** combining traditional time-series analysis with advanced Machine Learning and Deep Learning models.  
The project predicts future stock prices using **ARIMA**, **Prophet**, **Linear Regression**, **Random Forest**, **XGBoost**, **LSTM**, and **GRU**,  
all integrated into an interactive **Streamlit web application** for real-time visualization and forecasting.

---

## 🧭 Overview

This project provides a **comprehensive approach to stock market forecasting**:
- Fetches **live market data** using Yahoo Finance.
- Performs **exploratory data analysis (EDA)** and **trend visualization**.
- Implements **multiple forecasting techniques**:
  - 📊 *Statistical Models*: ARIMA, Prophet  
  - 🤖 *Machine Learning*: Linear Regression, Random Forest, XGBoost  
  - 🧠 *Deep Learning*: LSTM, GRU
- Evaluates models using **MAE**, **RMSE**, and **R²** metrics.
- Highlights the **best performing model** and provides business insights.

---

## 🧩 Project Structure

```
📦 stock_price_prediction/
├── 📜 stock_forecasting.py             # Streamlit Web App
├── 📓 stock_forecasting_colab.ipynb    # Model training and analysis notebook
├── 📄 requirements.txt                 # Python dependencies
├── 📘 README.md                        # Project documentation
└── 📂 models/                          # (Optional) Trained model files
```

---

## ⚙️ Technologies Used

| Category | Technologies |
|-----------|---------------|
| **Frontend / UI** | Streamlit, Plotly |
| **Data Source** | Yahoo Finance (via `yfinance`) |
| **Time-Series Models** | Prophet, ARIMA |
| **Machine Learning** | Linear Regression, Random Forest, XGBoost |
| **Deep Learning** | TensorFlow / Keras (LSTM, GRU) |
| **Metrics** | MAE, RMSE, R² |

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/omrohitchannoji/stock_price_prediction.git
cd stock_price_prediction
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run stock_forecasting.py
```

💡 Once it starts, open your browser and go to:
```
http://localhost:8501
```

---

## 🧮 Model Training (Colab Notebook)

Use `stock_forecasting_colab.ipynb` to:
- Perform detailed **data cleaning, feature engineering, and model training**.
- Train and compare **ARIMA, Prophet, ML, and DL models**.
- Save trained models (`.h5`, `.pkl`) for deployment.

Open the notebook in Google Colab:  
[🔗 Open in Colab](https://colab.research.google.com)

---

## 🖥️ Streamlit Dashboard Overview

The web app provides **6 interactive tabs**:

| Tab | Description |
|------|-------------|
| **📊 EDA** | Explore stock trends, moving averages, and volatility |
| **🔮 Prophet Forecast** | Nonlinear forecasting with trend and seasonality |
| **📈 ARIMA Forecast** | Classic time-series forecasting for stationary data |
| **🤖 ML Models** | Predictive models using Linear Regression, Random Forest, and XGBoost |
| **🧠 Deep Learning** | Sequential modeling with LSTM and GRU networks |
| **📋 Model Comparison** | Performance summary and best model recommendation |

---

## 📊 Model Evaluation Metrics

| Model | MAE ↓ | RMSE ↓ | R² ↑ | Type |
|--------|--------|--------|------|------|
| Prophet | — | — | — | Time-Series |
| ARIMA | ✅ | ✅ | ✅ | Statistical |
| Linear Regression | ✅ | ✅ | ✅ | ML |
| Random Forest | ✅ | ✅ | ✅ | ML |
| XGBoost | ✅ | ✅ | ✅ | ML |
| LSTM | ✅ | ✅ | ✅ | Deep Learning |
| GRU | ✅ | ✅ | ✅ | Deep Learning |

---

## 📈 Visual Insights

- Real-time stock data visualization
- Forecast comparison across models
- Highlight of best performing model (lowest RMSE)
- Insightful interpretation of each model’s performance

---

## ☁️ Deployment

### 🌐 Streamlit Cloud
1. Push this repository to GitHub.
2. Visit [https://share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository.
4. Select `stock_forecasting.py` as the main file.
5. Deploy 🚀  

Your app will be live at:  
```
https://yourusername-stock-forecasting.streamlit.app/
```

### 🤗 Hugging Face Spaces
Alternatively, deploy easily using **Hugging Face Spaces**:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create a new Space → Select **Streamlit**
3. Upload your project files
4. Done ✅

---

## 💡 Future Enhancements
🔹 Integrate **sentiment analysis** from financial news  
🔹 Add **real-time live stock updates**  
🔹 Implement **Reinforcement Learning-based trading signals**  
🔹 Support **multi-stock portfolio comparison**  

---

## 🧑‍💻 Author

**Omrohit Channoji**  
📍 Data Science & AI Enthusiast  
🔗 [LinkedIn](https://www.linkedin.com/in/yourprofile)  
🐙 [GitHub](https://github.com/yourusername)

---

## 🏆 Acknowledgements

- [Yahoo Finance API](https://pypi.org/project/yfinance/)  
- [Prophet (Meta)](https://facebook.github.io/prophet/)  
- [Streamlit](https://streamlit.io)  
- [TensorFlow/Keras](https://www.tensorflow.org/)  
- [XGBoost](https://xgboost.readthedocs.io/)  

---

⭐ **If you found this project useful, don’t forget to star the repo!**
