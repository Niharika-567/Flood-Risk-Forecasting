# 🌊 Flood Risk Forecasting & Early Warning System

An end-to-end deep learning web application that predicts hydrological flood risks using historical time-series data and an LSTM neural network. 

🔗 **Live App:** [View Live Streamlit Dashboard] (flood-risk-forecasting-htktg6vwynyt6zcsgbkh6y)

---

## 🚀 Key Features
* **LSTM Deep Learning Model:** Processes multi-feature temporal sequences to forecast accurate flood probabilities.
* **Interactive Timeline Mapping:** Translates day index sliders directly into real calendar dates ranging from **2014 to 2020**.
* **Dynamic Early Warning Thresholds:** Real-time risk probability scoring with automated alert banners for safe versus high-risk classifications.
* **Streamlit Web Dashboard:** Fully deployed and accessible online for instant simulation and user interaction.

---

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **Deep Learning:** TensorFlow / Keras (`flood_model.keras`)
* **Data Processing:** Pandas, NumPy
* **Frontend & Deployment:** Streamlit Community Cloud

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repository
git clone [https://github.com/niharika-567/flood-risk-forecasting.git](https://github.com/niharika-567/flood-risk-forecasting.git)
cd flood-risk-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run app.py
