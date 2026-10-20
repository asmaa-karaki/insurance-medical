# 💙 Insurance Cost Estimator

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?logo=plotly&logoColor=white)

A Streamlit web app that predicts **yearly medical insurance charges** from a person's profile (age, sex, BMI, children, smoker status, region) and compares the result with a reference average.

## 🎬 Demo

![App screenshot](screenshot.png)

<!-- Once deployed, add the link here: -->
<!-- **Live app:** https://your-app-name.streamlit.app -->

## ✨ Features

- 🎛️ Interactive sidebar form with sliders and dropdowns
- 🔮 Instant cost prediction from a trained machine learning model
- 🎨 Colorful result cards: estimate, reference average and gap in %
- 🚦 Cost level badge (low / moderate / high)
- 📊 Bar chart comparing your estimate with the average smoker / non-smoker cost
- ⚖️ BMI gauge with color zones (underweight, normal, overweight, obese)
- 📝 Summary of the profile used for the estimate

## 🛠️ Tech stack

- Python 3.10+
- Streamlit
- scikit-learn / joblib
- pandas, NumPy
- Plotly

## 📁 Project structure

```
.
├── app.py                  # Streamlit app
├── coding1.ipynb           # Data exploration and model training
├── medical_insurance.csv   # Dataset
├── insurance_model.pkl     # Trained model
├── model_features.pkl      # Feature columns expected by the model
├── model_name.pkl          # Name of the selected model
├── requirements.txt
└── README.md
```

## 🚀 Run locally

```bash
# 1. Clone the repository
git clone https://github.com/asmaa-karaki/insurance-medical.git
cd insurance-medical

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The app opens at http://localhost:8501.

## 🤖 Model

- **Target:** insurance charges. The model is trained on `log1p(charges)`, so predictions are converted back with `expm1`.
- **Inputs:** age, sex, BMI (plus BMI category), number of children, smoker, region
- **Dataset:** `medical_insurance.csv` — [add the source, e.g. the Kaggle "Medical Cost Personal Datasets"]
- **Algorithm:** [e.g. Random Forest / Gradient Boosting — see `coding1.ipynb`]
- **Performance:** [e.g. R² = 0.87 on the test set — see `coding1.ipynb`]

## ⚠️ Disclaimer

The average costs shown in the app are illustrative reference values. This tool is for learning purposes and does not provide a personalised insurance quote.

## 👩‍💻 Author

**Asmaa Karaki** – [github.com/asmaa-karaki](https://github.com/asmaa-karaki)
