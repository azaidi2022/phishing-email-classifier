import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------
# Load trained model artifacts
# ---------------------------

with open("model/phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------------------------
# Initialize FastAPI app
# ---------------------------

app = FastAPI(title="Phishing Email Classifier API")

# ---------------------------
# Define input schema
# ---------------------------

class EmailInput(BaseModel):
    text: str

# ---------------------------
# Prediction endpoint
# ---------------------------

@app.post("/classify_email")
def classify_email(input: EmailInput):
    # Convert text to numerical vector
    text_vector = vectorizer.transform([input.text])

    # Make prediction
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0].max()

    return {
        "prediction": "phishing" if prediction == 1 else "legitimate",
        "confidence": round(float(probability), 3)
    }
