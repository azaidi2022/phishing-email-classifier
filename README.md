# Phishing Email Detection System

This project implements an AI-powered phishing email detection system that classifies emails as phishing or legitimate using Natural Language Processing (NLP) and supervised machine learning. The trained model is deployed as a FastAPI web service to provide real-time predictions through a REST API.

---

## Features

- Supervised phishing email classification
- Text feature extraction using TF-IDF
- Logistic Regression classifier
- FastAPI-powered REST API for real-time inference
- Interactive API documentation via Swagger UI

---

## Architecture

**High-level workflow:**

Email Text  
→ Text Vectorization (TF-IDF)  
→ Machine Learning Classifier  
→ Prediction (Phishing / Legitimate)

---

## Tech Stack

- Python
- FastAPI
- scikit-learn
- Pandas
- Git & GitHub

---

## API Usage

### POST `/classify_email`

**Request**
```json
{
  "text": "Urgent! Your account has been compromised. Click here to reset."
}

Response

{
  "prediction": "phishing",
  "confidence": 0.99
}

## Model Performance
The phishing classifier achieved approximately 98% accuracy, with high precision and recall for both phishing and legitimate email classes when evaluated on real-world email data.

## Dataset
This project uses a publicly available phishing email dataset derived from Enron-based email corpora.
Due to GitHub file size limitations, the dataset is not included in this repository and must be downloaded separately.

## Motivation
Phishing attacks remain one of the most common and effective cybersecurity threats. This project demonstrates how NLP and machine learning techniques can be applied to automatically detect malicious email content and support security-focused workflows.

## Future Improvements
- Add email metadata features such as sender, subject, and URLs
- Experiment with transformer-based text embeddings
- Implement model retraining and versioning
- Deploy the service to cloud infrastructure

Running the Project Locally
1. Clone the repository
2. Install dependencies from requirements.txt
3. Train the model using python model/train.py
4. Start the API server with uvicorn app:app --reload
5. Access interactive API docs at http://127.0.0.1:8000/docs