import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data/emails.csv")

# Inputs and labels
X = df["text_combined"]
y = df["label"]

# Convert text to numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.9
)

X_vec = vectorizer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save trained model
with open("model/phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
